"""
Trajecto BLE Driver - Protocol-Compliant Implementation

This module implements the `TrajectoDriver` class for BLE communication with the
Trajecto hardware device. It follows the structured packet protocol defined in
firmware/components/trajecto_protocol/include/trajecto_protocol.h

The driver supports:
- Handshake on connection (Ping/Pong)
- Mode configuration (Raw IMU vs Trajectory)
- Calibration control (CRT + FOC)
- Dual streaming modes with proper packet parsing
"""

import asyncio
import struct
import sys
import time
from enum import IntEnum
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

from bleak import BleakClient, BleakScanner

# --- BLE Service and Characteristic UUIDs ---
SERVICE_UUID: str = "ad43434e-c549-4594-b474-543153544557"
DATA_CHAR_UUID: str = "ad43434f-c549-4594-b474-543153544557"  # Notify
CMD_CHAR_UUID: str = "ad43434d-c549-4594-b474-543153544557"   # Write
DEVICE_NAME: str = "Trajecto"


# --- Protocol Definitions (matching trajecto_protocol.h) ---

class PacketType(IntEnum):
    """Packet type identifiers matching firmware enum"""
    CMD_PING = 0x01
    RSP_PONG = 0x02

    CMD_SET_CONFIG = 0x10
    RSP_CONFIG_OK = 0x11
    CMD_GET_CONFIG = 0x12
    RSP_CONFIG = 0x13

    CMD_START_STREAM = 0x20
    RSP_STREAM_STARTED = 0x21
    CMD_STOP_STREAM = 0x22
    RSP_STREAM_STOPPED = 0x23

    CMD_CALIBRATE = 0x30
    RSP_CALIB_STATUS = 0x31

    DATA_RAW_IMU = 0x80
    DATA_TRAJECTORY = 0x81


@dataclass
class Header:
    """Packet header: type (1 byte) + length (1 byte)"""
    type: PacketType
    length: int

    @staticmethod
    def parse(data: bytes) -> Optional['Header']:
        if len(data) < 2:
            return None
        return Header(type=PacketType(data[0]), length=data[1])

    def pack(self) -> bytes:
        return struct.pack('<BB', self.type, self.length)


@dataclass
class ConfigPayload:
    """Configuration payload: mode, odr_hz, reserved[2]"""
    mode: int      # 0: Raw, 1: Trajectory
    odr_hz: int    # Sampling rate (fixed at 50Hz)
    reserved: tuple = (0, 0)

    @staticmethod
    def parse(data: bytes) -> Optional['ConfigPayload']:
        if len(data) < 4:
            return None
        unpacked = struct.unpack('<BBBB', data[:4])
        return ConfigPayload(
            mode=unpacked[0],
            odr_hz=unpacked[1],
            reserved=(unpacked[2], unpacked[3])
        )

    def pack(self) -> bytes:
        return struct.pack('<BBBB', self.mode, self.odr_hz,
                          self.reserved[0], self.reserved[1])


@dataclass
class RawImuPacket:
    """Raw IMU data packet"""
    timestamp_us: int
    accel: tuple  # (x, y, z) in m/s^2
    gyro: tuple   # (x, y, z) in rad/s
    force: int    # FSR reading
    temperature: float  # Temperature in °C

    @staticmethod
    def parse(data: bytes) -> Optional['RawImuPacket']:
        if len(data) < 34:  # 4 + 12 + 12 + 2 + 4
            return None
        unpacked = struct.unpack('<Iffffffhf', data[:34])
        return RawImuPacket(
            timestamp_us=unpacked[0],
            accel=(unpacked[1], unpacked[2], unpacked[3]),
            gyro=(unpacked[4], unpacked[5], unpacked[6]),
            force=unpacked[7],
            temperature=unpacked[8]
        )


@dataclass
class TrajectoryPacket:
    """Trajectory estimation packet (ESKF-TCN output)"""
    timestamp_us: int
    pos: tuple    # (x, y, z) in meters
    vel: tuple    # (x, y, z) in m/s
    quat: tuple   # (w, x, y, z) quaternion
    prob_zupt: float  # Zero-velocity probability

    @staticmethod
    def parse(data: bytes) -> Optional['TrajectoryPacket']:
        if len(data) < 48:  # 4 + 12 + 12 + 16 + 4
            return None
        unpacked = struct.unpack('<Iffffffffff', data[:48])
        return TrajectoryPacket(
            timestamp_us=unpacked[0],
            pos=(unpacked[1], unpacked[2], unpacked[3]),
            vel=(unpacked[4], unpacked[5], unpacked[6]),
            quat=(unpacked[7], unpacked[8], unpacked[9], unpacked[10]),
            prob_zupt=unpacked[11]
        )


# --- Driver Class ---

class TrajectoDriver:
    """
    BLE driver for Trajecto device with full protocol support.

    Features:
    - Automatic handshake on connection
    - Mode selection (Raw/Trajectory)
    - Runtime calibration trigger
    - Dual-mode data streaming
    """

    def __init__(
        self,
        device_name: str = DEVICE_NAME,
        raw_callback: Optional[Callable[[RawImuPacket], None]] = None,
        trajectory_callback: Optional[Callable[[TrajectoryPacket], None]] = None,
        verbose: bool = True
    ):
        """
        Initialize Trajecto BLE driver.

        Args:
            device_name: BLE device name to scan for
            raw_callback: Callback for raw IMU data packets
            trajectory_callback: Callback for trajectory packets
            verbose: Enable debug output
        """
        self.device_name = device_name
        self.client: Optional[BleakClient] = None
        self.verbose = verbose

        # Callbacks
        self.raw_callback = raw_callback
        self.trajectory_callback = trajectory_callback

        # Internal state
        self._connected_event = asyncio.Event()
        self._handshake_done = asyncio.Event()
        self._response_queue: asyncio.Queue = asyncio.Queue()

        # Current config
        self.current_config: Optional[ConfigPayload] = None
        self.streaming_mode: Optional[int] = None  # 0: Raw, 1: Trajectory

        # Data buffers (if no callbacks provided)
        self.raw_data: List[RawImuPacket] = []
        self.trajectory_data: List[TrajectoryPacket] = []

    def _log(self, msg: str):
        """Print log message if verbose enabled"""
        if self.verbose:
            print(f"[TrajectoDriver] {msg}")

    async def connect(self) -> bool:
        """
        Scan for and connect to Trajecto device.
        Performs handshake after connection.

        Returns:
            True if connection and handshake successful
        """
        self._log(f"Scanning for '{self.device_name}'...")
        device = await BleakScanner.find_device_by_name(self.device_name)

        if device is None:
            self._log(f"Could not find device '{self.device_name}'")
            return False

        self.client = BleakClient(device)
        self._log(f"Connecting to {device.address}...")

        try:
            await self.client.connect()
            self._log("Connected!")
            self._connected_event.set()

            # Start listening for notifications
            await self.client.start_notify(DATA_CHAR_UUID, self._notification_handler)
            self._log("Notifications enabled.")

            # Wait for initial handshake from firmware
            # Firmware sends RSP_STREAM_STOPPED on connection (line 566 in main.cpp)
            self._log("Waiting for initial status from device...")
            try:
                header, payload = await asyncio.wait_for(self._response_queue.get(), timeout=2.0)
                if header.type == PacketType.RSP_STREAM_STOPPED:
                    self._log("Device ready (IDLE mode)")
                    self._handshake_done.set()
            except asyncio.TimeoutError:
                self._log("No initial status received (continuing anyway)")
                self._handshake_done.set()

            # Perform ping-pong handshake
            if await self._ping():
                self._log("Handshake complete!")

                # Query current configuration
                config = await self.get_config()
                if config:
                    self._log(f"Device Config: Mode={config.mode}, ODR={config.odr_hz}Hz")

                return True
            else:
                self._log("Handshake failed!")
                await self.disconnect()
                return False

        except Exception as e:
            self._log(f"Connection failed: {e}")
            self.client = None
            return False

    async def disconnect(self):
        """Disconnect from device"""
        if self.client and self.client.is_connected:
            try:
                await self.client.stop_notify(DATA_CHAR_UUID)
            except:
                pass
            await self.client.disconnect()
            self._log("Disconnected.")

        self.client = None
        self._connected_event.clear()
        self._handshake_done.clear()

    def _notification_handler(self, sender: int, data: bytearray):
        """
        Handle incoming BLE notifications.
        Parses packets according to protocol and dispatches to callbacks.
        """
        if len(data) < 2:
            return

        header = Header.parse(data)
        if not header:
            self._log(f"Invalid header: {data[:10].hex()}")
            return

        payload = data[2:2+header.length]

        # Response packets → queue for async handlers (with payload)
        if header.type in [PacketType.RSP_PONG, PacketType.RSP_CONFIG,
                          PacketType.RSP_CONFIG_OK, PacketType.RSP_STREAM_STARTED,
                          PacketType.RSP_STREAM_STOPPED, PacketType.RSP_CALIB_STATUS]:
            # Store tuple of (header, payload) so we can parse response data
            asyncio.create_task(self._response_queue.put((header, payload)))

            if header.type == PacketType.RSP_CALIB_STATUS and len(payload) >= 1:
                status = payload[0]
                status_str = {0: "In Progress", 1: "Success", 2: "Failed"}
                self._log(f"Calibration Status: {status_str.get(status, 'Unknown')}")

        # Data packets → parse and callback
        elif header.type == PacketType.DATA_RAW_IMU:
            packet = RawImuPacket.parse(payload)
            if packet:
                if self.raw_callback:
                    self.raw_callback(packet)
                else:
                    self.raw_data.append(packet)

        elif header.type == PacketType.DATA_TRAJECTORY:
            packet = TrajectoryPacket.parse(payload)
            if packet:
                if self.trajectory_callback:
                    self.trajectory_callback(packet)
                else:
                    self.trajectory_data.append(packet)

    async def _send_command(self, packet_type: PacketType, payload: bytes = b'') -> bool:
        """Send command packet to device"""
        if not self.client or not self.client.is_connected:
            self._log("Not connected")
            return False

        header = Header(type=packet_type, length=len(payload))
        packet = header.pack() + payload

        try:
            await self.client.write_gatt_char(CMD_CHAR_UUID, packet, response=True)
            return True
        except Exception as e:
            self._log(f"Command failed: {e}")
            return False

    async def _ping(self) -> bool:
        """Send ping and wait for pong"""
        self._log("Sending PING...")
        if not await self._send_command(PacketType.CMD_PING):
            return False

        try:
            header, payload = await asyncio.wait_for(self._response_queue.get(), timeout=2.0)
            if header.type == PacketType.RSP_PONG:
                self._log("PONG received")
                return True
        except asyncio.TimeoutError:
            self._log("PING timeout")

        return False

    async def get_config(self) -> Optional[ConfigPayload]:
        """Query current device configuration"""
        self._log("Querying config...")
        if not await self._send_command(PacketType.CMD_GET_CONFIG):
            return None

        try:
            header, payload = await asyncio.wait_for(self._response_queue.get(), timeout=2.0)
            if header.type == PacketType.RSP_CONFIG:
                # Parse config from response payload
                config = ConfigPayload.parse(payload)
                if config:
                    self.current_config = config
                    self._log(f"Config received: Mode={config.mode}, ODR={config.odr_hz}Hz")
                    return config
                else:
                    self._log("Failed to parse config payload")
        except asyncio.TimeoutError:
            self._log("Config query timeout")

        return None

    async def set_config(self, mode: int, odr_hz: int = 50) -> bool:
        """
        Set device configuration.

        Args:
            mode: 0 for Raw IMU, 1 for Trajectory
            odr_hz: Sampling rate (typically 50Hz)

        Returns:
            True if config accepted
        """
        config = ConfigPayload(mode=mode, odr_hz=odr_hz)
        self._log(f"Setting config: Mode={mode}, ODR={odr_hz}Hz")

        if not await self._send_command(PacketType.CMD_SET_CONFIG, config.pack()):
            return False

        try:
            header, payload = await asyncio.wait_for(self._response_queue.get(), timeout=2.0)
            if header.type == PacketType.RSP_CONFIG_OK:
                self._log("Config set successfully")
                self.current_config = config
                self.streaming_mode = mode
                return True
        except asyncio.TimeoutError:
            self._log("Config set timeout")

        return False

    async def start_streaming(self, mode: Optional[int] = None) -> bool:
        """
        Start data streaming.

        Args:
            mode: Optional mode override (0: Raw, 1: Trajectory)
                 If None, uses current config

        Returns:
            True if streaming started
        """
        if mode is not None:
            if not await self.set_config(mode):
                return False

        self._log("Starting stream...")
        if not await self._send_command(PacketType.CMD_START_STREAM):
            return False

        try:
            header, payload = await asyncio.wait_for(self._response_queue.get(), timeout=2.0)
            if header.type == PacketType.RSP_STREAM_STARTED:
                mode_str = {0: "RAW IMU", 1: "TRAJECTORY", None: "CURRENT"}
                self._log(f"Streaming started ({mode_str.get(self.streaming_mode, 'Unknown')} mode)")
                return True
        except asyncio.TimeoutError:
            self._log("Stream start timeout")

        return False

    async def stop_streaming(self) -> bool:
        """Stop data streaming"""
        self._log("Stopping stream...")
        if not await self._send_command(PacketType.CMD_STOP_STREAM):
            return False

        try:
            header, payload = await asyncio.wait_for(self._response_queue.get(), timeout=2.0)
            if header.type == PacketType.RSP_STREAM_STOPPED:
                self._log("Streaming stopped")
                return True
        except asyncio.TimeoutError:
            self._log("Stream stop timeout")

        return False

    async def calibrate(self) -> bool:
        """
        Trigger CRT + FOC calibration.
        Device must be stationary on a table.

        Returns:
            True if calibration initiated (check status via notifications)
        """
        self._log("Starting calibration - KEEP DEVICE STILL!")
        if not await self._send_command(PacketType.CMD_CALIBRATE):
            return False

        # Initial status should arrive immediately
        try:
            header, payload = await asyncio.wait_for(self._response_queue.get(), timeout=2.0)
            if header.type == PacketType.RSP_CALIB_STATUS:
                self._log("Calibration started, waiting for completion...")
                # Final status will arrive via notification handler
                return True
        except asyncio.TimeoutError:
            self._log("Calibration start timeout")

        return False


# --- Example Usage ---

async def example_raw_stream():
    """Example: Stream raw IMU data"""

    def on_raw_data(packet: RawImuPacket):
        print(f"[{packet.timestamp_us/1e6:.3f}s] "
              f"Accel: ({packet.accel[0]:6.2f}, {packet.accel[1]:6.2f}, {packet.accel[2]:6.2f}) m/s² | "
              f"Gyro: ({packet.gyro[0]:6.2f}, {packet.gyro[1]:6.2f}, {packet.gyro[2]:6.2f}) rad/s | "
              f"FSR: {packet.force} | "
              f"Temp: {packet.temperature:.1f}°C")

    driver = TrajectoDriver(raw_callback=on_raw_data)

    if await driver.connect():
        # Stream raw data for 5 seconds
        await driver.start_streaming(mode=0)  # 0 = Raw
        await asyncio.sleep(5)
        await driver.stop_streaming()
        await driver.disconnect()


async def example_trajectory_stream():
    """Example: Stream trajectory estimates"""

    def on_trajectory(packet: TrajectoryPacket):
        print(f"[{packet.timestamp_us/1e6:.3f}s] "
              f"Pos: ({packet.pos[0]:6.3f}, {packet.pos[1]:6.3f}, {packet.pos[2]:6.3f}) m | "
              f"ZUPT: {packet.prob_zupt:.2f}")

    driver = TrajectoDriver(trajectory_callback=on_trajectory)

    if await driver.connect():
        # Stream trajectory for 5 seconds
        await driver.start_streaming(mode=1)  # 1 = Trajectory
        await asyncio.sleep(5)
        await driver.stop_streaming()
        await driver.disconnect()


async def example_calibration():
    """Example: Trigger device calibration"""

    driver = TrajectoDriver(verbose=True)

    if await driver.connect():
        print("\n" + "="*60)
        print("CALIBRATION PROCEDURE")
        print("="*60)
        print("1. Place the device on a FLAT, STABLE surface")
        print("2. DO NOT MOVE the device during calibration (~5 seconds)")
        print("3. Press Enter when ready...")
        input()

        await driver.calibrate()

        # Wait for calibration to complete (listen for status notifications)
        print("\nCalibrating... (Check logs for status)")
        await asyncio.sleep(8)

        await driver.disconnect()


async def main():
    """Interactive menu for testing driver"""

    print("\n" + "="*60)
    print("Trajecto BLE Driver - Test Interface")
    print("="*60)
    print("1. Stream Raw IMU Data")
    print("2. Stream Trajectory Data")
    print("3. Run Calibration")
    print("4. Exit")

    choice = input("\nSelect option [1-4]: ").strip()

    if choice == '1':
        await example_raw_stream()
    elif choice == '2':
        await example_trajectory_stream()
    elif choice == '3':
        await example_calibration()
    elif choice == '4':
        print("Exiting.")
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
