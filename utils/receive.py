# Trajecto: Real-time 3D Trajectory Reconstruction System
# Copyright (C) 2025-2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# [PATENT NOTICE]
# This implementation is protected under ROK Patent Applications 10-2025-0201093/092.
# Commercial use without a separate license is strictly prohibited.
#
# Contact: nemonanconcode@gmail.com

"""BLE driver for Trajecto device implementing firmware packet protocol.

Supports handshake, mode config (Raw/Trajectory), calibration (CRT+FOC), and dual streaming.
"""

import asyncio
import struct
import sys
import time
from enum import IntEnum
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

from bleak import BleakClient, BleakScanner

SERVICE_UUID: str = "ad43434e-c549-4594-b474-543153544557"
DATA_CHAR_UUID: str = "ad43434f-c549-4594-b474-543153544557"  # Notify
CMD_CHAR_UUID: str = "ad43434d-c549-4594-b474-543153544557"   # Write
DEVICE_NAME: str = "Trajecto"


class PacketType(IntEnum):
    """Packet type identifiers from firmware protocol"""
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


class TrajectoryFlags(IntEnum):
    """Trajectory packet flags (bitfield)"""
    NONE = 0x00
    ABSOLUTE_REF = 0x01  # Sent due to max_time_gap (absolute reference)
    PEN_DOWN = 0x02      # Pen is in contact (writing)
    KEYFRAME = 0x04      # Pen state changed (stroke boundary)


@dataclass
class Header:
    """Packet header: type (1B) + length (1B)"""
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
    """Config payload: mode (Raw=0/Trajectory=1), ODR, Swinging Door enable"""
    mode: int         # 0: Raw, 1: Trajectory
    odr_hz: int       # Sampling rate (fixed at 50Hz)
    enable_sda: int   # 0: Disabled, 1: Enabled (Swinging Door compression)
    reserved: int = 0

    @staticmethod
    def parse(data: bytes) -> Optional['ConfigPayload']:
        if len(data) < 4:
            return None
        unpacked = struct.unpack('<BBBB', data[:4])
        return ConfigPayload(
            mode=unpacked[0],
            odr_hz=unpacked[1],
            enable_sda=unpacked[2],
            reserved=unpacked[3]
        )

    def pack(self) -> bytes:
        return struct.pack('<BBBB', self.mode, self.odr_hz,
                          self.enable_sda, self.reserved)


@dataclass
class RawImuPacket:
    """Raw IMU packet (accel in m/s², gyro in rad/s)"""
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
    """Trajectory packet (ESKF-TCN output)

    Note: Due to Swinging Door compression, not all trajectory points are transmitted.
    Points are sent when:
    - Trajectory deviates beyond tolerance (delta compression)
    - Time gap exceeds max_time_gap_us (absolute reference, default 500ms)
    - Buffer fills up
    - Pen state changes (keyframe)

    The `flags` field explicitly indicates the reason for sending this packet.
    """
    timestamp_us: int
    pos: tuple    # (x, y, z) in meters
    vel: tuple    # (x, y, z) in m/s
    quat: tuple   # (w, x, y, z) quaternion
    prob_zupt: float  # Zero-velocity probability
    flags: int    # Bitfield of TrajectoryFlags

    @staticmethod
    def parse(data: bytes) -> Optional['TrajectoryPacket']:
        if len(data) < 52:  # 4 + 12 + 12 + 16 + 4 + 1 + 3 (padding)
            return None
        # Format: I=uint32(4) + 11*f=float[11](44) + B=uint8(1) + xxx=pad(3) = 52 bytes
        # 11 floats = pos[3] + vel[3] + quat[4] + zupt_prob[1]
        unpacked = struct.unpack('<IfffffffffffBxxx', data[:52])
        return TrajectoryPacket(
            timestamp_us=unpacked[0],
            pos=(unpacked[1], unpacked[2], unpacked[3]),
            vel=(unpacked[4], unpacked[5], unpacked[6]),
            quat=(unpacked[7], unpacked[8], unpacked[9], unpacked[10]),
            prob_zupt=unpacked[11],
            flags=unpacked[12]
        )

    def is_absolute_reference(self) -> bool:
        """Check if this packet is marked as absolute reference.

        Returns:
            True if ABSOLUTE_REF flag is set (sent due to time gap)
        """
        return bool(self.flags & TrajectoryFlags.ABSOLUTE_REF)

    def is_pen_down(self) -> bool:
        """Check if pen is in contact (writing).

        Returns:
            True if PEN_DOWN flag is set
        """
        return bool(self.flags & TrajectoryFlags.PEN_DOWN)

    def is_keyframe(self) -> bool:
        """Check if this is a keyframe (pen state transition).

        Returns:
            True if KEYFRAME flag is set (pen up/down transition)
        """
        return bool(self.flags & TrajectoryFlags.KEYFRAME)


class TrajectoDriver:
    """BLE driver for Trajecto with handshake, mode selection, calibration, and streaming."""

    def __init__(
        self,
        device_name: str = DEVICE_NAME,
        raw_callback: Optional[Callable[[RawImuPacket], None]] = None,
        trajectory_callback: Optional[Callable[[TrajectoryPacket], None]] = None,
        verbose: bool = True
    ):
        """Initializes BLE driver with optional packet callbacks."""
        self.device_name = device_name
        self.client: Optional[BleakClient] = None
        self.verbose = verbose

        self.raw_callback = raw_callback
        self.trajectory_callback = trajectory_callback

        self._connected_event = asyncio.Event()
        self._handshake_done = asyncio.Event()
        self._response_queue: asyncio.Queue = asyncio.Queue()

        self.current_config: Optional[ConfigPayload] = None
        self.streaming_mode: Optional[int] = None

        self.raw_data: List[RawImuPacket] = []
        self.trajectory_data: List[TrajectoryPacket] = []

    def _log(self, msg: str):
        """Prints log if verbose enabled"""
        if self.verbose:
            print(f"[TrajectoDriver] {msg}")

    async def connect(self) -> bool:
        """Scans, connects, and performs handshake. Returns True on success."""
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

            await self.client.start_notify(DATA_CHAR_UUID, self._notification_handler)
            self._log("Notifications enabled.")

            self._log("Waiting for initial status from device...")
            try:
                header, payload = await asyncio.wait_for(self._response_queue.get(), timeout=2.0)
                if header.type == PacketType.RSP_STREAM_STOPPED:
                    self._log("Device ready (IDLE mode)")
                    self._handshake_done.set()
            except asyncio.TimeoutError:
                self._log("No initial status received (continuing anyway)")
                self._handshake_done.set()

            if await self._ping():
                self._log("Handshake complete!")

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
        """Disconnects from device"""
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
        """Parses incoming BLE packets and dispatches to callbacks."""
        if len(data) < 2:
            return

        header = Header.parse(data)
        if not header:
            self._log(f"Invalid header: {data[:10].hex()}")
            return

        payload = data[2:2+header.length]

        if header.type in [PacketType.RSP_PONG, PacketType.RSP_CONFIG,
                          PacketType.RSP_CONFIG_OK, PacketType.RSP_STREAM_STARTED,
                          PacketType.RSP_STREAM_STOPPED, PacketType.RSP_CALIB_STATUS]:
            asyncio.create_task(self._response_queue.put((header, payload)))

            if header.type == PacketType.RSP_CALIB_STATUS and len(payload) >= 1:
                status = payload[0]
                status_str = {0: "In Progress", 1: "Success", 2: "Failed"}
                self._log(f"Calibration Status: {status_str.get(status, 'Unknown')}")

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
        """Sends command packet to device"""
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
        """Sends ping and waits for pong"""
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
        """Queries current device config"""
        self._log("Querying config...")
        if not await self._send_command(PacketType.CMD_GET_CONFIG):
            return None

        try:
            header, payload = await asyncio.wait_for(self._response_queue.get(), timeout=2.0)
            if header.type == PacketType.RSP_CONFIG:
                config = ConfigPayload.parse(payload)
                if config:
                    self.current_config = config
                    sda_status = "Enabled" if config.enable_sda else "Disabled"
                    self._log(f"Config received: Mode={config.mode}, ODR={config.odr_hz}Hz, SDA={sda_status}")
                    return config
                else:
                    self._log("Failed to parse config payload")
        except asyncio.TimeoutError:
            self._log("Config query timeout")

        return None

    async def set_config(self, mode: int, odr_hz: int = 50, enable_sda: int = 1) -> bool:
        """Sets device config.

        Args:
            mode: 0=Raw IMU, 1=Trajectory
            odr_hz: Sampling rate (default 50Hz)
            enable_sda: 0=Disabled, 1=Enabled Swinging Door compression (default 1)
        """
        config = ConfigPayload(mode=mode, odr_hz=odr_hz, enable_sda=enable_sda)
        sda_status = "Enabled" if enable_sda else "Disabled"
        self._log(f"Setting config: Mode={mode}, ODR={odr_hz}Hz, SDA={sda_status}")

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
        """Starts streaming (optionally sets mode first)."""
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
        """Stops streaming"""
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
        """Triggers CRT+FOC calibration (device must be stationary)."""
        self._log("Starting calibration - KEEP DEVICE STILL!")
        if not await self._send_command(PacketType.CMD_CALIBRATE):
            return False

        try:
            header, payload = await asyncio.wait_for(self._response_queue.get(), timeout=2.0)
            if header.type == PacketType.RSP_CALIB_STATUS:
                self._log("Calibration started, waiting for completion...")
                return True
        except asyncio.TimeoutError:
            self._log("Calibration start timeout")

        return False


async def example_raw_stream():
    """Example: streams raw IMU data"""

    def on_raw_data(packet: RawImuPacket):
        print(f"[{packet.timestamp_us/1e6:.3f}s] "
              f"Accel: ({packet.accel[0]:6.2f}, {packet.accel[1]:6.2f}, {packet.accel[2]:6.2f}) m/s² | "
              f"Gyro: ({packet.gyro[0]:6.2f}, {packet.gyro[1]:6.2f}, {packet.gyro[2]:6.2f}) rad/s | "
              f"FSR: {packet.force} | "
              f"Temp: {packet.temperature:.1f}°C")

    driver = TrajectoDriver(raw_callback=on_raw_data)

    if await driver.connect():
        await driver.start_streaming(mode=0)
        await asyncio.sleep(5)
        await driver.stop_streaming()
        await driver.disconnect()


async def example_trajectory_stream():
    """Example: streams trajectory estimates"""

    def on_trajectory(packet: TrajectoryPacket):
        # Build flags string
        flags_str = []
        if packet.is_absolute_reference():
            flags_str.append("ABS")
        if packet.is_pen_down():
            flags_str.append("PEN")
        if packet.is_keyframe():
            flags_str.append("KEY")
        flags_display = "|".join(flags_str) if flags_str else "---"

        print(f"[{packet.timestamp_us/1e6:.3f}s] "
              f"Pos: ({packet.pos[0]:6.3f}, {packet.pos[1]:6.3f}, {packet.pos[2]:6.3f}) m | "
              f"ZUPT: {packet.prob_zupt:.2f} | "
              f"Flags: {flags_display}")

    driver = TrajectoDriver(trajectory_callback=on_trajectory)

    if await driver.connect():
        await driver.start_streaming(mode=1)
        await asyncio.sleep(5)
        await driver.stop_streaming()
        await driver.disconnect()


async def example_calibration():
    """Example: triggers device calibration"""

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

        print("\nCalibrating... (Check logs for status)")
        await asyncio.sleep(8)

        await driver.disconnect()


async def main():
    """Interactive test menu"""

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
