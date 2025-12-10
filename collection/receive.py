"""
This module implements the `TrajectoDriver` class, which facilitates Bluetooth
Low Energy (BLE) communication with a custom hardware device named "Trajecto".

The driver handles the essential aspects of BLE interaction, including scanning
for the device, establishing and managing connections, sending control commands,
and receiving streaming sensor data via GATT notifications. It provides a
callback mechanism for real-time processing of incoming sensor data.
"""

import asyncio
import struct
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

from bleak import BleakClient, BleakScanner

# --- BLE Service and Characteristic UUIDs ---
# These UUIDs must match those defined in the Trajecto device's firmware.
SERVICE_UUID: str = "ad43434e-c549-4594-b474-543153544557"
"""The UUID for the custom BLE service provided by the Trajecto device."""
DATA_CHAR_UUID: str = "ad43434f-c549-4594-b474-543153544557"
"""The UUID for the characteristic used to receive sensor data notifications."""
CMD_CHAR_UUID: str = "ad43434d-c549-4594-b474-543153544557"
"""The UUID for the characteristic used to send commands to the device."""
DEVICE_NAME: str = "Trajecto"
"""The advertised name of the BLE device to scan for."""


class TrajectoDriver:
    """A driver class to connect to the Trajecto BLE device, manage data collection,
    and control device state.

    This class provides an asynchronous interface for interacting with the
    Trajecto hardware, allowing for connection establishment, command transmission
    (e.g., 'start'/'stop' data stream), and processing of incoming sensor data.
    """

    def __init__(
        self,
        device_name: str = DEVICE_NAME,
        data_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        """Initializes the TrajectoDriver.

        Args:
            device_name: The advertised name of the BLE device to connect to.
            data_callback: An optional callback function to be called with each
                received sensor data point. If provided, `self.data` will not
                store the data internally. The callback should accept a single
                argument: a dictionary representing the sensor data.
        """
        self.device_name: str = device_name
        self.client: Optional[BleakClient] = None  # Bleak client instance for BLE communication.
        self.data: List[Dict[str, Any]] = []  # Internal buffer for collected data if no callback.
        self.data_callback: Optional[
            Callable[[Dict[str, Any]], None]
        ] = data_callback  # User-defined function for data processing.
        self._connected_event: asyncio.Event = asyncio.Event()  # Event to signal connection status.

    async def connect(self) -> bool:
        """Scans for the specified BLE device and establishes a connection.

        Returns:
            True if connection was successful, False otherwise.
        """
        print(f"Scanning for '{self.device_name}'...")
        # Search for the device by its advertised name.
        device = await BleakScanner.find_device_by_name(self.device_name)
        if device is None:
            print(f"Could not find device with name '{self.device_name}'")
            return False

        self.client = BleakClient(device)
        print(f"Connecting to {self.device_name} ({device.address})...")
        try:
            # Attempt to connect to the discovered BLE device.
            await self.client.connect()
            print("Connected successfully!")
            self._connected_event.set()  # Signal that connection is established.
            return True
        except Exception as e:
            print(f"Failed to connect: {e}")
            self.client = None  # Reset client on failure.
            return False

    async def disconnect(self) -> None:
        """Disconnects from the BLE device if an active connection exists."""
        if self.client and self.client.is_connected:
            await self.client.disconnect()
            print("Disconnected.")
        self.client = None  # Clear client instance.
        self._connected_event.clear()  # Clear connection event.

    def _notification_handler(self, sender: int, data: bytearray) -> None:
        """Handles incoming data notifications received from the BLE device.

        This method is registered as a callback for the data characteristic.
        It parses the raw byte array into structured sensor data, supporting
        batches of data points within a single notification, and passes them
        to the user-defined `data_callback` or stores them internally.

        The expected C++ struct format is:
        struct SensorData {
            float time;
            float accel_x, accel_y, accel_z;
            float gyro_x, gyro_y, gyro_z;
            uint32_t fsr; // Or float if it's not a raw uint32
        };
        Total size: 7 floats (4 bytes each) + 1 uint32_t (4 bytes) = 28 + 4 = 32 bytes.

        Args:
            sender: The handle of the characteristic that sent the notification.
            data: The raw bytearray received from the device.
        """
        # Define the size and format of the C++ struct being sent by the device.
        struct_size: int = 32
        # '<fffffffI' specifies:
        # '<': Little-endian byte order.
        # 'f': 7 single-precision floats (time, accel_x,y,z, gyro_x,y,z).
        # 'I': 1 unsigned integer (fsr).
        struct_format: str = "<fffffffI"

        # Check if the received data length is a multiple of the expected struct size.
        # This allows processing batches of sensor readings sent in one notification.
        if len(data) % struct_size == 0:
            num_structs: int = len(data) // struct_size
            for i in range(num_structs):
                offset: int = i * struct_size
                chunk: bytes = data[offset : offset + struct_size]
                try:
                    # Unpack the byte chunk into a tuple of Python values.
                    unpacked_data: Tuple[Any, ...] = struct.unpack(struct_format, chunk)
                    sensor_data: Dict[str, Any] = {
                        "time": unpacked_data[0],
                        "accel_x": unpacked_data[1],
                        "accel_y": unpacked_data[2],
                        "accel_z": unpacked_data[3],
                        "gyro_x": unpacked_data[4],
                        "gyro_y": unpacked_data[5],
                        "gyro_z": unpacked_data[6],
                        "fsr": unpacked_data[7],  # FSR data, assumed to be unsigned int.
                    }
                    if self.data_callback:
                        self.data_callback(sensor_data)  # Pass to external callback.
                    else:
                        self.data.append(sensor_data)  # Store internally.
                        # Optional: print status periodically if storing internally.
                        if len(self.data) % (10 * num_structs) == 0:
                            print(f"Received data point #{len(self.data)}")
                except struct.error as e:
                    print(f"Error unpacking chunk {i+1}/{num_structs}: {e}")
        else:
            print(
                f"Received unexpected data length (len: {len(data)}). Hex: {data.hex()}"
            )

    async def start_data_collection(self) -> None:
        """Subscribes to notifications on the data characteristic and sends the 'strt' command.

        Requires an active BLE connection. This sequence initiates the sensor
        data streaming from the device.
        """
        if not self.client or not self.client.is_connected:
            print("Client not connected. Cannot start data collection.")
            return

        print("Starting data collection...")
        try:
            # Start receiving notifications from the DATA_CHAR_UUID.
            await self.client.start_notify(DATA_CHAR_UUID, self._notification_handler)
            # Send the 'strt' command to the device to begin data streaming.
            await self.client.write_gatt_char(CMD_CHAR_UUID, b"strt", response=True)
            print("Sent 'strt' command and started notifications.")
        except Exception as e:
            print(f"Failed to start data collection: {e}")
            # Attempt to clean up if something went wrong during startup.
            await self.stop_data_collection()

    async def stop_data_collection(self) -> None:
        """Sends the 'stop' command to the device and unsubscribes from notifications.

        Requires an active BLE connection. This terminates the sensor data
        streaming from the device.
        """
        if not self.client or not self.client.is_connected:
            print("Client not connected. Cannot stop data collection.")
            return

        print("Stopping data collection...")
        try:
            # Send the 'stop' command to the device.
            await self.client.write_gatt_char(CMD_CHAR_UUID, b"stop", response=True)
            # Stop receiving notifications from the DATA_CHAR_UUID.
            await self.client.stop_notify(DATA_CHAR_UUID)
            print("Sent 'stop' command and stopped notifications.")
        except Exception as e:
            print(f"Failed to stop data collection: {e}")

    async def wait_for_connection(self) -> None:
        """Waits indefinitely until a BLE connection is established."""
        await self._connected_event.wait()

    async def write_command(self, command: str) -> None:
        """Writes a command string to the command characteristic.

        Args:
            command: The command string to write (e.g., "strt", "stop").
                Commands are typically short (e.g., 4 characters) due to BLE limitations.
        """
        if not self.client or not self.client.is_connected:
            print("Client not connected. Cannot write command.")
            return

        if len(command) > 4:
            print("Warning: Command string might be truncated to 4 bytes on device.")

        try:
            # Encode the command string to ASCII bytes before sending.
            await self.client.write_gatt_char(
                CMD_CHAR_UUID, command.encode("ascii"), response=True
            )
            print(f"Command '{command}' sent.")
        except Exception as e:
            print(f"Failed to send command '{command}': {e}")


async def main() -> None:
    """Example of how to use TrajectoDriver as a standalone script for testing."""

    def my_data_processor(sensor_data: Dict[str, Any]) -> None:
        """Example callback to process data as it arrives. Prints selected fields."""
        # For brevity in continuous stream, only print time, accel_x, and fsr.
        print(
            f"Time: {sensor_data['time']:.6f}s, Accel X: {sensor_data['accel_x']:.2f}g, FSR: {sensor_data['fsr']}"
        )

    # Instantiate the driver, providing the example data processor as a callback.
    collector: TrajectoDriver = TrajectoDriver(data_callback=my_data_processor)
    try:
        # Attempt to connect to the device.
        if await collector.connect():
            # If connected, start data collection.
            await collector.start_data_collection()

            print("\n--- Data Collection Started ---")
            print("Press Ctrl+C to stop data collection and disconnect.")

            # Keep the program running to receive notifications for a duration.
            await asyncio.sleep(10)  # Collect data for 10 seconds.
        else:
            print("Failed to connect to the device.")

    except asyncio.CancelledError:
        print("\nProgram cancelled (e.g., by asyncio.run() timeout).")
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected. Stopping data collection.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # Ensure proper cleanup: stop data collection and disconnect.
        if collector.client and collector.client.is_connected:
            await collector.stop_data_collection()
            await collector.disconnect()
        print("Exiting.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram interrupted by user on startup. Exiting gracefully.")
    except Exception as e:
        print(f"An error occurred during startup: {e}")