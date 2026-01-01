"""
Stationary IMU Data Acquisition for Allan Variance Analysis

This script collects long-duration stationary IMU data from the Trajecto device
for Allan Variance analysis. The data is saved in CSV format compatible with
allan_variance_analysis.py.

Allan Variance analysis requires:
- Device must remain completely stationary
- Recording duration: 1-3 hours recommended (longer = better)
- Stable temperature environment
- No vibrations or disturbances

Usage:
    python utils/acquire_stationary.py --duration 3600 --output stationary_data.csv
    python utils/acquire_stationary.py --duration 7200 --output long_stationary.csv
"""

import asyncio
import argparse
import csv
import os
import sys
from datetime import datetime
from typing import List, Dict

# Import the TrajectoDriver from receive.py
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Add parent directory to sys.path for model imports
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from receive import TrajectoDriver, RawImuPacket
except ImportError:
    print("Error: Could not import 'receive.py'. Ensure it is in the 'utils/' directory.")
    sys.exit(1)

from model.config import Config


# Constants
GRAVITY = Config.GRAVITY_MAGNITUDE  # Standard gravity (m/s²) - CODATA 2018
DEFAULT_DURATION_S = 5  # 2 hour default
DEFAULT_OUTPUT_DIR = "acquired_data/stationary"


class StationaryDataCollector:
    """
    Collects stationary IMU data for Allan Variance analysis.

    Streams raw IMU data from Trajecto device and saves it to CSV format
    with columns compatible with allan_variance_analysis.py:
    - accel_x_g, accel_y_g, accel_z_g (in g's)
    - gyro_x_rads, gyro_y_rads, gyro_z_rads (in rad/s)
    - temperature_c (in degrees Celsius)
    - timestamp (in seconds)

    CRITICAL: Data is written incrementally to prevent loss on connection drop.
    """

    def __init__(self, output_path: str, duration_s: float, flush_interval_s: float = 10.0):
        """
        Initialize the stationary data collector.

        Args:
            output_path: Path to save CSV file
            duration_s: Recording duration in seconds
            flush_interval_s: How often to flush data to disk (default: 10s)
        """
        self.output_path = output_path
        self.duration_s = duration_s
        self.flush_interval_s = flush_interval_s
        self.driver = None
        self.data_buffer: List[Dict] = []
        self.start_time = None
        self.recording = False

        # CSV file handle (kept open during recording)
        self.csv_file = None
        self.csv_writer = None

        # Statistics
        self.sample_count = 0
        self.dropped_packets = 0
        self.last_timestamp_us = None
        self.last_flush_time = None

        # Timestamp overflow correction (uint32_t max = 4,294,967,295 µs ≈ 71.6 min)
        self.timestamp_offset_us = 0
        self.last_raw_timestamp_us = None

    def _on_imu_data(self, packet: RawImuPacket):
        """
        Callback for raw IMU packets.

        Converts units and stores data:
        - Accel: m/s² → g (divide by 9.80665)
        - Gyro: rad/s (no conversion needed)
        - Temperature: °C (no conversion needed)

        Handles uint32_t timestamp overflow (wraps at 4,294,967,295 µs ≈ 71.6 min)
        by detecting rollover and applying cumulative offset.
        """
        if not self.recording:
            return

        # Detect and correct uint32_t timestamp overflow
        raw_timestamp_us = packet.timestamp_us
        if self.last_raw_timestamp_us is not None:
            # Overflow detected: current timestamp wrapped around to 0
            # (allows for some backwards jitter but catches rollover)
            UINT32_MAX = 4_294_967_295
            if raw_timestamp_us < self.last_raw_timestamp_us - 1_000_000:  # 1 second tolerance
                self.timestamp_offset_us += UINT32_MAX + 1
                print(f"\n[Timestamp overflow detected at sample {self.sample_count}]")
                print(f"  Offset applied: {self.timestamp_offset_us / 1_000_000:.1f}s")

        self.last_raw_timestamp_us = raw_timestamp_us
        corrected_timestamp_us = raw_timestamp_us + self.timestamp_offset_us

        # Check for dropped packets (50Hz = 20ms = 20000µs)
        if self.last_timestamp_us is not None:
            expected_dt = 20000  # µs
            actual_dt = corrected_timestamp_us - self.last_timestamp_us
            if actual_dt > expected_dt * 1.5:  # Allow 50% tolerance
                drops = int((actual_dt - expected_dt) / expected_dt)
                self.dropped_packets += drops

        self.last_timestamp_us = corrected_timestamp_us

        # Convert and store (use corrected timestamp to handle uint32_t overflow)
        data = {
            'timestamp': corrected_timestamp_us / 1_000_000.0,  # µs → seconds
            'accel_x_g': packet.accel[0] / GRAVITY,  # m/s² → g
            'accel_y_g': packet.accel[1] / GRAVITY,
            'accel_z_g': packet.accel[2] / GRAVITY,
            'gyro_x_rads': packet.gyro[0],  # rad/s (already correct)
            'gyro_y_rads': packet.gyro[1],
            'gyro_z_rads': packet.gyro[2],
            'temperature_c': packet.temperature,  # °C
        }

        self.data_buffer.append(data)
        self.sample_count += 1

    async def connect(self) -> bool:
        """Connect to Trajecto device."""
        print("Connecting to Trajecto device...")
        self.driver = TrajectoDriver(raw_callback=self._on_imu_data, verbose=True)

        if await self.driver.connect():
            print("Connected successfully!")
            return True
        else:
            print("Connection failed!")
            return False

    async def disconnect(self):
        """Disconnect from device and close CSV file."""
        if self.driver:
            await self.driver.disconnect()
            print("Disconnected from device.")

        # Close CSV file if open
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None

    def _flush_buffer_to_disk(self):
        """Write buffered data to disk and clear buffer."""
        if not self.data_buffer or not self.csv_writer:
            return

        # Write all buffered rows
        self.csv_writer.writerows(self.data_buffer)
        self.csv_file.flush()  # Force OS to write to disk

        # Clear buffer to free memory
        self.data_buffer.clear()

    async def record(self) -> bool:
        """
        Start recording stationary data for specified duration.
        Data is written incrementally to prevent loss on connection drop.

        Returns:
            True if recording completed successfully
        """
        if not self.driver:
            print("Not connected to device!")
            return False

        print("\n" + "="*70)
        print("STATIONARY IMU DATA ACQUISITION")
        print("="*70)
        print(f"Duration: {self.duration_s}s ({self.duration_s/60:.1f} minutes)")
        print(f"Output: {self.output_path}")
        print(f"Auto-save interval: {self.flush_interval_s}s (data is safe even if connection drops)")
        print("\nIMPORTANT INSTRUCTIONS:")
        print("  1. Place device on a STABLE, FLAT surface")
        print("  2. Ensure NO vibrations (away from fans, AC, traffic)")
        print("  3. Keep temperature STABLE (no direct sunlight)")
        print("  4. DO NOT TOUCH or MOVE the device during recording")
        print("="*70)

        input("\nPress Enter when the device is positioned and ready...")

        # Create output directory and open CSV file
        os.makedirs(os.path.dirname(self.output_path) or '.', exist_ok=True)

        try:
            self.csv_file = open(self.output_path, 'w', newline='')
            fieldnames = [
                'timestamp',
                'accel_x_g', 'accel_y_g', 'accel_z_g',
                'gyro_x_rads', 'gyro_y_rads', 'gyro_z_rads',
                'temperature_c'
            ]
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
            self.csv_writer.writeheader()
            self.csv_file.flush()
            print(f"\nCSV file created: {self.output_path}")
        except Exception as e:
            print(f"Failed to create CSV file: {e}")
            return False

        # Start streaming in RAW mode
        print("\nStarting data stream...")
        if not await self.driver.start_streaming(mode=0):
            print("Failed to start streaming!")
            return False

        print("Streaming started!")

        # Wait for stabilization
        print("\nWaiting 5 seconds for sensor stabilization...")
        await asyncio.sleep(5)

        # Clear buffer from stabilization period
        self.data_buffer.clear()
        self.sample_count = 0
        self.dropped_packets = 0
        self.last_timestamp_us = None
        self.timestamp_offset_us = 0
        self.last_raw_timestamp_us = None

        # Start recording
        print("\n" + "="*70)
        print("RECORDING IN PROGRESS - DO NOT MOVE DEVICE!")
        print("="*70)
        self.recording = True
        self.start_time = asyncio.get_event_loop().time()
        self.last_flush_time = self.start_time

        # Progress updates
        update_interval = 60  # Update every 60 seconds
        next_update = update_interval

        try:
            while True:
                await asyncio.sleep(1)

                elapsed = asyncio.get_event_loop().time() - self.start_time

                # Periodic flush to disk
                if elapsed - (self.last_flush_time - self.start_time) >= self.flush_interval_s:
                    self._flush_buffer_to_disk()
                    self.last_flush_time = asyncio.get_event_loop().time()

                # Progress update
                if elapsed >= next_update:
                    remaining = self.duration_s - elapsed
                    rate = self.sample_count / elapsed if elapsed > 0 else 0

                    # Calculate temperature stats if available
                    if self.data_buffer:
                        temps = [d.get('temperature_c', 0) for d in self.data_buffer]
                        temp_str = f"Temp: {sum(temps)/len(temps):.1f}°C | "
                    else:
                        temp_str = ""

                    print(f"  [{elapsed:.0f}s] Samples: {self.sample_count} | "
                          f"Rate: {rate:.1f} Hz | {temp_str}"
                          f"Remaining: {remaining:.0f}s ({remaining/60:.1f} min)")
                    next_update += update_interval

                # Check if duration reached
                if elapsed >= self.duration_s:
                    break

        finally:
            # Final flush before stopping
            self._flush_buffer_to_disk()

            # Stop recording
            self.recording = False
            await self.driver.stop_streaming()

        print("\n" + "="*70)
        print("RECORDING COMPLETE!")
        print("="*70)
        print(f"Total samples: {self.sample_count}")
        print(f"Duration: {elapsed:.1f}s")
        print(f"Actual rate: {self.sample_count/elapsed:.2f} Hz")
        print(f"Dropped packets: {self.dropped_packets}")
        print("="*70)

        return True

    def finalize_csv(self) -> bool:
        """
        Finalize CSV file (data already written incrementally).
        Just reports file statistics.

        Returns:
            True if file exists and is valid
        """
        if not os.path.exists(self.output_path):
            print("ERROR: CSV file does not exist!")
            return False

        try:
            # Calculate file size
            file_size = os.path.getsize(self.output_path)
            print(f"\nData saved to: {self.output_path}")
            print(f"File size: {file_size / 1024:.1f} KB ({file_size / (1024*1024):.2f} MB)")

            return True

        except Exception as e:
            print(f"Error accessing CSV file: {e}")
            return False

    def print_statistics(self):
        """Print basic statistics about collected data by reading the CSV file."""
        if not os.path.exists(self.output_path):
            print("No data file to analyze!")
            return

        try:
            import numpy as np
            import pandas as pd

            print("\n" + "="*70)
            print("DATA STATISTICS (from saved file)")
            print("="*70)

            # Read the CSV file
            df = pd.read_csv(self.output_path)

            if len(df) == 0:
                print("No data in file!")
                return

            # Extract arrays
            accel_x = df['accel_x_g'].values
            accel_y = df['accel_y_g'].values
            accel_z = df['accel_z_g'].values
            gyro_x = df['gyro_x_rads'].values
            gyro_y = df['gyro_y_rads'].values
            gyro_z = df['gyro_z_rads'].values
            temp = df['temperature_c'].values

            print("\nAccelerometer (g):")
            print(f"  X: mean={accel_x.mean():+.4f}, std={accel_x.std():.4f}")
            print(f"  Y: mean={accel_y.mean():+.4f}, std={accel_y.std():.4f}")
            print(f"  Z: mean={accel_z.mean():+.4f}, std={accel_z.std():.4f}")
            print(f"  |a|: mean={np.sqrt(accel_x**2 + accel_y**2 + accel_z**2).mean():.4f} g")

            print("\nGyroscope (rad/s):")
            print(f"  X: mean={gyro_x.mean():+.6f}, std={gyro_x.std():.6f}")
            print(f"  Y: mean={gyro_y.mean():+.6f}, std={gyro_y.std():.6f}")
            print(f"  Z: mean={gyro_z.mean():+.6f}, std={gyro_z.std():.6f}")

            print("\nTemperature (°C):")
            print(f"  Mean: {temp.mean():.2f}°C")
            print(f"  Std: {temp.std():.3f}°C")
            print(f"  Range: {temp.min():.2f} - {temp.max():.2f}°C")
            print(f"  Drift: {temp[-1] - temp[0]:+.2f}°C")

            print("\nData Quality:")
            print(f"  Expected samples (50Hz): {int(self.duration_s * 50)}")
            print(f"  Actual samples: {len(df)}")
            print(f"  Completeness: {len(df) / (self.duration_s * 50) * 100:.2f}%")
            print(f"  Dropped packets: {self.dropped_packets}")
            print("="*70)

        except Exception as e:
            print(f"Error analyzing statistics: {e}")
            import traceback
            traceback.print_exc()


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Acquire stationary IMU data for Allan Variance analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Record 1 hour of data (recommended minimum)
  python utils/acquire_stationary.py --duration 3600 --output stationary_1hr.csv

  # Record 3 hours of data (better for bias stability analysis)
  python utils/acquire_stationary.py --duration 10800 --output stationary_3hr.csv

  # Quick test (5 minutes)
  python utils/acquire_stationary.py --duration 300 --output test.csv

After recording, analyze with:
  python utils/allan_variance_analysis.py stationary_1hr.csv --rate 50
        """
    )

    parser.add_argument(
        '--duration',
        type=float,
        default=DEFAULT_DURATION_S,
        help=f'Recording duration in seconds (default: {DEFAULT_DURATION_S}s = 1 hour)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file path (default: auto-generated with timestamp)'
    )

    args = parser.parse_args()

    # Generate default output path if not specified
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = os.path.join(
            DEFAULT_OUTPUT_DIR,
            f"stationary_{int(args.duration)}s_{timestamp}.csv"
        )

    # Create collector
    collector = StationaryDataCollector(args.output, args.duration)

    try:
        # Connect to device
        if not await collector.connect():
            return

        # Record data (automatically saved incrementally)
        if await collector.record():
            # Finalize CSV and report statistics
            if collector.finalize_csv():
                # Print statistics
                collector.print_statistics()

                print("\n" + "="*70)
                print("NEXT STEPS")
                print("="*70)
                print(f"Run Allan Variance analysis with:")
                print(f"  python utils/allan_variance_analysis.py {args.output} --rate 50")
                print("\nOptional slicing (to remove start/end transients):")
                print(f"  python utils/allan_variance_analysis.py {args.output} --rate 50 \\")
                print(f"         --slice-start 60 --slice-end 60")
                print("="*70)

    except KeyboardInterrupt:
        print("\n\nRecording interrupted by user!")
        if collector.recording:
            collector.recording = False
            # Flush any remaining data
            if collector.data_buffer:
                collector._flush_buffer_to_disk()

        # Data already saved incrementally
        print(f"\nPartial data automatically saved to: {args.output}")
        print(f"Samples collected: {collector.sample_count}")

        # Show statistics if data exists
        if os.path.exists(args.output):
            show_stats = input("\nShow statistics for collected data? (y/n): ")
            if show_stats.lower() == 'y':
                collector.print_statistics()

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

    finally:
        await collector.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
