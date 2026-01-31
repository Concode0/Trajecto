# Trajecto: Real-time 3D Trajectory Reconstruction System (Software)
# Copyright 2025-2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# NOTICE: This software implements the "Hybrid ESKF-Stateful TCN" logic 
# protected under ROK Patent Application No. 10-2025-YYYYYYY.
# Commercial use requires a separate license from the author.

"""
This script connects to a specified serial port, reads incoming data line by line,
and logs it to an output file.
"""

import argparse
import serial
import os
import sys

def main():
    """Main function to run the serial logger."""
    parser = argparse.ArgumentParser(description="Serial Port Data Logger.")
    parser.add_argument("--port", required=True, help="Serial port to connect to (e.g., /dev/ttyUSB0, COM1).")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate for serial communication.")
    parser.add_argument("--output", required=True, help="Path to the output file where data will be saved.")
    parser.add_argument("--echo", action="store_true", help="Echo received data to console.")
    parser.add_argument("--log-raw-errors", help="Optional: Path to a file where raw undecodable bytes will be logged.")

    args = parser.parse_args()

    # Ensure the output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Ensure the raw error log directory exists, if specified
    raw_error_dir = None
    if args.log_raw_errors:
        raw_error_dir = os.path.dirname(args.log_raw_errors)
        if raw_error_dir and not os.path.exists(raw_error_dir):
            os.makedirs(raw_error_dir)

    ser = None
    output_file = None
    raw_error_file = None

    try:
        print(f"Attempting to open serial port {args.port} at {args.baud} baud...")
        ser = serial.Serial(args.port, args.baud, timeout=1) # timeout ensures read_until doesn't block forever
        print(f"Successfully connected to {args.port}.")

        print(f"Opening output file {args.output}...")
        output_file = open(args.output, 'w', newline='') # 'newline=' for consistent line endings
        print(f"Logging data to {args.output}. Press Ctrl+C to stop.")

        if args.log_raw_errors:
            print(f"Opening raw error log file {args.log_raw_errors}...")
            raw_error_file = open(args.log_raw_errors, 'wb') # 'wb' for writing raw bytes
            print(f"Undecodable raw bytes will be logged to {args.log_raw_errors}.")

        while True:
            line_bytes = ser.read_until(b'\n')
            if line_bytes:
                try:
                    line_str = line_bytes.decode('utf-8').strip()
                    if line_str: # Only process non-empty lines
                        output_file.write(line_str + '\n')
                        if args.echo:
                            sys.stdout.write(line_str + '\n')
                            sys.stdout.flush() # Ensure it's printed immediately
                except UnicodeDecodeError:
                    print(f"Warning: Could not decode bytes: {line_bytes}", file=sys.stderr)
                    if raw_error_file:
                        raw_error_file.write(line_bytes + b'\n')
                        raw_error_file.flush() # Ensure bytes are written immediately

    except serial.SerialException as e:
        print(f"Error: Could not open serial port '{args.port}': {e}", file=sys.stderr)
    except FileNotFoundError:
        print(f"Error: A specified directory for output or raw error log does not exist or cannot be created.", file=sys.stderr)
    except KeyboardInterrupt:
        print("\nStopping data logger.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
    finally:
        if output_file:
            output_file.close()
            print(f"Output file '{args.output}' closed.")
        if raw_error_file:
            raw_error_file.close()
            print(f"Raw error log file '{args.log_raw_errors}' closed.")
        if ser and ser.is_open:
            ser.close()
            print(f"Serial port '{args.port}' closed.")

if __name__ == "__main__":
    main()
