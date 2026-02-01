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

"""
CLI Monitoring Tool for Entangle HPO System.
"""

import socket
import time
import os
import sys
import datetime
from protocol import *

def clear_screen():
    print("\033[H\033[J", end="")

def format_time(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))

def main(host='127.0.0.1', port=9999):
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((host, port))
            
            while True:
                # Request Status
                sock.sendall(pack_packet(OP_GET_STATUS))
                opcode, data = unpack_packet(sock)
                
                if opcode == OP_STATUS_RSP:
                    status = data
                    clear_screen()
                    print(f"=== Entangle HPO Monitor ===")
                    print(f"Server: {host}:{port}")
                    print(f"Best Loss: {status['best_loss']:.6f}")
                    print(f"Queue Size: {status['queue_size']}")
                    print("-" * 40)
                    
                    print(f"ASHA Rung Progress:")
                    for rung, count in status['rung_stats'].items():
                        print(f"  Rung {rung} epochs: {count} trials completed")
                    print("-" * 40)
                    
                    print(f"Active Workers ({len(status['workers'])}):")
                    print(f"{'Address':<20} | {'Last Seen':<10} | {'Current Task Trial ID'}")
                    print("-" * 60)
                    for w in status['workers']:
                        print(f"{w['addr']:<20} | {w['seen']}s ago   | {w['task']}")
                    print("-" * 60)
                    
                else:
                    print("Error: Unexpected response from kernel.")
                    break
                
                time.sleep(1)
                
        except ConnectionRefusedError:
            clear_screen()
            print(f"Connecting to Kernel at {host}:{port}...")
            time.sleep(2)
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(2)
        finally:
            try:
                sock.close()
            except:
                pass

if __name__ == '__main__':
    host = sys.argv[1] if len(sys.argv) > 1 else '127.0.0.1'
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 9999
    main(host, port)
