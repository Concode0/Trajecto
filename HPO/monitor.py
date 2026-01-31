# Trajecto: Real-time 3D Trajectory Reconstruction System
# Copyright 2025-2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# NOTICE: This software is protected under the following ROK Patent Applications:
# 1. Hybrid ESKF-Stateful TCN Architecture (No. 10-2025-0201093)
# 2. 3D Ground Truth Generation via Hovering Signal Engineering (No. 10-2025-0201092)
#
# Commercial use or redistribution of the core logic requires a separate license.
# For inquiries, contact: nemonanconcode@gmail.com

#!/usr/bin/env python3
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
