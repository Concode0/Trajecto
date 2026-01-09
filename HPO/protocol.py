"""
This module defines the communication protocol and Instruction Set Architecture (ISA)
for the Entangle distributed HPO system.

It defines the operation codes (Opcodes) used for message passing between the
Kernel (server) and Agents (workers), as well as helper functions for packing
and unpacking network packets.
"""

import struct
import pickle
from typing import Any, Tuple, Optional

# --- ISA (Instruction Set Architecture) ---

# Local Stack & Task Management
OP_GET_TASK     = 0xA0  # Request a new task range from Kernel
OP_TASK_RSP     = 0xA1  # Response from Kernel containing task parameters
OP_FOUND        = 0xB0  # Report result/solution back to Kernel
OP_HALT         = 0xFF  # Command to stop/exit the agent
OP_HEARTBEAT    = 0xC0  # Agent heartbeat to signal liveness

# Monitoring
OP_GET_STATUS   = 0xD0  # Request system status (workers, progress)
OP_STATUS_RSP   = 0xD1  # Response containing status data

def pack_packet(opcode: int, data: Any = None) -> bytes:
    """Packs an opcode and data into a length-prefixed binary packet.

    Args:
        opcode (int): The operation code (1 byte).
        data (Any, optional): Python object to serialize and send. Defaults to None.

    Returns:
        bytes: The packed binary message ready for transmission.
    """
    payload = struct.pack('B', opcode) + pickle.dumps(data)
    return struct.pack('>I', len(payload)) + payload

def unpack_packet(sock) -> Tuple[Optional[int], Any]:
    """Reads and unpacks a length-prefixed packet from a socket.

    Args:
        sock: The connected socket object.

    Returns:
        Tuple[Optional[int], Any]: A tuple containing the opcode and the deserialized data.
            Returns (None, None) if the connection is closed or an error occurs.
    """
    try:
        raw_len = sock.recv(4)
        if not raw_len:
            return None, None
        msg_len = struct.unpack('>I', raw_len)[0]
        
        payload = b''
        while len(payload) < msg_len:
            chunk = sock.recv(msg_len - len(payload))
            if not chunk:
                return None, None
            payload += chunk
            
        return payload[0], pickle.loads(payload[1:])
    except Exception:
        return None, None
