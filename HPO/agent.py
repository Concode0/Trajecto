"""
This module defines the HPOAgent, a worker client for the Entangle distributed
Hyperparameter Optimization (HPO) system.

The Agent is responsible for:
1.  Connecting to the central HPO Kernel.
2.  Requesting training tasks (hyperparameters).
3.  Executing the training script (`train.py`) as a subprocess.
4.  Parsing the training output to extract the loss metric.
5.  Reporting the result back to the Kernel.
6.  Sending periodic heartbeats to signal liveness.
"""

import socket
import time
import sys
import subprocess
import re
import threading
import argparse
from typing import List, Tuple, Any, Optional
from protocol import *

# Global lock for synchronized printing
print_lock = threading.Lock()

def synchronized_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)

class HPOAgent:
    """A worker agent that executes HPO tasks assigned by the Kernel.

    Attributes:
        server_addr (tuple): The (IP, Port) of the HPO Kernel.
        sock (socket.socket): The TCP socket connected to the Kernel.
        worker_id (int): A unique identifier for this worker instance (for logging).
        prefix (str): Log prefix containing the worker ID.
        is_alive (bool): Flag to control the heartbeat thread's lifecycle.
    """

    def __init__(self, kernel_ip='127.0.0.1', port=9999, worker_id=0):
        """Initializes the HPOAgent.

        Args:
            kernel_ip (str): IP address of the HPO Kernel. Defaults to '127.0.0.1'.
            port (int): TCP port of the HPO Kernel. Defaults to 9999.
            worker_id (int): ID for this worker thread/process. Defaults to 0.
        """
        self.server_addr = (kernel_ip, port)
        self.sock = None
        self.sock_lock = threading.Lock()
        self.worker_id = worker_id
        self.prefix = f"[Worker-{worker_id}]"
        self.is_alive = True # Flag to control heartbeat thread

    def log(self, message):
        """Thread-safe logging to stdout."""
        synchronized_print(f"{self.prefix} {message}")

    def connect(self):
        """Establishes a TCP connection to the Kernel. Retries on failure."""
        while True:
            try:
                new_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                new_sock.connect(self.server_addr)
                with self.sock_lock:
                    self.sock = new_sock
                self.log(f"Connected to Kernel at {self.server_addr}")
                return
            except ConnectionRefusedError:
                self.log("Kernel unavailable. Retrying in 3s...")
                time.sleep(3)

    def request_task(self) -> Tuple[Optional[int], Any]:
        """Requests a new task from the Kernel.

        Returns:
            Tuple[Optional[int], Any]: A tuple containing the opcode and the payload
            (task parameters) received from the Kernel. Returns (None, None) on error.
        """
        if not self.sock: return None, None
        try:
            with self.sock_lock:
                if self.sock:
                    self.sock.sendall(pack_packet(OP_GET_TASK))
            return unpack_packet(self.sock)
        except Exception as e:
            self.log(f"Network Error: {e}")
            self.sock.close()
            self.sock = None
            return None, None

    def send_result(self, params, loss):
        """Sends the training result back to the Kernel.

        Args:
            params (dict): The hyperparameters used for the task.
            loss (float): The final validation loss achieved.
        """
        if self.sock:
            # We reuse OP_FOUND for reporting results
            with self.sock_lock:
                self.sock.sendall(pack_packet(OP_FOUND, (params, loss)))

    def send_heartbeat(self):
        """Sends a single heartbeat packet to the Kernel."""
        if self.sock:
            try:
                with self.sock_lock:
                    self.sock.sendall(pack_packet(OP_HEARTBEAT))
            except Exception as e:
                self.log(f"Heartbeat failed: {e}")
                # Don't close socket here, main loop will handle reconnection if needed

    def _heartbeat_sender(self):
        """Background thread function to send periodic heartbeats."""
        while self.is_alive:
            self.send_heartbeat()
            time.sleep(5) # Send heartbeat every 5 seconds

    def run_training(self, params):
        """Executes the training script with the given hyperparameters.

        Args:
            params (dict): Dictionary containing 'lr', 'batch_size', 'epochs', etc.

        Returns:
            Optional[float]: The parsed 'FINAL_LOSS' from the training script output,
            or None if training failed or the loss couldn't be parsed.
        """
        trial_id = params.get('trial_id', 'unknown')

        # Build command with all HPO parameters
        cmd = [
            sys.executable, "tune.py",
            "--lr", str(params['lr']),
            "--batch_size", str(int(params['batch_size'])),
            "--epochs", str(params['epochs']),
        ]

        # Add optional HPO parameters if present
        if 'mahalanobis_threshold' in params:
            cmd.extend(["--mahalanobis_threshold", str(params['mahalanobis_threshold'])])
        if 'dropout' in params:
            cmd.extend(["--dropout", str(params['dropout'])])
        if 'kernel_size' in params:
            cmd.extend(["--kernel_size", str(int(params['kernel_size']))])
        if 'tcn_channel_size' in params:
            cmd.extend(["--tcn_channel_size", str(int(params['tcn_channel_size']))])
        if 'num_tcn_layers' in params:
            cmd.extend(["--num_tcn_layers", str(int(params['num_tcn_layers']))])

        self.log(f"Trial {trial_id}: {' '.join(cmd)}")

        try:
            # Run the command and capture output (stream stderr for progress)
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=3600,  # 1 hour max per trial
            )
            output = result.stdout

            # Extract loss using FINAL_LOSS pattern
            match = re.search(r"FINAL_LOSS:\s*([\d\.eE\-\+]+)", output)
            if match:
                loss = float(match.group(1))
                self.log(f"Trial {trial_id} completed: loss={loss:.6f}")
                return loss
            else:
                self.log(f"Could not find FINAL_LOSS in output. Last 500 chars:\n{output[-500:]}")
                return None

        except subprocess.TimeoutExpired:
            self.log(f"Trial {trial_id} timed out after 1 hour")
            return None
        except subprocess.CalledProcessError as e:
            self.log(f"Training failed for trial {trial_id}: exit code {e.returncode}")
            if e.stderr:
                self.log(f"Stderr (last 500): {e.stderr[-500:]}")
            return None

    def run(self):
        """Main loop for the Agent worker."""
        self.connect()
        self.log("Started. Ready for HPO tasks...")
        
        # Start heartbeat sender thread
        heartbeat_thread = threading.Thread(target=self._heartbeat_sender, daemon=True)
        heartbeat_thread.start()

        try:
            while True:
                if not self.sock:
                    self.connect()
                    # Re-start heartbeat if connection was lost and re-established
                    if not heartbeat_thread.is_alive():
                        heartbeat_thread = threading.Thread(target=self._heartbeat_sender, daemon=True)
                        heartbeat_thread.start()


                opcode, data = self.request_task()
                
                if opcode == OP_HALT:
                    self.log("Kernel sent HALT.")
                    break
                
                elif opcode == OP_TASK_RSP:
                    params = data
                    self.log(f"Received Params: {params}")
                    
                    loss = self.run_training(params)
                    
                    if loss is not None:
                        self.log(f"Training complete. Loss: {loss}")
                        self.send_result(params, loss)
                    else:
                        self.log("Training failed or produced no loss.")
                
                else:
                    self.log("Invalid response or disconnected.")
                    self.sock.close()
                    self.sock = None
                    time.sleep(1)

        except KeyboardInterrupt:
            self.log("Stopping...")
        finally:
            self.is_alive = False # Stop heartbeat thread
            if self.sock: self.sock.close()

def start_worker(ip, port, worker_id):
    agent = HPOAgent(kernel_ip=ip, port=port, worker_id=worker_id)
    agent.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="HPO Agent")
    parser.add_argument("ip", nargs='?', default='127.0.0.1', help="Kernel IP Address")
    parser.add_argument("--port", type=int, default=9999, help="Kernel Port")
    parser.add_argument("--threads", type=int, default=1, help="Number of concurrent worker threads")
    
    args = parser.parse_args()

    print(f"[Main] Starting Agent with {args.threads} threads targeting {args.ip}:{args.port}")

    threads = []
    try:
        for i in range(args.threads):
            t = threading.Thread(target=start_worker, args=(args.ip, args.port, i+1))
            t.daemon = True # Daemon threads exit when main exits
            t.start()
            threads.append(t)
            time.sleep(0.2) # Stagger start slightly

        # Keep main thread alive
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[Main] Shutting down agent...")