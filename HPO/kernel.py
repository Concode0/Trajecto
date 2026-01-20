"""
This module defines the HPOKernel, the central server for the Entangle distributed
Hyperparameter Optimization (HPO) system.

The Kernel is responsible for:
1.  Managing the ASHA (Asynchronous Successive Halving) task queue.
2.  Distributing tasks to connected Agent workers.
3.  Collecting and logging results from Agents.
4.  Monitoring Agent liveness and handling status requests.
"""

import csv
import os
import random
import socket
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats

from protocol import (
    OP_GET_TASK,
    OP_FOUND,
    OP_GET_STATUS,
    OP_HEARTBEAT,
    OP_TASK_RSP,
    OP_HALT,
    OP_STATUS_RSP,
    pack_packet,
    unpack_packet,
)


class HPOKernel:
    """The central coordinator for distributed HPO tasks using ASHA and TPE.

    Attributes:
        server (socket.socket): The TCP server socket.
        best_loss (float): The lowest loss recorded so far.
        best_params (Optional[Dict[str, Any]]): The hyperparameters achieving the best loss.
        lock (threading.Lock): Mutex for thread-safe access to shared state.
        task_queue (List[Dict[str, Any]]): List of pending tasks to evaluate.
        assigned_tasks (Dict[str, Tuple[Dict[str, Any], float]]): Tracks tasks currently assigned to workers.
        worker_last_seen (Dict[str, float]): Timestamps of last heartbeat from each worker.
        results_file (str): Path to the CSV file where results are logged.

        # ASHA Configuration
        rungs (List[int]): The epoch milestones (e.g., [1, 2, 4, 8]).
        reduction_factor (int): The 'eta' parameter (1/eta fraction promoted).
        rung_history (Dict[int, List[float]]): Stores all losses recorded at each rung.
        trials (Dict[str, Dict[str, Any]]): Stores hyperparams for each trial ID.
        completed_trials (List[Dict[str, Any]]): History of completed trials for Bayesian Optimization.
    """

    # Configuration Constants
    DEFAULT_PORT = 9999
    WORKER_TIMEOUT_SEC = 300  # 5 min timeout for longer training runs
    RESULTS_FILENAME = "HPO/hpo_results.csv"  # Store results in HPO directory

    # ASHA Settings - Optimized for Trajecto training (epochs scale)
    # Rung progression: 5 -> 10 -> 20 -> 40 -> 80 epochs
    RUNGS = [5, 10, 20, 40, 80]
    REDUCTION_FACTOR = 2

    # TPE Settings
    MIN_HISTORY_FOR_TPE = 10
    TPE_GOOD_PERCENTILE = 20

    # Hyperparameter Search Space - Optimized for ESKF-TCN Trajecto Model
    # Updated for current system with DWA, delta loss, and context-aware weighting
    PARAM_SPACE = {
        # === Training Parameters ===
        # Learning rate: Narrowed to [3e-5, 3e-4] based on train_eskf.py defaults
        # Current default: 1e-4, search ±3x around it
        "lr": {"type": "log_float", "range": (-4.5, -3.5)},  # 3e-5 to 3e-4

        # Batch size: Keep fixed for memory/GPU stability
        "batch_size": {"type": "fixed", "value": 16},

        # === Regularization ===
        # Dropout: Narrow range around current default (0.15)
        # Too low → overfitting, too high → underfitting
        "dropout": {"type": "float", "range": (0.10, 0.25)},

        # Regularization weight: Tighter range around current default (1e-7)
        # This penalizes large TCN velocity corrections
        "reg_weight": {"type": "log_float", "range": (-7.5, -6.0)},  # 3e-8 to 1e-6

        # === ESKF Parameters ===
        # Mahalanobis threshold: Narrowed to practical range
        # Current default: 30, search around it
        # Lower → stricter gating (fewer outliers), Higher → more lenient
        "mahalanobis_threshold": {"type": "float", "range": (15.0, 45.0)},

        # === TCN Architecture ===
        # Kernel size: Keep 3, 5, 7 (current default: 3)
        # Larger kernel → larger receptive field but slower
        "kernel_size": {"type": "categorical", "values": [3, 5, 7]},

        # TCN channels: Expanded to include higher capacity options
        # Current default: 96, add 64 (lighter), 128 (heavier) for exploration
        "tcn_channel_size": {"type": "categorical", "values": [64, 96, 128]},

        # === Loss Weights (DWA will adjust these, but initial values matter) ===
        # Magnitude loss initial weight
        "w_mag": {"type": "float", "range": (0.5, 2.0)},

        # Cosine (direction) loss initial weight
        "w_cos": {"type": "float", "range": (0.5, 2.0)},

        # ZUPT loss initial weight
        "w_zupt": {"type": "float", "range": (0.2, 1.0)},

        # Covariance NLL loss initial weight
        "w_cov": {"type": "float", "range": (0.005, 0.05)},

        # FFT loss initial weight
        "w_fft": {"type": "float", "range": (0.2, 1.0)},

        # Delta loss (semi-loop closure) weight - FIXED by DWA
        "w_delta": {"type": "float", "range": (0.3, 0.7)},

        # === ZUPT Parameters ===
        # ZUPT velocity threshold: when to consider "stopped"
        # Current default: 0.005 m/s
        "zupt_vel_threshold": {"type": "float", "range": (0.003, 0.010)},
    }

    def __init__(self, port: int = DEFAULT_PORT):
        """Initialize the HPO Kernel.

        Args:
            port (int): The TCP port to listen on.
        """
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(('0.0.0.0', port))
        self.server.listen()

        # HPO State
        self.best_loss = float('inf')
        self.best_params: Optional[Dict[str, Any]] = None
        self.lock = threading.Lock()

        # ASHA State
        self.rung_history: Dict[int, List[float]] = {r: [] for r in self.RUNGS}
        self.trials: Dict[str, Dict[str, Any]] = {}  # trial_id -> params
        self.completed_trials: List[Dict[str, Any]] = []

        # Task Queue
        self.task_queue: List[Dict[str, Any]] = []
        self.assigned_tasks: Dict[str, Tuple[Dict[str, Any], float]] = {}
        self.worker_last_seen: Dict[str, float] = {}

        # Initialize Results Logging
        self._init_logging()

        # Populate initial tasks
        self.populate_base_rung(n=20)

        print(f"[Kernel] ASHA Kernel Initialized. Listening on {port}.")
        print(f"[Kernel] Rungs: {self.RUNGS}, Reduction Factor: {self.REDUCTION_FACTOR}")

        # Start Watchdog
        threading.Thread(target=self.watchdog, daemon=True).start()

    def _init_logging(self):
        """Initializes the CSV results file if it doesn't exist."""
        if not os.path.exists(self.RESULTS_FILENAME):
            with open(self.RESULTS_FILENAME, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'loss', 'rung', 'trial_id', 'params'])

    def suggest_hyperparameters(self) -> Dict[str, Any]:
        """Suggests hyperparameters using TPE (Tree-structured Parzen Estimator).

        Returns:
            Dict[str, Any]: A dictionary of suggested hyperparameters.
        """
        # Fallback to random search if insufficient history
        if len(self.completed_trials) < self.MIN_HISTORY_FOR_TPE:
            return self._get_random_params()

        # Separate trials into "good" and "bad" based on loss percentile
        losses = [t['loss'] for t in self.completed_trials]
        threshold = np.percentile(losses, self.TPE_GOOD_PERCENTILE)

        good_trials = [t for t in self.completed_trials if t['loss'] <= threshold]
        bad_trials = [t for t in self.completed_trials if t['loss'] > threshold]

        new_params = {}

        for name, conf in self.PARAM_SPACE.items():
            if conf['type'] == 'fixed':
                new_params[name] = conf['value']

            elif conf['type'] == 'categorical':
                new_params[name] = self._sample_categorical(name, conf, good_trials)

            elif conf['type'] in ['float', 'log_float']:
                new_params[name] = self._sample_numerical(name, conf, good_trials, bad_trials)

        return new_params

    def _sample_categorical(self, name: str, conf: Dict, good_trials: List[Dict]) -> Any:
        """Samples a categorical parameter based on historical frequency."""
        counts = {v: 0 for v in conf['values']}
        for t in good_trials:
            val = t['params'].get(name)
            if val in counts:
                counts[val] += 1

        # Laplace smoothing
        total = sum(counts.values()) + len(conf['values'])
        probs = [(counts[v] + 1) / total for v in conf['values']]
        return np.random.choice(conf['values'], p=probs)

    def _sample_numerical(self, name: str, conf: Dict, good_trials: List[Dict], bad_trials: List[Dict]) -> float:
        """Samples a numerical parameter using KDE-based Expected Improvement."""
        is_log = (conf['type'] == 'log_float')

        vals_good = [t['params'].get(name) for t in good_trials]
        vals_bad = [t['params'].get(name) for t in bad_trials]

        if is_log:
            vals_good = [np.log10(v) for v in vals_good if v > 0]
            vals_bad = [np.log10(v) for v in vals_bad if v > 0]

        try:
            # Ensure enough variance for KDE
            if len(set(vals_good)) < 2 or len(set(vals_bad)) < 2:
                raise ValueError("Insufficient variance for KDE")

            kde_good = stats.gaussian_kde(vals_good, bw_method='scott')
            kde_bad = stats.gaussian_kde(vals_bad, bw_method='scott')

            # Sample candidates from "good" distribution
            candidates = kde_good.resample(20)[0]

            # Clip to defined range
            low, high = conf['range']
            candidates = np.clip(candidates, low, high)

            # Calculate Expected Improvement (log-likelihood ratio)
            log_l = kde_good.logpdf(candidates)
            log_g = kde_bad.logpdf(candidates)
            scores = log_l - log_g

            best_idx = np.argmax(scores)
            val = candidates[best_idx]

            if is_log:
                val = 10 ** val
            return float(val)

        except Exception:
            # Fallback to random sampling on error
            val = random.uniform(*conf['range'])
            if is_log:
                val = 10 ** val
            return float(val)

    def _get_random_params(self) -> Dict[str, Any]:
        """Generates random hyperparameters based on the defined space."""
        params = {}
        for name, conf in self.PARAM_SPACE.items():
            if conf['type'] == 'fixed':
                params[name] = conf['value']
            elif conf['type'] == 'categorical':
                params[name] = random.choice(conf['values'])
            elif conf['type'] == 'float':
                params[name] = random.uniform(*conf['range'])
            elif conf['type'] == 'log_float':
                params[name] = 10 ** random.uniform(*conf['range'])
        return params

    def populate_base_rung(self, n: int = 10):
        """Adds initial trials to the base rung.

        Args:
            n (int): Number of trials to generate.
        """
        print(f"[ASHA] Generating {n} trials for Rung {self.RUNGS[0]}...")
        for _ in range(n):
            trial_id = str(uuid.uuid4())[:8]

            params = self.suggest_hyperparameters()
            # Ensure model type is set if not in param space
            if "model" not in params:
                params["model"] = "eskf_tcn"

            params["trial_id"] = trial_id
            params["target_rung"] = 0  # Index in self.RUNGS
            params["epochs"] = self.RUNGS[0]

            self.trials[trial_id] = params
            self.task_queue.append(params)

    def check_promotion(self, loss: float, rung_idx: int, trial_id: str):
        """Checks if a trial qualifies for promotion to the next rung.

        Args:
            loss (float): The loss value of the completed trial.
            rung_idx (int): The current rung index of the trial.
            trial_id (str): The unique ID of the trial.
        """
        if rung_idx >= len(self.RUNGS) - 1:
            return  # Max rung reached

        current_rung_losses = self.rung_history[self.RUNGS[rung_idx]]

        # Simple ASHA logic: Promote if in top 1/REDUCTION_FACTOR
        sorted_losses = sorted(current_rung_losses)
        cutoff_index = len(sorted_losses) // self.REDUCTION_FACTOR

        # Ensure we have a valid cutoff index
        cutoff_val = sorted_losses[min(cutoff_index, len(sorted_losses) - 1)]

        if loss <= cutoff_val:
            next_rung_idx = rung_idx + 1
            next_epochs = self.RUNGS[next_rung_idx]
            print(f"[ASHA] Promoting Trial {trial_id} to Rung {next_rung_idx} ({next_epochs} epochs)")

            # Create new task with updated epochs and rung
            new_params = self.trials[trial_id].copy()
            new_params["epochs"] = next_epochs
            new_params["target_rung"] = next_rung_idx

            self.task_queue.append(new_params)
        else:
            print(f"[ASHA] Trial {trial_id} stopped at Rung {rung_idx} (Loss {loss:.4f} > Cutoff {cutoff_val:.4f})")

    def watchdog(self):
        """Monitors worker liveness and replenishes the task queue."""
        while True:
            time.sleep(5)
            now = time.time()
            with self.lock:
                self._check_timeouts(now)
                self._replenish_queue()

    def _check_timeouts(self, now: float):
        """Re-queues tasks from timed-out workers."""
        timed_out_workers = []
        for addr_str, (params, _) in list(self.assigned_tasks.items()):
            last_seen = self.worker_last_seen.get(addr_str, 0)
            if (now - last_seen) > self.WORKER_TIMEOUT_SEC:
                print(f"[Watchdog] Worker {addr_str} timed out. Re-queueing {params['trial_id']}")
                self.task_queue.append(params)
                timed_out_workers.append(addr_str)

        for addr_str in timed_out_workers:
            del self.assigned_tasks[addr_str]
            if addr_str in self.worker_last_seen:
                del self.worker_last_seen[addr_str]

    def _replenish_queue(self):
        """Ensures the task queue has a minimum number of tasks."""
        if len(self.task_queue) < 5:
            self.populate_base_rung(n=1)

    def _handle_result(self, addr_str: str, payload: Tuple[Dict, float]):
        """Processes a result received from a worker."""
        try:
            params, loss = payload
            rung_idx = params.get("target_rung", 0)
            epochs = params.get("epochs")
            trial_id = params.get("trial_id")

            print(f"[Result] {trial_id} | Rung {rung_idx} ({epochs} ep) | Loss: {loss:.6f}")

            with self.lock:
                if addr_str in self.assigned_tasks:
                    del self.assigned_tasks[addr_str]

                # Log result
                with open(self.RESULTS_FILENAME, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([time.time(), loss, rung_idx, trial_id, str(params)])

                # Update best
                if loss < self.best_loss:
                    self.best_loss = loss
                    self.best_params = params
                    print(f"[!!!] NEW BEST: {loss:.6f} (Trial {trial_id})")

                # ASHA Logic
                self.rung_history[self.RUNGS[rung_idx]].append(loss)
                self.check_promotion(loss, rung_idx, trial_id)

                # Store for Bayesian Optimization
                self.completed_trials.append({'params': params, 'loss': loss})

        except ValueError:
            print(f"[Err] Invalid result payload from {addr_str}")

    def handle_worker(self, conn: socket.socket, addr: Tuple[str, int]):
        """Handles connection for a worker or monitor.

        Args:
            conn (socket.socket): The client connection socket.
            addr (Tuple[str, int]): The client address.
        """
        addr_str = f"{addr[0]}:{addr[1]}"
        print(f"[Kernel] Connection from {addr_str}")

        with self.lock:
            self.worker_last_seen[addr_str] = time.time()

        try:
            while True:
                opcode, payload = unpack_packet(conn)
                if opcode is None:
                    break  # Disconnect

                with self.lock:
                    self.worker_last_seen[addr_str] = time.time()

                if opcode == OP_GET_TASK:
                    params = None
                    with self.lock:
                        if self.task_queue:
                            params = self.task_queue.pop(0)
                            self.assigned_tasks[addr_str] = (params, time.time())

                    if params:
                        conn.sendall(pack_packet(OP_TASK_RSP, params))
                        print(f"[Sched] Assigned Rung {params['target_rung']} ({params['epochs']} ep) to {addr_str}")
                    else:
                        conn.sendall(pack_packet(OP_HALT))
                        break

                elif opcode == OP_FOUND:
                    self._handle_result(addr_str, payload)

                elif opcode == OP_GET_STATUS:
                    # Monitor request
                    status = {
                        "queue_size": len(self.task_queue),
                        "active_workers": len(self.assigned_tasks),
                        "best_loss": self.best_loss,
                        "workers": [
                            {"addr": k, "task": v[0]['trial_id'], "seen": int(time.time() - self.worker_last_seen.get(k, 0))}
                            for k, v in self.assigned_tasks.items()
                        ],
                        "rung_stats": {r: len(l) for r, l in self.rung_history.items()}
                    }
                    conn.sendall(pack_packet(OP_STATUS_RSP, status))

                elif opcode == OP_HEARTBEAT:
                    pass

        except Exception as e:
            print(f"[Err] Connection {addr_str} error: {e}")
            with self.lock:
                if addr_str in self.assigned_tasks:
                    params, _ = self.assigned_tasks.pop(addr_str)
                    self.task_queue.append(params)
        finally:
            conn.close()
            with self.lock:
                if addr_str in self.worker_last_seen:
                    del self.worker_last_seen[addr_str]
            print(f"[Kernel] {addr_str} disconnected.")

    def run(self):
        """Starts the Kernel server loop."""
        print(f"[Kernel] Listening on 0.0.0.0:{self.server.getsockname()[1]}...")
        while True:
            try:
                conn, addr = self.server.accept()
                threading.Thread(target=self.handle_worker, args=(conn, addr), daemon=True).start()
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"[Kernel] Accept error: {e}")

if __name__ == '__main__':
    kernel = HPOKernel()
    kernel.run()