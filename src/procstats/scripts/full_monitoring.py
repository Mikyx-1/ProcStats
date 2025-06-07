import logging
import multiprocessing as mp
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable, Dict, Tuple

import dill
import psutil

from .cpu_ram_monitoring import (AdaptiveMonitor,
                                 monitor_cpu_and_ram_by_pid_advanced)
from .gpu_monitoring import GPUMonitor
from .system_info import SystemInfo

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ComprehensiveMonitor:
    def __init__(self, pid: int, base_interval: float = 0.05):
        self.logger = logging.getLogger(__name__)
        self.pid = pid
        self.base_interval = base_interval
        self.cpu_monitor = AdaptiveMonitor(pid, base_interval)
        self.gpu_monitor = GPUMonitor()
        self.system_info = SystemInfo()
        self.max_processes = 0
        self.tracked_pids = set()

    def _get_all_process_pids(self):
        """Get all PIDs in the process tree."""
        try:
            parent = psutil.Process(self.pid)
            children = parent.children(recursive=True)
            all_pids = [self.pid] + [child.pid for child in children]
            self.tracked_pids.update(all_pids)
            return all_pids
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return []

    def _track_process_count(self):
        """Track the maximum number of processes in the process tree."""
        try:
            all_pids = self._get_all_process_pids()
            current_count = len(all_pids)

            # Update max processes if current count is higher
            if current_count > self.max_processes:
                self.max_processes = current_count

            return current_count
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0

    def monitor_resources(self, result_container: queue.Queue, timeout: float = 12.0):
        """Monitor CPU, RAM, and GPU resources for the given PID."""
        start_time = time.time()
        cpu_ram_result_container = queue.Queue()
        gpu_result_container = queue.Queue()

        # Wrapper to adapt Queue to list-like interface for monitor_cpu_and_ram_by_pid_advanced
        class QueueWrapper:
            def __init__(self, q):
                self.q = q

            def append(self, item):
                self.q.put(item)

        # Start CPU/RAM monitoring thread
        cpu_ram_queue_wrapper = QueueWrapper(cpu_ram_result_container)
        cpu_ram_thread = threading.Thread(
            target=monitor_cpu_and_ram_by_pid_advanced,
            args=(self.pid, self.base_interval, cpu_ram_queue_wrapper),
        )
        cpu_ram_thread.start()

        # Start GPU monitoring in a separate thread if available
        gpu_thread = None
        if self.gpu_monitor.nvidia_initialized:
            gpu_thread = threading.Thread(
                target=self._monitor_gpu_process,
                args=(self.pid, self.base_interval, timeout, gpu_result_container),
            )
            gpu_thread.start()

        # Start process counting thread
        process_count_stop_event = threading.Event()
        process_count_thread = threading.Thread(
            target=self._monitor_process_count,
            args=(process_count_stop_event, self.base_interval),
        )
        process_count_thread.start()

        # Wait for the target process to complete or timeout
        try:
            target_proc = psutil.Process(self.pid)
            target_proc.wait(timeout=timeout)
        except (psutil.NoSuchProcess, psutil.TimeoutExpired):
            self.logger.info(f"Process {self.pid} either terminated or timed out")

        # Stop process counting and ensure monitoring threads complete
        process_count_stop_event.set()
        cpu_ram_thread.join()
        process_count_thread.join()
        if gpu_thread:
            gpu_thread.join()

        # Combine results
        result = {
            "cpu_max": 0,
            "cpu_avg": 0,
            "cpu_p95": 0,
            "ram_max": 0,
            "ram_avg": 0,
            "ram_p95": 0,
            "num_cores": psutil.cpu_count(),
            "measurements_taken": 0,
            "data_quality_score": 0,
            "gpu_max_util": {},
            "gpu_mean_util": {},
            "vram_max_mb": {},
            "vram_mean_mb": {},
            "start_time": start_time,
            "duration": time.time() - start_time,
            "timeout_reached": time.time() - start_time >= timeout,
            "system_info": self.system_info.get_all_info(),
            "num_processes": self.max_processes,
            "tracked_pids": list(self.tracked_pids),
        }

        # Update with CPU/RAM results
        if not cpu_ram_result_container.empty():
            cpu_ram_data = cpu_ram_result_container.get()
            result.update(
                {
                    "cpu_max": cpu_ram_data["cpu_max"],
                    "cpu_avg": cpu_ram_data["cpu_avg"],
                    "cpu_p95": cpu_ram_data["cpu_p95"],
                    "ram_max": cpu_ram_data["ram_max"],
                    "ram_avg": cpu_ram_data["ram_avg"],
                    "ram_p95": cpu_ram_data["ram_p95"],
                    "num_cores": cpu_ram_data["num_cores"],
                    "measurements_taken": cpu_ram_data["measurements_taken"],
                    "data_quality_score": cpu_ram_data["data_quality_score"],
                }
            )

        # Update with GPU results
        if not gpu_result_container.empty():
            gpu_result = gpu_result_container.get()
            result.update(
                {
                    "gpu_max_util": gpu_result["gpu_max_util"],
                    "gpu_mean_util": gpu_result["gpu_mean_util"],
                    "vram_max_mb": gpu_result["vram_max_mb"],
                    "vram_mean_mb": gpu_result["vram_mean_mb"],
                    "duration": max(result["duration"], gpu_result["duration"]),
                    "timeout_reached": result["timeout_reached"]
                    or gpu_result["timeout"],
                }
            )

        result_container.put(result)

    def _monitor_process_count(self, stop_event, interval):
        """Continuously monitor process count in a separate thread."""
        while not stop_event.is_set():
            self._track_process_count()
            time.sleep(interval)

    def _monitor_gpu_process(
        self, pid: int, interval: float, timeout: float, result_container: queue.Queue
    ):
        """Run GPU monitoring in a separate thread to ensure proper NVIDIA ML initialization."""
        try:
            gpu_monitor = GPUMonitor()  # Reinitialize in new thread
            # Monitor all PIDs in the process tree, not just the parent
            result = gpu_monitor.monitor_gpu_utilisation_by_pid(pid, interval, timeout)
            result_container.put(result)
        except Exception as e:
            self.logger.error(f"GPU monitoring failed: {e}")
            result_container.put(
                {
                    "gpu_max_util": {},
                    "gpu_mean_util": {},
                    "vram_max_mb": {},
                    "vram_mean_mb": {},
                    "start_time": time.time(),
                    "duration": 0.0,
                    "timeout": False,
                    "exit_code": False,
                }
            )


def _run_serialized_function(serialized_path: str, output_path: str):
    """
    This helper is just the Python command string passed to subprocess,
    it loads serialized function, runs it, writes result to output_path.
    """

    code = f"""
import dill
import sys
import traceback
import os

# Print current PID for debugging
print(f"[Subprocess] Running with PID: {{os.getpid()}}", flush=True)

try:
    with open({serialized_path!r}, 'rb') as f:
        func, args, kwargs = dill.load(f)
    result = func(*args, **kwargs)
    with open({output_path!r}, 'wb') as out_f:
        dill.dump({{'result': result, 'error': None}}, out_f)
except Exception as e:
    with open({output_path!r}, 'wb') as out_f:
        dill.dump({{'result': None, 'error': traceback.format_exc()}}, out_f)
    sys.exit(1)
"""
    return code


def _kill_process_tree(pid: int, logger):
    """Kill a process and all its children recursively."""
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)

        # Kill children first
        for child in children:
            try:
                logger.info(f"Terminating child process {child.pid}")
                child.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        # Kill parent
        try:
            logger.info(f"Terminating parent process {parent.pid}")
            parent.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        # Wait for graceful termination
        _, alive = psutil.wait_procs(children + [parent], timeout=3)

        # Force kill any remaining processes
        for p in alive:
            try:
                logger.info(f"Force killing process {p.pid}")
                p.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    except psutil.NoSuchProcess:
        logger.info(f"Process {pid} already terminated")
    except Exception as e:
        logger.error(f"Error killing process tree for PID {pid}: {e}")


def monitor_function_resources(
    target: Callable[..., Any],
    args: Tuple = (),
    kwargs: Dict[str, Any] = None,
    base_interval: float = 0.05,
    timeout: float = 12.0,
) -> Dict[str, Any]:
    if kwargs is None:
        kwargs = {}

    logger = logging.getLogger(__name__)
    result_container = queue.Queue()

    with tempfile.TemporaryDirectory() as tempdir:
        serialized_path = os.path.join(tempdir, "func.dill")
        output_path = os.path.join(tempdir, "output.dill")

        # Serialize the function, args, kwargs together
        with open(serialized_path, "wb") as f:
            dill.dump((target, args, kwargs), f)

        # Create Python code command to run the serialized function
        python_code = _run_serialized_function(serialized_path, output_path)

        # Launch subprocess running python -c 'python_code'
        # Use exec to replace the shell process with Python process
        process = subprocess.Popen(
            [sys.executable, "-c", python_code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid if hasattr(os, 'setsid') else None,  # Create new process group
        )

        logger.info(f"Started subprocess with PID: {process.pid}")

        # Wait a small amount of time for the subprocess to start
        time.sleep(0.1)

        # Start monitoring the subprocess pid
        monitor = ComprehensiveMonitor(process.pid, base_interval)
        monitor_thread = threading.Thread(
            target=monitor.monitor_resources, args=(result_container, timeout)
        )
        monitor_thread.start()

        timeout_occurred = False
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timeout_occurred = True
            logger.info(f"Timeout reached, killing process tree for PID {process.pid}")

            # Kill the entire process tree
            _kill_process_tree(process.pid, logger)

            # Get any remaining output
            try:
                stdout, stderr = process.communicate(timeout=1)
            except subprocess.TimeoutExpired:
                stdout, stderr = b"", b""

            logger.info(f"Process tree for PID {process.pid} terminated due to timeout")

        # Wait for monitoring thread to finish
        monitor_thread.join()

        # Load function execution result from output file (if exists)
        func_result = None
        func_error = None
        if os.path.exists(output_path):
            try:
                with open(output_path, "rb") as f:
                    res = dill.load(f)
                    func_result = res.get("result")
                    func_error = res.get("error")
            except Exception as e:
                logger.error(f"Failed to load function output: {e}")

        # Get monitoring result
        monitor_result = (
            result_container.get()
            if not result_container.empty()
            else {
                "cpu_max": 0,
                "cpu_avg": 0,
                "cpu_p95": 0,
                "ram_max": 0,
                "ram_avg": 0,
                "ram_p95": 0,
                "num_cores": 0,
                "measurements_taken": 0,
                "data_quality_score": 0,
                "gpu_max_util": {},
                "gpu_mean_util": {},
                "vram_max_mb": {},
                "vram_mean_mb": {},
                "start_time": time.time(),
                "duration": 0.0,
                "timeout_reached": True,
                "system_info": SystemInfo().get_all_info(),
            }
        )

        # Ensure timeout status is correctly set
        if timeout_occurred:
            monitor_result["timeout_reached"] = True

        # Attach function result or error to monitoring info if needed
        monitor_result["function_result"] = func_result
        monitor_result["function_error"] = func_error
        monitor_result["stdout"] = (
            stdout.decode("utf-8", errors="ignore") if stdout else ""
        )
        monitor_result["stderr"] = (
            stderr.decode("utf-8", errors="ignore") if stderr else ""
        )

        return monitor_result


if __name__ == "__main__":
    pass