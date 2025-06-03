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

from cpu_ram_monitoring import (AdaptiveMonitor,
                                monitor_cpu_and_ram_by_pid_advanced)
from gpu_monitoring import GPUMonitor
from system_info import SystemInfo

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

        # Wait for the target process to complete or timeout
        try:
            target_proc = psutil.Process(self.pid)
            target_proc.wait(timeout=timeout)
        except (psutil.NoSuchProcess, psutil.TimeoutExpired):
            self.logger.info(f"Process {self.pid} either terminated or timed out")

        # Ensure monitoring threads complete
        cpu_ram_thread.join()
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

    def _monitor_gpu_process(
        self, pid: int, interval: float, timeout: float, result_container: queue.Queue
    ):
        """Run GPU monitoring in a separate thread to ensure proper NVIDIA ML initialization."""
        try:
            gpu_monitor = GPUMonitor()  # Reinitialize in new thread
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


def monitor_function_resources(
    target: Callable[..., Any],
    args: Tuple = (),
    kwargs: Dict[str, Any] = None,
    base_interval: float = 0.05,
    timeout: float = 12.0,
) -> Dict[str, Any]:
    if kwargs is None:
        kwargs = {}

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
        process = subprocess.Popen(
            [sys.executable, "-c", python_code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Start monitoring the subprocess pid
        monitor = ComprehensiveMonitor(process.pid, base_interval)
        monitor_thread = threading.Thread(
            target=monitor.monitor_resources, args=(result_container, timeout)
        )
        monitor_thread.start()

        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            monitor.logger.info(f"Subprocess {process.pid} killed due to timeout")

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
                monitor.logger.error(f"Failed to load function output: {e}")

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

        # Attach function result or error to monitoring info if needed
        monitor_result["function_result"] = func_result
        monitor_result["function_error"] = func_error

        return monitor_result


if __name__ == "__main__":
    pass
