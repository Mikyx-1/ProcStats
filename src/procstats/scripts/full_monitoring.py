import logging
import multiprocessing as mp
import time
from typing import Any, Callable, Dict, Tuple

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

    def monitor_resources(self, result_container: list, timeout: float = 12.0):
        """Monitor CPU, RAM, and GPU resources for the given PID."""
        start_time = time.time()
        cpu_ram_result_container = mp.Manager().list()
        gpu_result_container = mp.Manager().list()

        # Start CPU/RAM monitoring process
        cpu_ram_proc = mp.Process(
            target=monitor_cpu_and_ram_by_pid_advanced,
            args=(self.pid, self.base_interval, cpu_ram_result_container),
        )
        cpu_ram_proc.start()

        # Start GPU monitoring in a separate process if available
        gpu_proc = None
        if self.gpu_monitor.nvidia_initialized:
            gpu_proc = mp.Process(
                target=self._monitor_gpu_process,
                args=(self.pid, self.base_interval, timeout, gpu_result_container),
            )
            gpu_proc.start()

        # Wait for the target process to complete or timeout
        try:
            target_proc = psutil.Process(self.pid)
            target_proc.wait(timeout=timeout)
        except (psutil.NoSuchProcess, psutil.TimeoutExpired):
            self.logger.info(f"Process {self.pid} either terminated or timed out")

        # Ensure monitoring processes complete
        cpu_ram_proc.join()
        if gpu_proc:
            gpu_proc.join()

        # Combine results
        result = {
            "cpu_max": 0,
            "cpu_avg": 0,
            "cpu_p95": 0,
            "ram_max": 0,
            "ram_avg": 0,
            "ram_p95": 0,
            "num_cores": mp.cpu_count(),
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
        if cpu_ram_result_container:
            cpu_ram_data = cpu_ram_result_container[0]
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
        if gpu_result_container:
            gpu_result = gpu_result_container[0]
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

        result_container.append(result)

    def _monitor_gpu_process(
        self, pid: int, interval: float, timeout: float, result_container: list
    ):
        """Run GPU monitoring in a separate process to ensure proper NVIDIA ML initialization."""
        try:
            gpu_monitor = GPUMonitor()  # Reinitialize in new process
            result = gpu_monitor.monitor_gpu_utilisation_by_pid(pid, interval, timeout)
            result_container.append(result)
        except Exception as e:
            self.logger.error(f"GPU monitoring failed: {e}")
            result_container.append(
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


def monitor_function_resources(
    target: Callable[..., Any],
    args: Tuple = (),
    kwargs: Dict[str, Any] = None,
    base_interval: float = 0.05,
    timeout: float = 12.0,
) -> Dict[str, Any]:
    """Monitor resources used by a target function."""
    if kwargs is None:
        kwargs = {}

    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    result_container = manager.list()

    # Start the target function process
    process = mp.Process(target=target, args=args, kwargs=kwargs)
    process.start()

    # Start comprehensive monitoring
    monitor = ComprehensiveMonitor(process.pid, base_interval)
    monitor_proc = mp.Process(
        target=monitor.monitor_resources, args=(result_container, timeout)
    )
    monitor_proc.start()

    # Wait for the target process to complete or timeout
    process.join(timeout=timeout)
    if process.is_alive():
        process.terminate()
        process.join()
        monitor.logger.info(f"Target process {process.pid} terminated due to timeout")

    # Wait for monitoring to complete
    monitor_proc.join()

    return (
        result_container[0]
        if result_container
        else {
            "cpu_max": 0,
            "cpu_avg": 0,
            "cpu_p95": 0,
            "ram_max": 0,
            "ram_avg": 0,
            "ram_p95": 0,
            "num_cores": mp.cpu_count(),
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


if __name__ == "__main__":
    pass
