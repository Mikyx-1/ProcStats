import logging
import multiprocessing as mp
import time
from typing import Any, Callable, Dict, Tuple

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
        gpu_result = None

        # Start CPU/RAM monitoring process
        cpu_ram_proc = mp.Process(
            target=monitor_cpu_and_ram_by_pid_advanced,
            args=(self.pid, self.base_interval, cpu_ram_result_container),
        )
        cpu_ram_proc.start()

        # Start GPU monitoring in a separate process to ensure NVIDIA ML initialization
        if self.gpu_monitor.nvidia_initialized:
            gpu_result_container = mp.Manager().list()
            gpu_proc = mp.Process(
                target=self._monitor_gpu_process,
                args=(self.pid, self.base_interval, timeout, gpu_result_container),
            )
            gpu_proc.start()
        else:
            gpu_proc = None

        # Wait for CPU/RAM monitoring to complete
        cpu_ram_proc.join()
        if gpu_proc:
            gpu_proc.join()
            gpu_result = gpu_result_container[0] if gpu_result_container else None

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
        if gpu_result:
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

    # Wait for processes to complete
    process.join()
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
            "timeout_reached": False,
            "system_info": SystemInfo().get_all_info(),
        }
    )


def heavy_gpu_task():
    import os

    import torch

    print("PID:", os.getpid())
    try:
        a = torch.randn(5000, 5000, device="cuda:1")
        for _ in range(3000):
            b = torch.matmul(a, a.T)
    except RuntimeError as e:
        print(f"GPU task failed: {e}")


if __name__ == "__main__":
    result = monitor_function_resources(
        heavy_gpu_task, base_interval=0.05, timeout=10.0
    )
    print(result)
