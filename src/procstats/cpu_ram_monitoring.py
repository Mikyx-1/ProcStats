import multiprocessing as mp
import time
from typing import Any, Callable, Dict, Tuple

import psutil


def monitor_cpu_and_ram_by_pid(pid: int, interval: float, result_container: list):

    cpu_usages = []
    ram_usages = []

    try:
        proc = psutil.Process(pid)
        proc.cpu_percent(interval=interval)  # Prime the CPU meter

        while True:
            try:
                if not proc.is_running() or proc.status() == psutil.STATUS_ZOMBIE:
                    break
                cpu_percent = proc.cpu_percent(interval=None)
                ram_usage = proc.memory_info().rss
                cpu_usages.append(cpu_percent)
                ram_usages.append(ram_usage)
                time.sleep(interval)
            except (psutil.NoSuchProcess, psutil.ZombieProcess, ProcessLookupError):
                break
            except Exception as e:
                print(f"[Monitor] Warning: {e}")
                break

    finally:
        result_container.append(
            {
                "cpu_max": max(cpu_usages, default=0),
                "cpu_avg": sum(cpu_usages) / len(cpu_usages) if cpu_usages else 0,
                "ram_max": max(ram_usages, default=0) / 1024**2,
                "ram_avg": (
                    (sum(ram_usages) / len(ram_usages) / 1024**2) if ram_usages else 0
                ),
            }
        )


def monitor_cpu_and_ram_on_function(
    target: Callable[..., Any],
    args: Tuple = (),
    kwargs: Dict[str, Any] = None,
    interval: float = 0.1,
) -> Dict[str, float]:
    """Run a target function and monitor its CPU and RAM usage.

    Args:
        target: The function to execute.
        args: Positional arguments for the target function.
        kwargs: Keyword arguments for the target function.
        interval: Sampling interval in seconds (default: 0.1).

    Returns:
        Dictionary with max/avg CPU usage (%) and RAM usage (MB).
        Returns zeros if monitoring fails or no data is collected.
    """
    if kwargs is None:
        kwargs = {}

    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    result_container = manager.list()

    # Launch target process
    process = mp.Process(target=target, args=args, kwargs=kwargs)
    process.start()

    # Launch monitor process
    monitor_proc = mp.Process(
        target=monitor_cpu_and_ram_by_pid,
        args=(process.pid, interval, result_container),
    )
    monitor_proc.start()

    # Wait for both
    process.join()
    monitor_proc.join()

    return (
        result_container[0]
        if result_container
        else {
            "cpu_max": 0,
            "cpu_avg": 0,
            "ram_max": 0,
            "ram_avg": 0,
        }
    )


def heavy_cpu_gpu_task():
    import os

    import torch

    print("Inside PID:", os.getpid())
    a = torch.randn(5000, 5000)
    for _ in range(10):
        b = torch.matmul(a, a.T)


if __name__ == "__main__":
    resource_usg = monitor_cpu_and_ram_on_function(heavy_cpu_gpu_task, interval=0.01)
    print(resource_usg)
