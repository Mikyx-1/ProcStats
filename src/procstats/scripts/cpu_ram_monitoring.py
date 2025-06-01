import multiprocessing as mp
import time
from typing import Any, Callable, Dict, Tuple

import psutil

from test_burn_cpu import burn_cpu_accurate


def get_cpu_cores():
    """Get the number of CPU cores."""
    return psutil.cpu_count() or 1


def monitor_cpu_and_ram_by_pid(pid: int, interval: float, result_container: list):
    cpu_usages = []
    ram_usages = []
    num_cores = get_cpu_cores()
    start_time = time.time()
    timeout = 12.0  # Allow extra time beyond 10s for burn_cpu_accurate

    try:
        parent_proc = psutil.Process(pid)
        all_procs = [parent_proc] + parent_proc.children(recursive=True)

        # Prime the CPU meter
        for proc in all_procs:
            try:
                proc.cpu_percent(interval=interval)
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                continue

        # Stabilization delay
        time.sleep(interval)

        while time.time() - start_time < timeout:
            try:
                if (
                    not parent_proc.is_running()
                    or parent_proc.status() == psutil.STATUS_ZOMBIE
                ):
                    print(f"[Monitor] Parent process {pid} terminated early")
                    break

                # Refresh process list to catch new children
                all_procs = [parent_proc] + parent_proc.children(recursive=True)
                total_cpu_percent = 0.0
                total_ram_usage = 0.0

                for proc in all_procs:
                    try:
                        if (
                            not proc.is_running()
                            or proc.status() == psutil.STATUS_ZOMBIE
                        ):
                            continue
                        cpu_percent = proc.cpu_percent(interval=interval)
                        ram_usage = proc.memory_info().rss / 1024**2
                        total_cpu_percent += cpu_percent
                        total_ram_usage += ram_usage
                    except (
                        psutil.NoSuchProcess,
                        psutil.ZombieProcess,
                        psutil.AccessDenied,
                    ):
                        continue

                cpu_usages.append(total_cpu_percent)
                ram_usages.append(total_ram_usage)
                time.sleep(interval)

            except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied):
                print(f"[Monitor] Parent process {pid} no longer running")
                break
            except Exception as e:
                print(f"[Monitor] Warning: {e}")
                break

    finally:
        result_container.append(
            {
                "cpu_max": max(cpu_usages, default=0.0),
                "cpu_avg": sum(cpu_usages) / len(cpu_usages) if cpu_usages else 0.0,
                "ram_max": max(ram_usages, default=0.0),
                "ram_avg": sum(ram_usages) / len(ram_usages) if ram_usages else 0.0,
                "num_cores": num_cores,
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
        else {"cpu_max": 0, "cpu_avg": 0, "ram_max": 0, "ram_avg": 0, "num_cores": 1}
    )

def heavy_cpu_gpu_task():
    import os

    import torch

    print("Inside PID:", os.getpid())
    a = torch.randn(5000, 5000)
    for _ in range(10):
        b = torch.matmul(a, a.T)

if __name__ == "__main__":
    resource_usg = monitor_cpu_and_ram_on_function(heavy_cpu_gpu_task, interval=0.1)
    print("\nResource Usage Summary:")
    print(f"CPU Max: {resource_usg['cpu_max']:.2f}% (Normalized)")
    print(f"CPU Avg: {resource_usg['cpu_avg']:.2f}% (Normalized)")
    print(f"RAM Max: {resource_usg['ram_max']:.2f} MB")
    print(f"RAM Avg: {resource_usg['ram_avg']:.2f} MB")
    print(f"Number of Cores: {resource_usg['num_cores']}")
