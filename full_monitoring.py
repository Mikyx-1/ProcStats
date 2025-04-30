import logging
import multiprocessing as mp
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from cpu_ram_monitoring import monitor_cpu_and_ram_by_pid
from gpu_monitoring import monitor_gpu_utilization_by_pid, validate_gpu_index

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def wrap_target(target: Callable, *args, **kwargs):
    """Wrap the target function to capture and log exceptions."""
    try:
        target(*args, **kwargs)
    except Exception as e:
        logging.error(f"Target function failed: {str(e)}")
        raise


def get_all_gpu_indices() -> List[int]:
    """Return a list of all available GPU indices using pynvml."""
    try:
        from pynvml import nvmlDeviceGetCount, nvmlInit, nvmlShutdown

        nvmlInit()
        device_count = nvmlDeviceGetCount()
        nvmlShutdown()
        return list(range(device_count))
    except Exception as e:
        logging.error(f"Failed to detect GPUs: {str(e)}")
        return []


def full_resource_monitor(
    target: Callable,
    args: Tuple = (),
    kwargs: Optional[Dict[str, Any]] = None,
    interval: float = 0.01,
    gpu_index: Union[int, List[int], None] = 0,
) -> Tuple[Dict[str, float], Union[Dict[str, float], List[Dict[str, float]]]]:
    """
    Monitor CPU, RAM, and GPU usage of a target process.

    Args:
        target: Function to monitor.
        args: Positional arguments for the target function.
        kwargs: Keyword arguments for the target function.
        interval: Monitoring interval in seconds.
        gpu_index: GPU index (int), list of indices (List[int]), or None to monitor all GPUs.

    Returns:
        Tuple of (cpu_results, gpu_results), where:
        - cpu_results is a dictionary with CPU and RAM metrics.
        - gpu_results is a dictionary (single GPU) or list of dictionaries (multiple GPUs).
    """
    if kwargs is None:
        kwargs = {}

    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    cpu_and_ram_result_container = manager.list()
    gpu_result_containers = []  # List to hold result containers for each GPU

    # Initialize process variables to avoid UnboundLocalError
    process = None
    cpu_and_ram_monitor_process = None
    gpu_monitor_processes = []

    # Prepare GPU indices
    gpu_indices = []
    if gpu_index is None:
        gpu_indices = get_all_gpu_indices()
        logging.info(f"Monitoring all available GPUs: {gpu_indices}")
    elif isinstance(gpu_index, int):
        gpu_indices = [gpu_index]
        logging.info(f"Monitoring single GPU: {gpu_index}")
    elif isinstance(gpu_index, list):
        gpu_indices = gpu_index
        logging.info(f"Monitoring specified GPUs: {gpu_index}")
    else:
        logging.error(
            f"Invalid gpu_index type: {type(gpu_index)}. Must be int, List[int], or None."
        )
        gpu_indices = []

    # Validate each GPU index
    valid_gpu_indices = []
    for idx in gpu_indices:
        if not isinstance(idx, int):
            logging.error(f"Invalid GPU index: {idx}. Must be an integer.")
            continue
        if validate_gpu_index(idx):
            valid_gpu_indices.append(idx)
        else:
            logging.warning(f"Skipping GPU index {idx} due to validation failure")

    # Create result containers for each valid GPU
    for _ in valid_gpu_indices:
        gpu_result_containers.append(manager.list())

    try:
        # Launch the target process
        process = mp.Process(target=wrap_target, args=(target,) + args, kwargs=kwargs)
        process.start()
        pid = process.pid
        if pid is None:
            logging.error("Failed to obtain PID for target process")
            raise ValueError("Target process PID is None")
        logging.info(f"Target process started with PID {pid}")

        # Launch CPU/RAM monitoring process
        cpu_and_ram_monitor_process = mp.Process(
            target=monitor_cpu_and_ram_by_pid,
            args=(pid, interval, cpu_and_ram_result_container),
        )
        cpu_and_ram_monitor_process.start()
        logging.info("CPU and RAM monitoring started")

        # Launch GPU monitoring processes
        for idx, container in zip(valid_gpu_indices, gpu_result_containers):
            gpu_proc = mp.Process(
                target=monitor_gpu_utilization_by_pid,
                args=(idx, pid, interval, container),
            )
            gpu_proc.start()
            gpu_monitor_processes.append(gpu_proc)
            logging.info(f"GPU monitoring started for GPU index {idx}")

        # Join processes
        process.join()
        logging.info("Target process completed")
        cpu_and_ram_monitor_process.join()
        logging.info("CPU/RAM monitoring process completed")
        for idx, proc in zip(valid_gpu_indices, gpu_monitor_processes):
            proc.join()
            logging.info(f"GPU monitoring process for GPU index {idx} completed")

        # Collect results
        cpu_results = (
            cpu_and_ram_result_container[0]
            if cpu_and_ram_result_container
            else {"cpu_max": 0, "cpu_avg": 0, "ram_max": 0.0, "ram_avg": 0}
        )

        if not valid_gpu_indices:
            gpu_results = []  # Empty list if no valid GPUs
            logging.info("No valid GPUs monitored")
        elif len(valid_gpu_indices) == 1:
            gpu_results = (
                gpu_result_containers[0][0]
                if gpu_result_containers[0]
                else {
                    "gpu_util_mean": 0,
                    "gpu_util_max": 0,
                    "vram_usage_mean": 0,
                    "vram_usage_max": 0,
                }
            )
        else:
            gpu_results = [
                (
                    container[0]
                    if container
                    else {
                        "gpu_util_mean": 0,
                        "gpu_util_max": 0,
                        "vram_usage_mean": 0,
                        "vram_usage_max": 0,
                    }
                )
                for container in gpu_result_containers
            ]

        return cpu_results, gpu_results

    except Exception as e:
        logging.error(f"Monitoring failed: {str(e)}")
        raise
    finally:
        # Ensure all processes are terminated
        for proc in (
            ([process] if process else [])
            + ([cpu_and_ram_monitor_process] if cpu_and_ram_monitor_process else [])
            + gpu_monitor_processes
        ):
            if proc and proc.is_alive():
                proc.terminate()
                proc.join()
        logging.info("All processes terminated")


def heavy_cpu_gpu_task():
    import os

    import torch

    time.sleep(2)
    print("Inside PID:", os.getpid())
    try:
        a = torch.randn(5000, 5000, device="cuda:1")
        for _ in range(10):
            b = torch.matmul(a, a.T)
    except RuntimeError as e:
        print(f"GPU task failed: {e}")


if __name__ == "__main__":

    # Example with all GPUs (None)
    result = full_resource_monitor(heavy_cpu_gpu_task, gpu_index=None)
    print("All GPUs Result:", result)
