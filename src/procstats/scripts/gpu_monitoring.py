import logging
import multiprocessing as mp
import time
from typing import Any, Callable, Dict, Tuple

import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

try:
    from pynvml import (NVMLError, NVMLError_NotSupported,
                        nvmlDeviceGetComputeRunningProcesses,
                        nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex,
                        nvmlDeviceGetProcessUtilization, nvmlInit,
                        nvmlShutdown)

    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


def validate_gpu_index(gpu_index: int) -> bool:
    """Validate the GPU index. Return True if valid, False if pynvml is unavailable or fails."""
    if not PYNVML_AVAILABLE:
        logging.error(
            "pynvml is not available. Install it via `pip install nvidia-ml-py3` to enable GPU monitoring."
        )
        return False
    try:
        nvmlInit()
        device_count = nvmlDeviceGetCount()
        if gpu_index >= device_count:
            logging.error(
                f"GPU index {gpu_index} is out of range (found {device_count} devices). "
                "Check your GPU index or NVIDIA driver installation."
            )
            return False
        nvmlShutdown()
        return True
    except NVMLError as e:
        logging.error(
            f"Failed to initialize NVML: {e}. Check NVIDIA driver installation or permissions."
        )
        return False


def _sample_gpu_utilisation(handle, pid: int) -> int:
    try:
        proc_utils = nvmlDeviceGetProcessUtilization(handle, 1000)
        return next((p.smUtil for p in proc_utils if p.pid == pid), 0)
    except (NVMLError_NotSupported, NVMLError):
        return 0


def _sample_gpu_vram(handle, pid: int) -> float:
    try:
        processes = nvmlDeviceGetComputeRunningProcesses(handle)
        return (
            next((p.usedGpuMemory for p in processes if p.pid == pid), 0) / 1024**2
        )  # MB
    except NVMLError:
        return 0


def monitor_gpu_utilization_by_pid(
    gpu_index: int, pid: int, interval: float, result_container
):
    if not PYNVML_AVAILABLE or not validate_gpu_index(gpu_index):
        result_container.append(
            {
                "gpu_util_mean": 0,
                "gpu_util_max": 0,
                "vram_usage_mean": 0,
                "vram_usage_max": 0,
            }
        )
        return

    try:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(gpu_index)
    except NVMLError as e:
        logging.error(
            f"Failed to access GPU {gpu_index}: {e}. Check NVIDIA driver or GPU availability."
        )
        result_container.append(
            {
                "gpu_util_mean": 0,
                "gpu_util_max": 0,
                "vram_usage_mean": 0,
                "vram_usage_max": 0,
            }
        )
        return

    gpu_utils, vram_usages = [], []

    try:
        while psutil.pid_exists(pid) and psutil.Process(pid).is_running():
            gpu_utils.append(_sample_gpu_utilisation(handle, pid))
            vram_usages.append(_sample_gpu_vram(handle, pid))
            time.sleep(interval)
    except Exception:  # Catch any unexpected errors to ensure cleanup
        pass
    finally:
        try:
            nvmlShutdown()
        except NVMLError:
            logging.warning("Failed to shutdown NVML. This may indicate driver issues.")
        result_container.append(
            {
                "gpu_util_mean": sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0,
                "gpu_util_max": max(gpu_utils, default=0),
                "vram_usage_mean": (
                    sum(vram_usages) / len(vram_usages) if vram_usages else 0
                ),
                "vram_usage_max": max(vram_usages, default=0),
            }
        )


def run_and_monitor_gpu_on_function(
    target: Callable[..., Any],
    args: Tuple = (),
    kwargs: Dict[str, Any] = None,
    gpu_index: int = 0,
    interval: float = 0.1,
) -> Dict[str, float]:
    """Run a target function and monitor GPU utilization on the specified GPU.

    Args:
        target: The function to execute.
        args: Positional arguments for the target function.
        kwargs: Keyword arguments for the target function.
        gpu_index: Index of the GPU to monitor.
        interval: Sampling interval in seconds.

    Returns:
        Dictionary with mean/max GPU utilization and VRAM usage. Returns zeros if monitoring fails.
    """
    if kwargs is None:
        kwargs = {}

    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    result_container = manager.list()

    process = mp.Process(target=target, args=args, kwargs=kwargs)
    process.start()

    monitor_proc = mp.Process(
        target=monitor_gpu_utilization_by_pid,
        args=(gpu_index, process.pid, interval, result_container),
    )
    monitor_proc.start()

    process.join()
    monitor_proc.join()

    return (
        result_container[0]
        if result_container
        else {
            "gpu_util_mean": 0,
            "gpu_util_max": 0,
            "vram_usage_mean": 0,
            "vram_usage_max": 0,
        }
    )


def heavy_gpu_task():
    import os

    import torch

    print("PID:", os.getpid())
    try:
        a = torch.randn(5000, 5000, device="cuda:1")
        for _ in range(1000):
            b = torch.matmul(a, a.T)
    except RuntimeError as e:
        print(f"GPU task failed: {e}")


if __name__ == "__main__":
    result = run_and_monitor_gpu_on_function(
        target=heavy_gpu_task, gpu_index=1, interval=0.01
    )
    print("GPU Monitoring Result:", result)
