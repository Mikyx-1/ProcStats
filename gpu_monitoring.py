import multiprocessing as mp
import os
import time
from typing import Union

import numpy as np
import psutil

# GPU monitoring setup
try:
    from pynvml import (NVMLError, NVMLError_NotSupported,
                        nvmlDeviceGetComputeRunningProcesses,
                        nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex,
                        nvmlDeviceGetProcessUtilization, nvmlInit,
                        nvmlShutdown)

    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


class MonitorGPU:
    def __init__(self, gpu_index=0, interval=0.1):
        self.gpu_index = gpu_index
        self.interval = interval

        if not PYNVML_AVAILABLE:
            raise ImportError(
                "pynvml is not available. Install it via `pip install nvidia-ml-py3`."
            )

        nvmlInit()
        device_count = nvmlDeviceGetCount()
        if gpu_index >= device_count:
            nvmlShutdown()
            raise ValueError(
                f"GPU index {gpu_index} is out of range (found {device_count} devices)."
            )

        nvmlShutdown()  # Re-initialise inside subprocess later

    @staticmethod
    def monitor_until_static(gpu_index, pid, interval, result_container):
        import psutil
        from pynvml import (NVMLError, NVMLError_NotSupported,
                            nvmlDeviceGetComputeRunningProcesses,
                            nvmlDeviceGetHandleByIndex,
                            nvmlDeviceGetProcessUtilization, nvmlInit,
                            nvmlShutdown)

        def sample_gpu_utilisation(handle, pid):
            try:
                proc_utils = nvmlDeviceGetProcessUtilization(handle, 1000)
                for p in proc_utils:
                    if p.pid == pid:
                        return p.smUtil
            except NVMLError_NotSupported:
                print("Per-process GPU utilisation not supported.")
            except NVMLError:
                pass
            return 0

        def sample_gpu_vram(handle, pid):
            try:
                processes = nvmlDeviceGetComputeRunningProcesses(handle)
                for p in processes:
                    if p.pid == pid:
                        return p.usedGpuMemory / 1024**2  # MB
            except NVMLError:
                pass
            return 0

        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(gpu_index)

        gpu_utils = []
        vram_usages = []

        try:
            while psutil.pid_exists(pid) and psutil.Process(pid).is_running():
                gpu_utils.append(sample_gpu_utilisation(handle, pid))
                vram_usages.append(sample_gpu_vram(handle, pid))
                time.sleep(interval)
        finally:
            nvmlShutdown()
            result_container.append(
                {
                    "gpu_util_mean": (
                        sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0
                    ),
                    "gpu_util_max": max(gpu_utils, default=0),
                    "vram_usage_mean": (
                        sum(vram_usages) / len(vram_usages) if vram_usages else 0
                    ),
                    "vram_usage_max": max(vram_usages, default=0),
                }
            )

    def monitor(self, target, args=(), kwargs=None):
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
            target=MonitorGPU.monitor_until_static,
            args=(self.gpu_index, process.pid, self.interval, result_container),
        )
        monitor_proc.start()

        # Wait for both
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


def heavy_cpu_gpu_task():
    import os

    import torch

    print("Inside PID:", os.getpid())
    a = torch.randn(5000, 5000, device="cuda:1")
    for _ in range(1000):
        b = torch.matmul(a, a.T)


if __name__ == "__main__":
    monitor_gpu = MonitorGPU(gpu_index=1, interval=0.0001)
    res = monitor_gpu.monitor(heavy_cpu_gpu_task)
    print(f"res: {res}")
