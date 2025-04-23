import os
import time
import psutil
import functools
import multiprocessing as mp
import numpy as np
from typing import Union

# GPU monitoring setup
try:
    from pynvml import (
        nvmlInit,
        nvmlShutdown,
        nvmlDeviceGetCount,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetComputeRunningProcesses,
        nvmlDeviceGetProcessUtilization,
        NVMLError,
        NVMLError_NotSupported,
    )

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

        self.handle = nvmlDeviceGetHandleByIndex(gpu_index)

    def _sample_gpu_utilisation(self, pid):
        try:
            proc_utils = nvmlDeviceGetProcessUtilization(self.handle, 1000)
            for p in proc_utils:
                if p.pid == pid:
                    return p.smUtil
        except NVMLError_NotSupported:
            print("Per-process GPU utilisation not supported.")
        except NVMLError:
            pass
        return 0

    def _sample_gpu_vram(self, pid):
        try:
            processes = nvmlDeviceGetComputeRunningProcesses(self.handle)
            for p in processes:
                if p.pid == pid:
                    return p.usedGpuMemory / 1024**2  # MB
        except NVMLError:
            pass
        return 0

    def _monitor(self, pid):
        gpu_utils = []
        vram_usages = []

        try:
            proc = psutil.Process(pid)
            while proc.is_running():
                gpu_utils.append(self._sample_gpu_utilisation(pid))
                vram_usages.append(self._sample_gpu_vram(pid))
                time.sleep(self.interval)
        except psutil.NoSuchProcess:
            pass
        finally:
            nvmlShutdown()
            gpu_util_mean = sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0
            gpu_util_max = max(gpu_utils, default=0)

            vram_usage_mean = sum(vram_usages) / len(vram_usages) if vram_usages else 0
            vram_usage_max = max(vram_usages, default=0)

            return {
                "gpu_util_mean": gpu_util_mean,
                "gpu_util_max": gpu_util_max,
                "vram_usage_mean": vram_usage_mean,
                "vram_usage_max": vram_usage_max,
            }


if __name__ == "__main__":
    gpu_resource_sampler = MonitorGPU(gpu_index=1, interval=0.001)
    res = gpu_resource_sampler._monitor(pid=221628)
    print(res)
