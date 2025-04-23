import os
import time
import psutil
import functools
import multiprocessing as mp
import numpy as np

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


def monitor_gpu_usage(gpu_index=0, pid=None):
    if not PYNVML_AVAILABLE:
        return {"gpu_utilisation_percent": None, "vram_usage_mb": None}

    try:
        nvmlInit()
        device_count = nvmlDeviceGetCount()
        if gpu_index >= device_count:
            nvmlShutdown()
            return {"gpu_utilisation_percent": None, "vram_usage_mb": None}

        handle = nvmlDeviceGetHandleByIndex(gpu_index)
        if pid is None:
            pid = os.getpid()

        vram_usage = 0
        try:
            processes = nvmlDeviceGetComputeRunningProcesses(handle)
            for p in processes:
                if p.pid == pid:
                    vram_usage = p.usedGpuMemory / 1024**2
                    break
        except NVMLError:
            pass

        gpu_util = 0
        try:
            time.sleep(0.5)
            proc_utils = nvmlDeviceGetProcessUtilization(handle, 1000)
            for p in proc_utils:
                if p.pid == pid:
                    gpu_util = p.smUtil
                    break
        except NVMLError_NotSupported:
            print("Per-process GPU utilisation not supported.")
        except NVMLError:
            pass

        nvmlShutdown()
        return {
            "gpu_utilisation_percent": gpu_util,
            "vram_usage_mb": vram_usage,
        }

    except Exception as e:
        print(f"GPU monitoring failed: {e}")
        return {"gpu_utilisation_percent": None, "vram_usage_mb": None}


class Benchmark:
    def __init__(self, interval=0.1, use_gpu=False, gpu_index=0):
        self.interval = interval
        self.use_gpu = use_gpu
        self.gpu_index = gpu_index

    def _monitor_process(self, pid, stats, stop_event):
        proc = psutil.Process(pid)
        cpu_usages = []
        ram_usages = []
        gpu_utils = []
        vram_usages = []
        proc.cpu_percent(interval=None)

        while not stop_event.is_set():
            if proc.is_running():
                try:
                    cpu_usages.append(proc.cpu_percent(interval=None))
                    ram_usages.append(proc.memory_info().rss)
                    if self.use_gpu:
                        gpu_resource_usg = monitor_gpu_usage(
                            gpu_index=self.gpu_index, pid=pid
                        )
                        gpu_utils.append(gpu_resource_usg["gpu_utilisation_percent"])
                        vram_usages.append(gpu_resource_usg["vram_usage_mb"])
                except psutil.NoSuchProcess:
                    break
            time.sleep(self.interval)

        stats["cpu_max"] = max(cpu_usages, default=0)
        stats["cpu_avg"] = sum(cpu_usages) / len(cpu_usages) if cpu_usages else 0
        stats["ram_max"] = max(ram_usages, default=0) / 1024**2
        stats["ram_avg"] = (
            sum(ram_usages) / len(ram_usages) / 1024**2 if ram_usages else 0
        )
        stats["gpu_utilisation_avg"] = (
            sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0
        )
        stats["gpu_utilisation_max"] = max(gpu_utils, default=0)
        stats["vram_usage_avg"] = (
            sum(vram_usages) / len(vram_usages) if vram_usages else 0
        )
        stats["vram_usage_max"] = max(vram_usages, default=0)

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def target_func(return_dict, stats_dict, *a, **kw):
                result = func(*a, **kw)
                return_dict["result"] = result

                if self.use_gpu:
                    gpu_stats = monitor_gpu_usage(gpu_index=self.gpu_index)
                    stats_dict["gpu_utilisation_percent"] = gpu_stats[
                        "gpu_utilisation_percent"
                    ]
                    stats_dict["vram_usage_mb"] = gpu_stats["vram_usage_mb"]

            manager = mp.Manager()
            return_dict = manager.dict()
            stats = manager.dict()
            stop_event = manager.Event()

            proc = mp.Process(
                target=target_func, args=(return_dict, stats, *args), kwargs=kwargs
            )
            start_time = time.time()
            proc.start()

            monitor = mp.Process(
                target=self._monitor_process, args=(proc.pid, stats, stop_event)
            )
            monitor.start()

            proc.join()
            stop_event.set()
            monitor.join()
            end_time = time.time()

            duration = end_time - start_time

            print(f"[{func.__name__}] Execution time: {duration:.2f} seconds")
            print(f"[{func.__name__}] CPU max usage: {stats.get('cpu_max', 0):.2f}%")
            print(f"[{func.__name__}] CPU avg usage: {stats.get('cpu_avg', 0):.2f}%")
            print(f"[{func.__name__}] RAM max usage: {stats.get('ram_max', 0):.2f} MB")
            print(f"[{func.__name__}] RAM avg usage: {stats.get('ram_avg', 0):.2f} MB")

            if self.use_gpu:
                print(
                    f"[{func.__name__}] GPU utilisation: {stats.get('gpu_utilisation_percent', 0)}%"
                )
                print(
                    f"[{func.__name__}] VRAM usage: {stats.get('vram_usage_mb', 0):.2f} MB"
                )

            return return_dict.get("result", None)

        return wrapper


# Example usage:
@Benchmark(interval=0.1, use_gpu=True, gpu_index=1)
def heavy_gpu_task():
    import torch

    print("Inside task PID:", os.getpid())
    a = torch.randn(10000, 10000, device="cuda:1")
    for _ in range(10):
        b = torch.matmul(a, a.T)
        time.sleep(1)
    # time.sleep(5)


heavy_gpu_task()
