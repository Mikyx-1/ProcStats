import os
import time

try:
    from pynvml import (
        nvmlInit,
        nvmlShutdown,
        nvmlDeviceGetCount,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetComputeRunningProcesses,
        nvmlDeviceGetUtilizationRates,
        nvmlDeviceGetProcessUtilization,
        nvmlDeviceGetMemoryInfo,
        NVMLError,
        NVMLError_NotSupported,
    )

    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


def monitor_gpu_usage(gpu_index=0, pid=None):
    if not PYNVML_AVAILABLE:
        print("pynvml is not installed. GPU monitoring not available.")
        return None

    try:
        nvmlInit()
        device_count = nvmlDeviceGetCount()
        if gpu_index >= device_count:
            print(f"Invalid GPU index: {gpu_index}. Only {device_count} GPUs found.")
            return None

        handle = nvmlDeviceGetHandleByIndex(gpu_index)

        if pid is None:
            pid = os.getpid()

        # Get process-specific VRAM usage
        vram_usage = 0
        try:
            processes = nvmlDeviceGetComputeRunningProcesses(handle)
            for p in processes:
                if p.pid == pid:
                    vram_usage = p.usedGpuMemory / 1024**2  # in MB
                    break
        except NVMLError:
            pass

        # Get per-process GPU utilization
        gpu_util = 0
        try:
            # Wait to allow measurement interval (NVML requires a delay)
            time.sleep(0.5)
            proc_utils = nvmlDeviceGetProcessUtilization(handle, 1000)  # 1000ms window
            for p in proc_utils:
                if p.pid == pid:
                    gpu_util = p.smUtil  # in percent
                    break
        except NVMLError_NotSupported:
            print("Per-process GPU utilization not supported on this driver.")
        except NVMLError:
            pass

        nvmlShutdown()
        return {
            "gpu_utilization_percent": gpu_util,
            "vram_usage_mb": vram_usage,
        }

    except Exception as e:
        print(f"GPU monitoring failed: {e}")
        return None


result = monitor_gpu_usage(gpu_index=0)

if result:
    print(f"GPU Utilization: {result['gpu_utilization_percent']}%")
    print(f"VRAM Usage (PID {os.getpid()}): {result['vram_usage_mb']:.2f} MB")
