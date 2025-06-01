import datetime
import logging
import multiprocessing as mp
import statistics
import time
from typing import Any, Callable, Dict, Tuple

import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

try:
    from pynvml import (NVML_ERROR_NOT_SUPPORTED, NVMLError,
                        nvmlDeviceGetComputeRunningProcesses,
                        nvmlDeviceGetCount, nvmlDeviceGetCudaComputeCapability,
                        nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo,
                        nvmlDeviceGetName, nvmlDeviceGetProcessUtilization,
                        nvmlInit, nvmlShutdown, nvmlSystemGetDriverVersion)

    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


class GPUMonitor:
    """Only support NVIDIA GPU"""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.nvidia_initialized = False
        if PYNVML_AVAILABLE:
            try:
                nvmlInit()
                self.nvidia_initialized = True
                self.logger.info("NVIDIA ML initialized successfully")
            except NVMLError as e:
                self.logger.warning(f"Failed to initialize NVIDIA ML: {e}")

    def __del__(self):
        """Cleanup NVIDIA ML on destruction"""
        if self.nvidia_initialized and PYNVML_AVAILABLE:
            try:
                nvmlShutdown()
            except NVMLError:
                pass

    def _detect_nvidia_architecture(self, compute_capability: str) -> str:
        """
        Detect NVIDIA GPU architecture based on compute capability.
        Knowledge till 1st June 2025.
        """
        compute_capability_arch_map = {
            "2.0": "Fermi",
            "2.1": "Fermi",
            "3.0": "Kepler",
            "3.2": "Kepler",
            "3.5": "Kepler",
            "3.7": "Kepler",
            "5.0": "Maxwell",
            "5.2": "Maxwell",
            "5.3": "Maxwell",
            "6.0": "Pascal",
            "6.1": "Pascal",
            "6.2": "Pascal",
            "7.0": "Volta",
            "7.2": "Xavier",
            "7.5": "Turing",
            "8.0": "Ampere",
            "8.6": "Ampere",
            "8.9": "Ada Lovelace",
            "9.0": "Hopper",
            "9.1": "Hopper",
            "10.0": "Blackwell",
            "10.1": "Blackwell",
            "12.0": "Rubin",
        }
        return compute_capability_arch_map.get(compute_capability, "Unknown")

    def get_information(self) -> dict:
        """
        Q1: How many gpus are there ?
        Q2: What are the names of the gpus ?
        Q3: What are their indexes ?
        Q4: What are their total VRAM ?
        Q5: If NVIDIA, what are Driver Version ?
        Q6: If NVIDIA, what compute capability ?
        Q7: If NVIDIA, what architecture ?
        """
        result = {"gpu_count": 0, "gpus": [], "driver_version": None}

        if not PYNVML_AVAILABLE or not self.nvidia_initialized:
            self.logger.warning("NVIDIA monitoring unavailable or not initialized")
            return result

        try:
            # Get driver version
            driver_version = nvmlSystemGetDriverVersion()
            result["driver_version"] = driver_version

            # Get GPU count
            gpu_count = nvmlDeviceGetCount()
            result["gpu_count"] = gpu_count

            # Collect info for each GPU
            for i in range(gpu_count):
                try:
                    handle = nvmlDeviceGetHandleByIndex(i)
                    name = nvmlDeviceGetName(handle)
                    mem_info = nvmlDeviceGetMemoryInfo(handle)
                    total_vram = mem_info.total // (1024 * 1024)  # Convert to MB
                    cc_major, cc_minor = nvmlDeviceGetCudaComputeCapability(handle)
                    compute_capability = f"{cc_major}.{cc_minor}"
                    architecture = self._detect_nvidia_architecture(compute_capability)

                    gpu_info = {
                        "index": i,
                        "name": name,
                        "total_vram_mb": total_vram,
                        "compute_capability": compute_capability,
                        "architecture": architecture,
                    }
                    result["gpus"].append(gpu_info)
                except NVMLError as e:
                    self.logger.error(f"Error getting info for GPU {i}: {e}")
                    result["gpus"].append(
                        {
                            "index": i,
                            "name": "Unknown NVIDIA GPU",
                            "total_vram_mb": None,
                            "compute_capability": None,
                            "architecture": "Unknown",
                        }
                    )

        except NVMLError as e:
            self.logger.error(f"Error detecting NVIDIA GPUs: {e}")

        return result

    @staticmethod
    def sample_gpu_utilisation(handle, pid: int) -> int:
        try:
            proc_utils = nvmlDeviceGetProcessUtilization(handle, 1000)
            return next((p.smUtil for p in proc_utils if p.pid == pid), 0)
        except NVMLError as e:
            return 0  # Return 0 for all NVML errors, including NotFound

    @staticmethod
    def sample_gpu_vram(handle, pid: int) -> int:
        try:
            processes = nvmlDeviceGetComputeRunningProcesses(handle)
            return (
                next((p.usedGpuMemory for p in processes if p.pid == pid), 0) / 1024**2
            )  # MB
        except NVMLError:
            return 0

    def monitor_gpu_utilisation_by_pid(
        self, pid: int, interval: float = 1.0, timeout: float = float("inf")
    ) -> dict:
        """
        Return process's gpu usage.
        Return a dict of GPU Max util, GPU Mean Util, VRAM Max, VRAM mean for each available GPU, start time, duration,
        exit_code (True if Ctrl+C), and timeout (True if timeout reached).
        """
        result = {
            "gpu_max_util": {},
            "gpu_mean_util": {},
            "vram_max_mb": {},
            "vram_mean_mb": {},
            "start_time": None,
            "duration": None,
            "exit_code": False,
            "timeout": False,
        }

        if not PYNVML_AVAILABLE or not self.nvidia_initialized:
            self.logger.warning("NVIDIA monitoring unavailable or not initialized")
            result["start_time"] = time.time()
            result["duration"] = 0.0
            return result

        if not psutil.pid_exists(pid):
            self.logger.error(f"Process with PID {pid} does not exist")
            result["start_time"] = time.time()
            result["duration"] = 0.0
            return result

        try:
            proc = psutil.Process(pid)
            gpu_count = nvmlDeviceGetCount()
            self.logger.info(
                f"Starting GPU monitoring for PID {pid} on {gpu_count} GPU(s) with interval {interval}s and timeout {timeout}s"
            )

            # Initialize storage for samples
            gpu_util_samples = {i: [] for i in range(gpu_count)}
            vram_samples = {i: [] for i in range(gpu_count)}
            start_time = time.time()

            try:
                while True:
                    # Check if process is still running
                    if not proc.is_running() or proc.status() == psutil.STATUS_ZOMBIE:
                        self.logger.info(
                            f"Process with PID {pid} has terminated or is a zombie"
                        )
                        break

                    # Check for timeout
                    elapsed_time = time.time() - start_time
                    if elapsed_time >= timeout:
                        self.logger.info(f"Timeout of {timeout}s reached for PID {pid}")
                        result["timeout"] = True
                        break

                    for i in range(gpu_count):
                        try:
                            handle = nvmlDeviceGetHandleByIndex(i)
                            gpu_util = self.sample_gpu_utilisation(handle, pid)
                            vram_usage = self.sample_gpu_vram(handle, pid)
                            gpu_util_samples[i].append(gpu_util)
                            vram_samples[i].append(vram_usage)
                            # self.logger.info(
                            #     f"PID {pid} on GPU {i}: GPU Util {gpu_util}% | VRAM {vram_usage:.2f} MB"
                            # )
                        except NVMLError as e:
                            self.logger.error(f"Error sampling GPU {i}: {e}")
                            gpu_util_samples[i].append(0)
                            vram_samples[i].append(0.0)

                    time.sleep(interval)

            except KeyboardInterrupt:
                self.logger.info(
                    f"Monitoring stopped for PID {pid} due to user interrupt"
                )
                result["exit_code"] = True

            finally:
                duration = time.time() - start_time
                # Compute max and mean for each GPU
                for i in range(gpu_count):
                    util_samples = gpu_util_samples[i]
                    vram_samples_list = vram_samples[i]
                    result["gpu_max_util"][i] = max(util_samples) if util_samples else 0
                    result["gpu_mean_util"][i] = (
                        statistics.mean(util_samples) if util_samples else 0.0
                    )
                    result["vram_max_mb"][i] = (
                        max(vram_samples_list) if vram_samples_list else 0.0
                    )
                    result["vram_mean_mb"][i] = (
                        statistics.mean(vram_samples_list) if vram_samples_list else 0.0
                    )

                result["start_time"] = start_time
                result["duration"] = duration
                return result

        except NVMLError as e:
            self.logger.error(f"Error accessing GPUs: {e}")
            result["start_time"] = time.time()
            result["duration"] = 0.0
            return result


if __name__ == "__main__":
    monitor = GPUMonitor()
    info = monitor.get_information()
    print(f"info: {info}")

    # Example: Monitor GPU usage for a process (replace 1234 with actual PID)
    result = monitor.monitor_gpu_utilisation_by_pid(10402, 0.01, 10.0)  # 60s timeout
    print(f"Monitoring result: {result}")
