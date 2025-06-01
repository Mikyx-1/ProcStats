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


if __name__ == "__main__":
    monitor = GPUMonitor()
    info = monitor.get_information()
    print(f"info: {info}")
