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

    def _detect_nvidia_architecture(self, gpu_name: str) -> str:
        """Detect NVIDIA GPU architecture based on GPU name"""
        gpu_name_lower = gpu_name.lower()
        if any(
            x in gpu_name_lower
            for x in ["rtx 4090", "rtx 4080", "rtx 4070", "rtx 4060"]
        ):
            return "Ada Lovelace"
        if any(
            x in gpu_name_lower
            for x in [
                "rtx 3090",
                "rtx 3080",
                "rtx 3070",
                "rtx 3060",
                "a100",
                "a40",
                "a30",
                "a10",
            ]
        ):
            return "Ampere"
        if any(
            x in gpu_name_lower
            for x in ["rtx 2080", "rtx 2070", "rtx 2060", "gtx 1660", "gtx 1650"]
        ):
            return "Turing"
        if any(
            x in gpu_name_lower
            for x in ["gtx 1080", "gtx 1070", "gtx 1060", "gtx 1050", "p100", "p40"]
        ):
            return "Pascal"
        if any(
            x in gpu_name_lower for x in ["gtx 980", "gtx 970", "gtx 960", "gtx 950"]
        ):
            return "Maxwell"
        if any(x in gpu_name_lower for x in ["gtx 780", "gtx 770", "gtx 760"]):
            return "Kepler"
        return "Unknown"

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
                    architecture = self._detect_nvidia_architecture(name)

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
