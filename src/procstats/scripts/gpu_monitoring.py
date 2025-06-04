import datetime
import logging
import multiprocessing as mp
import statistics
import time
from typing import Any, Callable, Dict, List, Set, Tuple

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

    def _get_all_child_pids(self, parent_pid: int) -> Set[int]:
        """Recursively get all child PIDs of a parent process"""
        all_pids = set()
        try:
            parent = psutil.Process(parent_pid)
            all_pids.add(parent_pid)

            # Get all descendants recursively
            for child in parent.children(recursive=True):
                try:
                    all_pids.add(child.pid)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        return all_pids

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
        self,
        pid: int,
        interval: float = 1.0,
        timeout: float = float("inf"),
        include_children: bool = True,
    ) -> dict:
        """
        Return process's gpu usage, including child processes if include_children=True.
        Return a dict of GPU Max util, GPU Mean Util, VRAM Max, VRAM mean for each available GPU, start time, duration,
        exit_code (True if Ctrl+C), timeout (True if timeout reached), and process_info with per-PID details.
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
            "process_info": {},  # Per-PID breakdown
            "monitored_pids": set(),
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
            gpu_count = nvmlDeviceGetCount()
            self.logger.info(
                f"Starting GPU monitoring for PID {pid} (include_children={include_children}) on {gpu_count} GPU(s) with interval {interval}s and timeout {timeout}s"
            )

            # Initialize storage for samples - aggregated across all PIDs
            gpu_util_samples = {i: [] for i in range(gpu_count)}
            vram_samples = {i: [] for i in range(gpu_count)}

            # Per-PID tracking
            per_pid_samples = {}

            start_time = time.time()
            monitored_pids = set()

            try:
                while True:
                    # Get current PIDs to monitor
                    if include_children:
                        current_pids = self._get_all_child_pids(pid)
                    else:
                        current_pids = {pid} if psutil.pid_exists(pid) else set()

                    # Check if any processes are still running
                    if not current_pids:
                        self.logger.info(
                            f"No processes to monitor (original PID: {pid})"
                        )
                        break

                    # Update monitored PIDs set
                    monitored_pids.update(current_pids)

                    # Check for timeout
                    elapsed_time = time.time() - start_time
                    if elapsed_time >= timeout:
                        self.logger.info(f"Timeout of {timeout}s reached for PID {pid}")
                        result["timeout"] = True
                        break

                    # Sample GPU usage for each GPU and each PID
                    current_gpu_util = {i: 0 for i in range(gpu_count)}
                    current_vram_usage = {i: 0.0 for i in range(gpu_count)}

                    for current_pid in current_pids:
                        # Initialize per-PID tracking if needed
                        if current_pid not in per_pid_samples:
                            per_pid_samples[current_pid] = {
                                "gpu_util": {i: [] for i in range(gpu_count)},
                                "vram": {i: [] for i in range(gpu_count)},
                            }

                        for i in range(gpu_count):
                            try:
                                handle = nvmlDeviceGetHandleByIndex(i)
                                gpu_util = self.sample_gpu_utilisation(
                                    handle, current_pid
                                )
                                vram_usage = self.sample_gpu_vram(handle, current_pid)

                                # Store per-PID samples
                                per_pid_samples[current_pid]["gpu_util"][i].append(
                                    gpu_util
                                )
                                per_pid_samples[current_pid]["vram"][i].append(
                                    vram_usage
                                )

                                # Aggregate for overall stats
                                current_gpu_util[i] += gpu_util
                                current_vram_usage[i] += vram_usage

                            except NVMLError as e:
                                self.logger.error(
                                    f"Error sampling GPU {i} for PID {current_pid}: {e}"
                                )
                                per_pid_samples[current_pid]["gpu_util"][i].append(0)
                                per_pid_samples[current_pid]["vram"][i].append(0.0)

                    # Store aggregated samples
                    for i in range(gpu_count):
                        gpu_util_samples[i].append(current_gpu_util[i])
                        vram_samples[i].append(current_vram_usage[i])

                    # Log current status
                    active_pids = len(current_pids)
                    if active_pids > 0:
                        self.logger.debug(
                            f"Monitoring {active_pids} processes: {sorted(current_pids)}"
                        )

                    time.sleep(interval)

            except KeyboardInterrupt:
                self.logger.info(
                    f"Monitoring stopped for PID {pid} due to user interrupt"
                )
                result["exit_code"] = True

            finally:
                duration = time.time() - start_time

                # Compute aggregated max and mean for each GPU
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

                # Compute per-PID statistics
                for pid_key, samples in per_pid_samples.items():
                    result["process_info"][pid_key] = {
                        "gpu_max_util": {},
                        "gpu_mean_util": {},
                        "vram_max_mb": {},
                        "vram_mean_mb": {},
                    }

                    for i in range(gpu_count):
                        util_samples = samples["gpu_util"][i]
                        vram_samples_list = samples["vram"][i]

                        result["process_info"][pid_key]["gpu_max_util"][i] = (
                            max(util_samples) if util_samples else 0
                        )
                        result["process_info"][pid_key]["gpu_mean_util"][i] = (
                            statistics.mean(util_samples) if util_samples else 0.0
                        )
                        result["process_info"][pid_key]["vram_max_mb"][i] = (
                            max(vram_samples_list) if vram_samples_list else 0.0
                        )
                        result["process_info"][pid_key]["vram_mean_mb"][i] = (
                            statistics.mean(vram_samples_list)
                            if vram_samples_list
                            else 0.0
                        )

                result["start_time"] = start_time
                result["duration"] = duration
                result["monitored_pids"] = monitored_pids

                self.logger.info(
                    f"Monitoring completed. Tracked {len(monitored_pids)} PIDs: {sorted(monitored_pids)}"
                )
                return result

        except NVMLError as e:
            self.logger.error(f"Error accessing GPUs: {e}")
            result["start_time"] = time.time()
            result["duration"] = 0.0
            return result

    def monitor_function_with_multiprocessing(
        self,
        target_function: Callable,
        interval: float = 1.0,
        timeout: float = float("inf"),
    ) -> dict:
        """
        Monitor a function that may spawn child processes.
        This is a convenience method that runs the function and monitors all related processes.
        """
        import os

        def wrapper():
            # Get the current PID
            current_pid = os.getpid()
            self.logger.info(f"Starting function monitoring for PID {current_pid}")

            # Start monitoring in a separate process
            monitor_process = mp.Process(
                target=self._monitor_and_store_results,
                args=(current_pid, interval, timeout),
            )
            monitor_process.start()

            try:
                # Run the target function
                target_function()
            finally:
                # Stop monitoring
                monitor_process.terminate()
                monitor_process.join(timeout=5)
                if monitor_process.is_alive():
                    monitor_process.kill()

        # This is a simplified version - for full implementation,
        # you'd need inter-process communication to get results back
        self.logger.warning(
            "monitor_function_with_multiprocessing is a simplified implementation"
        )
        wrapper()
        return {
            "status": "completed",
            "note": "Use monitor_gpu_utilisation_by_pid directly for full results",
        }

    def _monitor_and_store_results(self, pid: int, interval: float, timeout: float):
        """Helper method for monitoring in separate process"""
        result = self.monitor_gpu_utilisation_by_pid(
            pid, interval, timeout, include_children=True
        )
        # In a full implementation, you'd store this result somewhere accessible
        # to the parent process, like a shared memory object or file
        self.logger.info(f"Monitoring result: {result}")


if __name__ == "__main__":
    monitor = GPUMonitor()
    info = monitor.get_information()
    print(f"info: {info}")

    # Example: Monitor GPU usage for a process with children
    result = monitor.monitor_gpu_utilisation_by_pid(
        24383, 0.01, 10.0, include_children=True
    )
    print(result)
