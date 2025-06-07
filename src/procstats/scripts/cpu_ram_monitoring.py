import multiprocessing as mp
import statistics
import time
from typing import Any, Callable, Dict, List, Tuple

import psutil

from ..tests.test_burn_cpu_ram import burn_cpu_accurate


def get_cpu_cores():
    """Get the number of CPU cores."""
    return psutil.cpu_count() or 1


class AdaptiveMonitor:
    def __init__(self, pid: int, base_interval: float = 0.05):
        self.pid = pid
        self.base_interval = base_interval
        self.min_interval = 0.01
        self.max_interval = 0.5
        self.stability_threshold = 5.0  # CPU % threshold for stability
        self.window_size = 10  # Moving window for stability analysis
        self.cpu_history = []
        self.ram_history = []
        self.interval_history = []

    def calculate_stability_score(self, values: List[float]) -> float:
        """Calculate stability score based on coefficient of variation."""
        if len(values) < 3:
            return float("inf")  # Not enough data, assume unstable

        mean_val = statistics.mean(values)
        if mean_val == 0:
            return 0.0

        std_val = statistics.stdev(values)
        return (std_val / mean_val) * 100  # Coefficient of variation as percentage

    def adaptive_interval(self, recent_cpu_values: List[float]) -> float:
        """Dynamically adjust sampling interval based on CPU stability."""
        if len(recent_cpu_values) < 3:
            return self.base_interval

        stability_score = self.calculate_stability_score(recent_cpu_values)

        # If CPU usage is highly variable, use shorter intervals
        # If CPU usage is stable, use longer intervals
        if stability_score > 20:  # High variability
            return max(self.min_interval, self.base_interval * 0.5)
        elif stability_score < 5:  # Low variability
            return min(self.max_interval, self.base_interval * 2)
        else:  # Medium variability
            return self.base_interval

    def outlier_filter(
        self, values: List[float], z_threshold: float = 2.0
    ) -> List[float]:
        """Remove outliers using modified Z-score method."""
        if len(values) < 3:
            return values

        median = statistics.median(values)
        mad = statistics.median([abs(x - median) for x in values])

        if mad == 0:
            return values

        # Modified Z-scores
        modified_z_scores = [0.6745 * (x - median) / mad for x in values]

        # Filter outliers
        filtered = [
            values[i] for i, z in enumerate(modified_z_scores) if abs(z) <= z_threshold
        ]

        return filtered if filtered else values  # Return original if all filtered


def monitor_cpu_and_ram_by_pid_advanced(
    pid: int, base_interval: float, result_container: list
):
    """Advanced monitoring with adaptive sampling and noise reduction."""
    monitor = AdaptiveMonitor(pid, base_interval)
    cpu_measurements = []
    ram_measurements = []
    start_time = time.time()
    timeout = 12.0

    # Separate tracking for raw vs processed data
    raw_cpu_data = []
    raw_ram_data = []

    try:
        parent_proc = psutil.Process(pid)

        # Initial CPU percent call to prime the system
        try:
            parent_proc.cpu_percent()
            for child in parent_proc.children(recursive=True):
                try:
                    child.cpu_percent()
                except (
                    psutil.NoSuchProcess,
                    psutil.ZombieProcess,
                    psutil.AccessDenied,
                ):
                    continue
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            pass

        # Brief stabilization period
        time.sleep(0.1)

        measurement_count = 0
        consecutive_stable_readings = 0

        while time.time() - start_time < timeout:
            try:
                if (
                    not parent_proc.is_running()
                    or parent_proc.status() == psutil.STATUS_ZOMBIE
                ):
                    print(f"[Monitor] Parent process {pid} terminated")
                    break

                # Get current interval based on recent CPU stability
                recent_window = (
                    raw_cpu_data[-monitor.window_size :]
                    if len(raw_cpu_data) >= monitor.window_size
                    else raw_cpu_data
                )
                current_interval = monitor.adaptive_interval(recent_window)

                # Collect measurements
                all_procs = [parent_proc] + parent_proc.children(recursive=True)
                total_cpu_percent = 0.0
                total_ram_usage = 0.0
                active_processes = 0

                for proc in all_procs:
                    try:
                        if (
                            not proc.is_running()
                            or proc.status() == psutil.STATUS_ZOMBIE
                        ):
                            continue

                        # Use the adaptive interval for CPU measurement
                        cpu_percent = proc.cpu_percent(interval=current_interval)
                        ram_usage = proc.memory_info().rss / 1024**2  # MB

                        total_cpu_percent += cpu_percent
                        total_ram_usage += ram_usage
                        active_processes += 1

                    except (
                        psutil.NoSuchProcess,
                        psutil.ZombieProcess,
                        psutil.AccessDenied,
                    ):
                        continue

                if active_processes > 0:  # Only record if we have valid data
                    raw_cpu_data.append(total_cpu_percent)
                    raw_ram_data.append(total_ram_usage)
                    measurement_count += 1

                    # Check for stability (for potential early termination of very stable loads)
                    if len(raw_cpu_data) >= 5:
                        recent_5 = raw_cpu_data[-5:]
                        if monitor.calculate_stability_score(recent_5) < 3.0:
                            consecutive_stable_readings += 1
                        else:
                            consecutive_stable_readings = 0

                # Small sleep to prevent overwhelming the system
                time.sleep(max(0.005, current_interval * 0.1))

            except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied):
                print(f"[Monitor] Process {pid} no longer accessible")
                break
            except Exception as e:
                print(f"[Monitor] Error: {e}")
                time.sleep(0.01)
                continue

        # Post-processing: Apply noise reduction techniques
        if raw_cpu_data and raw_ram_data:
            # Remove outliers
            filtered_cpu = monitor.outlier_filter(raw_cpu_data)
            filtered_ram = monitor.outlier_filter(raw_ram_data)

            # Apply moving average smoothing for final results
            def moving_average(data: List[float], window: int = 3) -> List[float]:
                if len(data) < window:
                    return data
                smoothed = []
                for i in range(len(data)):
                    start_idx = max(0, i - window // 2)
                    end_idx = min(len(data), i + window // 2 + 1)
                    smoothed.append(
                        sum(data[start_idx:end_idx]) / (end_idx - start_idx)
                    )
                return smoothed

            # Final smoothed data
            final_cpu_data = moving_average(filtered_cpu, window=3)
            final_ram_data = moving_average(filtered_ram, window=3)

            # Calculate statistics from both raw and processed data
            cpu_max_raw = max(raw_cpu_data)
            cpu_max_processed = max(final_cpu_data)

            # Use processed data for averages, but keep raw max if significantly higher
            # (to not miss genuine spikes)
            cpu_max_final = (
                cpu_max_raw
                if cpu_max_raw > cpu_max_processed * 1.2
                else cpu_max_processed
            )

            result_data = {
                "cpu_max": cpu_max_final,
                "cpu_avg": statistics.mean(final_cpu_data),
                "cpu_p95": (
                    statistics.quantiles(final_cpu_data, n=20)[18]
                    if len(final_cpu_data) > 10
                    else cpu_max_final
                ),  # 95th percentile
                "ram_max": max(final_ram_data),
                "ram_avg": statistics.mean(final_ram_data),
                "ram_p95": (
                    statistics.quantiles(final_ram_data, n=20)[18]
                    if len(final_ram_data) > 10
                    else max(final_ram_data)
                ),
                "num_cores": get_cpu_cores(),
                "measurements_taken": measurement_count,
                "data_quality_score": 100
                - min(
                    50, monitor.calculate_stability_score(final_cpu_data)
                ),  # 0-100 scale
            }
        else:
            result_data = {
                "cpu_max": 0,
                "cpu_avg": 0,
                "cpu_p95": 0,
                "ram_max": 0,
                "ram_avg": 0,
                "ram_p95": 0,
                "num_cores": get_cpu_cores(),
                "measurements_taken": 0,
                "data_quality_score": 0,
            }

    except Exception as e:
        print(f"[Monitor] Fatal error: {e}")
        result_data = {
            "cpu_max": 0,
            "cpu_avg": 0,
            "cpu_p95": 0,
            "ram_max": 0,
            "ram_avg": 0,
            "ram_p95": 0,
            "num_cores": get_cpu_cores(),
            "measurements_taken": 0,
            "data_quality_score": 0,
        }

    finally:
        result_container.append(result_data)


def monitor_cpu_and_ram_on_function_advanced(
    target: Callable[..., Any],
    args: Tuple = (),
    kwargs: Dict[str, Any] = None,
    base_interval: float = 0.05,
) -> Dict[str, float]:
    """
    Enhanced monitoring function with adaptive sampling and noise reduction.

    Args:
        target: The function to execute and monitor.
        args: Positional arguments for the target function.
        kwargs: Keyword arguments for the target function.
        base_interval: Base sampling interval (will be adapted during monitoring).

    Returns:
        Dictionary with comprehensive resource usage statistics.
    """
    if kwargs is None:
        kwargs = {}

    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    result_container = manager.list()

    # Launch target process
    process = mp.Process(target=target, args=args, kwargs=kwargs)
    process.start()

    # Launch advanced monitor process
    monitor_proc = mp.Process(
        target=monitor_cpu_and_ram_by_pid_advanced,
        args=(process.pid, base_interval, result_container),
    )
    monitor_proc.start()

    # Wait for both processes to complete
    process.join()
    monitor_proc.join()

    return (
        result_container[0]
        if result_container
        else {
            "cpu_max": 0,
            "cpu_avg": 0,
            "cpu_p95": 0,
            "ram_max": 0,
            "ram_avg": 0,
            "ram_p95": 0,
            "num_cores": 1,
            "measurements_taken": 0,
            "data_quality_score": 0,
        }
    )


if __name__ == "__main__":
    print("Running advanced resource monitoring...")
    print("=" * 50)

    # Test with the original function
    resource_usage = monitor_cpu_and_ram_on_function_advanced(
        burn_cpu_accurate, base_interval=0.05
    )

    print("\nAdvanced Resource Usage Summary:")
    print("=" * 40)
    print(f"CPU Max:     {resource_usage['cpu_max']:.2f}%")
    print(f"CPU Average: {resource_usage['cpu_avg']:.2f}%")
    print(f"CPU 95th %:  {resource_usage['cpu_p95']:.2f}%")
    print(f"RAM Max:     {resource_usage['ram_max']:.2f} MB")
    print(f"RAM Average: {resource_usage['ram_avg']:.2f} MB")
    print(f"RAM 95th %:  {resource_usage['ram_p95']:.2f} MB")
    print(f"CPU Cores:   {resource_usage['num_cores']}")
    print(f"Measurements: {resource_usage['measurements_taken']}")
    print(f"Data Quality: {resource_usage['data_quality_score']:.1f}/100")

    # Also run the original for comparison
    print("\n" + "=" * 50)
    print("Original Monitor (for comparison):")
