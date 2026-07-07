import multiprocessing as mp
import statistics
import time
from typing import Any, Callable, Dict, List, Tuple

import psutil

from .cpu_rate_estimator import RobustCpuRateEstimator
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
    pid: int, base_interval: float, result_container: list, warmup_time: float = 1.0
):
    """
    Advanced monitoring with adaptive sampling and noise reduction.

    Args:
        warmup_time: seconds to distrust a process's CPU reading after
            *that process* starts (each process in the monitored tree gets
            its own warmup window, measured from its own create_time -- a
            child spawned mid-run gets the same grace period a process
            running since the start got).

            Native libraries that spin up their own thread pool on import
            or first use -- numpy/OpenBLAS, PyTorch, TensorFlow, OpenCV,
            MKL -- do so with real OS threads outside Python's own
            threading module, so a process can genuinely burn many
            CPU-seconds within a tiny wall-clock window right after it
            starts, even though the workload you're trying to measure
            hasn't done anything yet. That's real CPU time, not a
            measurement artifact, so it can't be filtered out
            statistically -- it has to be excluded by knowing the process
            is still too young to trust.

            The default (1.0s) comfortably covers what was observed
            empirically for a numpy import (spikes appeared 0.6-0.75s
            into a process's life, settling within another 1-2 samples).
            Increase it for workloads with heavier native-thread-pool
            imports (torch, tensorflow); decrease it for lightweight,
            pure-Python workloads to start trusting readings sooner. Set
            to 0 to disable and trust every sample immediately.
    """
    monitor = AdaptiveMonitor(pid, base_interval)
    cpu_measurements = []
    ram_measurements = []
    start_time = time.time()
    timeout = 12.0

    # Separate tracking for raw vs processed data
    raw_cpu_data = []
    raw_ram_data = []
    warmup_excluded_samples = 0

    try:
        parent_proc = psutil.Process(pid)

        # Brief stabilization period, giving child processes time to spawn
        time.sleep(0.1)

        measurement_count = 0
        consecutive_stable_readings = 0
        num_cores = get_cpu_cores()
        cpu_estimators: Dict[int, RobustCpuRateEstimator] = {}

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

                # Sleep once for current_interval, then read cumulative CPU
                # time for the whole tree. Each process's own
                # RobustCpuRateEstimator remembers its previous reading and
                # derives the delta/elapsed itself (clamped to what's
                # physically possible and corroborated against a longer
                # sliding window), so no separate before-snapshot pass is
                # needed and single-sample OS accounting artifacts can't
                # leak into the reported rate.
                time.sleep(current_interval)
                now = time.time()

                procs = [parent_proc] + parent_proc.children(recursive=True)
                total_cpu_percent = 0.0
                total_ram_usage = 0.0
                active_processes = 0
                cpu_active_processes = 0

                for proc in procs:
                    try:
                        if (
                            not proc.is_running()
                            or proc.status() == psutil.STATUS_ZOMBIE
                        ):
                            continue

                        times = proc.cpu_times()
                        cpu_time = times.user + times.system
                        ram_usage = proc.memory_info().rss / 1024**2  # MB

                        try:
                            proc_age = now - proc.create_time()
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            proc_age = current_interval

                        estimator = cpu_estimators.get(proc.pid)
                        if estimator is None:
                            estimator = RobustCpuRateEstimator(num_cores=num_cores)
                            cpu_estimators[proc.pid] = estimator

                        sample = estimator.update(now, cpu_time)
                        if sample is None:
                            # First sample for this pid: no prior reading to
                            # diff against yet. cpu_time is this process's
                            # *entire lifetime* CPU usage, so it must be
                            # divided by its actual age (now - create_time),
                            # not this iteration's interval -- dividing a
                            # lifetime total by a tiny window is exactly the
                            # bug we're fixing.
                            cpu_percent = min(
                                100.0 * cpu_time / max(proc_age, 1e-6),
                                100.0 * num_cores,
                            )
                        else:
                            cpu_percent = sample["trusted_rate"]

                        total_ram_usage += ram_usage
                        active_processes += 1

                        if proc_age < warmup_time:
                            # Still inside this process's own warmup window.
                            # A native library (numpy/OpenBLAS, torch, ...)
                            # spinning up its thread pool on import/first use
                            # can genuinely burn many CPU-seconds within a
                            # fraction of a second here -- that's real CPU
                            # time, not a measurement artifact, so no amount
                            # of statistical filtering can catch it. Exclude
                            # it instead of letting it masquerade as the
                            # monitored workload's own usage.
                            warmup_excluded_samples += 1
                            continue

                        total_cpu_percent += max(0.0, cpu_percent)
                        cpu_active_processes += 1

                    except (
                        psutil.NoSuchProcess,
                        psutil.ZombieProcess,
                        psutil.AccessDenied,
                    ):
                        continue

                # Hard system-wide backstop: the whole tree can never
                # legitimately exceed num_cores x 100%, even if every
                # individual process's own reading is (independently)
                # within its own per-process ceiling.
                total_cpu_percent = min(total_cpu_percent, num_cores * 100.0)

                if active_processes > 0:  # Only record if we have valid data
                    raw_ram_data.append(total_ram_usage)

                if cpu_active_processes > 0:  # At least one process past warmup
                    raw_cpu_data.append(total_cpu_percent)
                    measurement_count += 1

                    # Check for stability (for potential early termination of very stable loads)
                    if len(raw_cpu_data) >= 5:
                        recent_5 = raw_cpu_data[-5:]
                        if monitor.calculate_stability_score(recent_5) < 3.0:
                            consecutive_stable_readings += 1
                        else:
                            consecutive_stable_readings = 0

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

            # raw_cpu_data has already had each process's own warmup window
            # excluded (see warmup_time above) and is spike-filtered per
            # process by RobustCpuRateEstimator (physical clamp +
            # two-timescale corroboration), so the true max no longer needs
            # a raw-vs-smoothed override to avoid losing genuine spikes.
            cpu_max_final = max(raw_cpu_data)

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
                "warmup_time": warmup_time,
                "warmup_excluded_samples": warmup_excluded_samples,
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
                "warmup_time": warmup_time,
                "warmup_excluded_samples": warmup_excluded_samples,
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
            "warmup_time": warmup_time,
            "warmup_excluded_samples": warmup_excluded_samples,
            "data_quality_score": 0,
        }

    finally:
        result_container.append(result_data)


def monitor_cpu_and_ram_on_function_advanced(
    target: Callable[..., Any],
    args: Tuple = (),
    kwargs: Dict[str, Any] = None,
    base_interval: float = 0.05,
    warmup_time: float = 1.0,
) -> Dict[str, float]:
    """
    Enhanced monitoring function with adaptive sampling and noise reduction.

    Args:
        target: The function to execute and monitor.
        args: Positional arguments for the target function.
        kwargs: Keyword arguments for the target function.
        base_interval: Base sampling interval (will be adapted during monitoring).
        warmup_time: seconds to distrust a process's CPU reading after that
            process starts, since native-thread-pool spin-up in imported
            libraries (numpy/OpenBLAS, torch, tensorflow, ...) can genuinely
            burn many CPU-seconds within a fraction of a second and get
            mistaken for the workload's own usage. See
            monitor_cpu_and_ram_by_pid_advanced for the full explanation.
            Raise it if your target does heavy native-library imports;
            lower it (down to 0) for lightweight pure-Python workloads.

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
        args=(process.pid, base_interval, result_container, warmup_time),
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
            "warmup_time": warmup_time,
            "warmup_excluded_samples": 0,
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
