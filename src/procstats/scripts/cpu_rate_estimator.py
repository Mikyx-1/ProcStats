"""
Isolated for review — not wired into the monitoring pipeline yet.

Turns a stream of (wall_time, cumulative_cpu_time) readings for one process
into a CPU% estimate that resists the "lumpy OS accounting under
contention" artifact: under heavy multi-process load, the OS's per-process
cumulative CPU-time counter doesn't update smoothly, so a naive
delta/elapsed over a very short window can occasionally report an
impossible or wildly inflated rate for a single sample.

Two independent safeguards, applied in order:

1. Physical clamp
   A process cannot burn more CPU-seconds than wall_seconds * num_cores.
   Any fine-window delta that implies otherwise is provably an accounting
   artifact (not a judgment call) and gets capped to that ceiling.

2. Two-timescale corroboration
   A fine-window (short interval) reading is only trusted as a genuine
   spike if a concurrently tracked, longer sliding window over the same
   process shows a proportionally elevated rate too. A single lumpy OS
   counter update inflates one fine sample but gets diluted away once
   averaged over the longer window. A genuine shift to high intensity
   (e.g. the monitored function entering a CPU-heavy phase) elevates both,
   because the process is actually busy for the whole longer window, not
   just one instant.
"""

from collections import deque
from typing import Deque, Dict, Optional, Tuple


class RobustCpuRateEstimator:
    def __init__(
        self,
        num_cores: int,
        coarse_window_s: float = 0.5,
        corroboration_fraction: float = 0.5,
    ):
        """
        Args:
            num_cores: logical cores on the machine; defines the physical
                ceiling for how much CPU% a single process can report.
            coarse_window_s: length of the sliding "trend" window used to
                corroborate fine-grained spikes. Should be several times
                the expected fine sampling interval (e.g. 0.5s for ~50ms
                samples) so a single-sample artifact gets meaningfully
                diluted, while a real phase change still shows up in it
                within a fraction of a second.
            corroboration_fraction: how close the coarse-window rate must
                be to the fine-window rate for the fine spike to be
                trusted. 0.5 means the coarse window must still show at
                least half of the fine spike's rate to corroborate it.
        """
        self.num_cores = num_cores
        self.coarse_window_s = coarse_window_s
        self.corroboration_fraction = corroboration_fraction

        self._prev: Optional[Tuple[float, float]] = None
        self._history: Deque[Tuple[float, float]] = deque()

    def update(self, wall_time: float, cpu_time: float) -> Optional[Dict]:
        """
        Feed one new (wall_time, cumulative_cpu_time) reading for the
        process (cpu_time = proc.cpu_times().user + .system, in seconds).

        Returns None on the very first call (no delta yet to compute), or
        a dict describing this step:
            fine_rate:    raw instantaneous CPU% from this sample alone
            coarse_rate:  CPU% from the sliding coarse_window_s window
            trusted_rate: the CPU% to actually report
            clamped:      True if the physical ceiling had to cap fine_rate
            corroborated: True if a fine-window spike was confirmed by the
                          coarse window (irrelevant/True when there was no
                          spike to begin with)
        """
        self._history.append((wall_time, cpu_time))
        self._trim_history(wall_time)

        if self._prev is None:
            self._prev = (wall_time, cpu_time)
            return None

        prev_wall, prev_cpu = self._prev
        self._prev = (wall_time, cpu_time)

        fine_elapsed = wall_time - prev_wall
        fine_delta = cpu_time - prev_cpu
        if fine_elapsed <= 0:
            return None

        # --- Safeguard 1: physical clamp ---
        max_possible_delta = fine_elapsed * self.num_cores
        clamped = fine_delta > max_possible_delta
        if clamped:
            fine_delta = max_possible_delta
        fine_rate = 100.0 * fine_delta / fine_elapsed

        # --- Safeguard 2: two-timescale corroboration ---
        coarse_wall, coarse_cpu = self._history[0]
        coarse_elapsed = wall_time - coarse_wall
        coarse_rate = (
            100.0 * (cpu_time - coarse_cpu) / coarse_elapsed
            if coarse_elapsed > 0
            else fine_rate
        )

        corroborated = coarse_rate >= self.corroboration_fraction * fine_rate
        trusted_rate = fine_rate if corroborated else coarse_rate

        return {
            "fine_rate": fine_rate,
            "coarse_rate": coarse_rate,
            "trusted_rate": trusted_rate,
            "clamped": clamped,
            "corroborated": corroborated,
        }

    def _trim_history(self, now: float) -> None:
        while self._history and now - self._history[0][0] > self.coarse_window_s:
            self._history.popleft()


if __name__ == "__main__":
    # Self-test with synthetic data: a quiet baseline, a single-sample
    # counter-batching artifact, then a genuine sustained ramp. Demonstrates
    # that the estimator ignores the artifact but reports the real ramp.
    NUM_CORES = 8
    FINE_INTERVAL = 0.05

    estimator = RobustCpuRateEstimator(num_cores=NUM_CORES, coarse_window_s=0.5)

    wall = 0.0
    cpu = 0.0
    print(f"{'t':>6} {'fine%':>8} {'coarse%':>8} {'trusted%':>9} {'clamped':>8} {'corrob':>7}")

    def step(cpu_delta):
        global wall, cpu
        wall += FINE_INTERVAL
        cpu += cpu_delta
        result = estimator.update(wall, cpu)
        if result:
            print(
                f"{wall:6.2f} {result['fine_rate']:8.1f} {result['coarse_rate']:8.1f} "
                f"{result['trusted_rate']:9.1f} {str(result['clamped']):>8} "
                f"{str(result['corroborated']):>7}"
            )

    # Quiet baseline: steady ~100% (1 core)
    for _ in range(6):
        step(1.0 * FINE_INTERVAL)

    # Single-sample artifact: OS counter "catches up" all at once,
    # implying an impossible/implausible one-off spike (~700%)
    step(7.0 * FINE_INTERVAL)

    # Back to baseline for a bit
    for _ in range(4):
        step(1.0 * FINE_INTERVAL)

    # Genuine sustained ramp to 400% — the function's real high-intensity phase
    for _ in range(10):
        step(4.0 * FINE_INTERVAL)
