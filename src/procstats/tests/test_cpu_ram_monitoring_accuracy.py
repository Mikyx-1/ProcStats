"""
Slow, real-CPU-burning accuracy checks: burn a known target CPU% and check
that the monitor reports something close to it. Not run by default (see
`slow` marker in pyproject.toml) -- run explicitly with:

    pytest -m slow src/procstats/tests/test_cpu_ram_monitoring_accuracy.py -v
"""

import multiprocessing

import pytest

from procstats.scripts.cpu_ram_monitoring import monitor_cpu_and_ram_on_function_advanced
from procstats.tests.test_burn_cpu_ram import burn_cpu_accurate

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        multiprocessing.cpu_count() < 2, reason="needs >=2 cores to burn 200% CPU"
    ),
]

TARGET_PERCENT = 200
BURN_DURATION_S = 4
# cpu_avg/cpu_p95 go through outlier_filter before averaging, which reliably
# strips startup-spike samples -- 5 manual runs across 100/200/400% targets
# stayed within 7% of target, so +/-20% has real margin without being loose
# enough to miss a genuine regression.
AVG_TOLERANCE = 0.2
# cpu_max skips outlier_filter by design, so it depends entirely on
# warmup_time excluding the process-startup contamination (numpy/OpenBLAS
# thread-pool spin-up mistaken for the workload's own usage -- see
# monitor_cpu_and_ram_by_pid_advanced's warmup_time docstring). With the
# default warmup_time, 6 manual runs across 50/100/200/400% targets stayed
# within 41% of target (worst case was the 50% run, where a small absolute
# jitter is a large relative one) -- +/-40% has some margin without being
# loose enough to miss a real regression back to the old 300-4700% spikes.
MAX_TOLERANCE = 0.4


def _burn_and_monitor():
    return monitor_cpu_and_ram_on_function_advanced(
        target=burn_cpu_accurate,
        args=(TARGET_PERCENT, BURN_DURATION_S),
        base_interval=0.05,
    )


def test_cpu_avg_matches_burned_target():
    result = _burn_and_monitor()
    low, high = TARGET_PERCENT * (1 - AVG_TOLERANCE), TARGET_PERCENT * (1 + AVG_TOLERANCE)
    assert low <= result["cpu_avg"] <= high, (
        f"cpu_avg={result['cpu_avg']:.1f}% not within "
        f"[{low:.0f}, {high:.0f}] of a {TARGET_PERCENT}% burn"
    )


def test_cpu_max_matches_burned_target():
    # Relies on the default warmup_time to exclude each process's own
    # startup window, during which a native library's thread-pool spin-up
    # (numpy/OpenBLAS here) can genuinely burn many CPU-seconds in a
    # fraction of a second -- real CPU time, not a measurement artifact,
    # so RobustCpuRateEstimator's safeguards alone can't catch it. See
    # test_cpu_max_without_warmup_reproduces_the_spike below for proof
    # this test would fail without warmup_time.
    result = _burn_and_monitor()
    low, high = TARGET_PERCENT * (1 - MAX_TOLERANCE), TARGET_PERCENT * (1 + MAX_TOLERANCE)
    assert low <= result["cpu_max"] <= high, (
        f"cpu_max={result['cpu_max']:.1f}% not within "
        f"[{low:.0f}, {high:.0f}] of a {TARGET_PERCENT}% burn"
    )


def test_cpu_max_without_warmup_reproduces_the_spike():
    # Regression lock: proves warmup_time is actually what fixes cpu_max
    # above, not an incidental side effect of something else. Disabling it
    # should reproduce the original bug -- cpu_max wildly overshooting a
    # 200% burn because process-startup thread-pool noise gets counted as
    # the workload's own usage.
    result = monitor_cpu_and_ram_on_function_advanced(
        target=burn_cpu_accurate,
        args=(TARGET_PERCENT, BURN_DURATION_S),
        base_interval=0.05,
        warmup_time=0,
    )
    assert result["warmup_excluded_samples"] == 0
    assert result["cpu_max"] > TARGET_PERCENT * 2, (
        f"cpu_max={result['cpu_max']:.1f}% -- expected the known startup-spike "
        f"artifact to reproduce with warmup_time=0"
    )
