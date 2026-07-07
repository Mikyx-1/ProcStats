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
# cpu_max skips outlier_filter by design and is currently known-broken (see
# test_cpu_max_matches_burned_target below), so its tolerance is irrelevant
# to whether that test passes -- kept wide just to document intent.
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
    # Known to currently fail: a process-startup OS-accounting artifact
    # (cpu_times() jumping by ~1-2s of CPU time in a ~50ms sample window)
    # slips past both RobustCpuRateEstimator safeguards during ramp-up --
    # the physical clamp uses the system-wide core count as a per-process
    # ceiling, and there isn't yet enough coarse-window history to refute
    # the spike via corroboration. Left failing intentionally to track the
    # bug rather than hiding it.
    result = _burn_and_monitor()
    low, high = TARGET_PERCENT * (1 - MAX_TOLERANCE), TARGET_PERCENT * (1 + MAX_TOLERANCE)
    assert low <= result["cpu_max"] <= high, (
        f"cpu_max={result['cpu_max']:.1f}% not within "
        f"[{low:.0f}, {high:.0f}] of a {TARGET_PERCENT}% burn"
    )
