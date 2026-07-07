import pytest

from procstats.scripts.cpu_rate_estimator import RobustCpuRateEstimator


def test_first_update_returns_none():
    est = RobustCpuRateEstimator(num_cores=4)
    assert est.update(0.0, 0.0) is None


def test_non_positive_elapsed_returns_none():
    est = RobustCpuRateEstimator(num_cores=4)
    est.update(1.0, 0.5)
    assert est.update(1.0, 0.6) is None  # same wall_time -> fine_elapsed == 0


def test_steady_state_rate_reported_without_clamp():
    est = RobustCpuRateEstimator(num_cores=8, coarse_window_s=0.5)
    est.update(0.0, 0.0)
    result = est.update(0.05, 0.05)  # 1 core busy for the whole interval -> 100%

    assert result["clamped"] is False
    assert result["fine_rate"] == pytest.approx(100.0)
    assert result["trusted_rate"] == pytest.approx(100.0)


def test_physical_clamp_caps_impossible_delta():
    est = RobustCpuRateEstimator(num_cores=4, coarse_window_s=0.5)
    est.update(0.0, 0.0)
    # A cpu_time delta of 1s over a 0.1s wall interval implies 1000% on a
    # 4-core machine -- impossible, must be capped to 4 * 100%.
    result = est.update(0.1, 1.0)

    assert result["clamped"] is True
    assert result["fine_rate"] == pytest.approx(400.0)


def test_uncorroborated_spike_falls_back_to_coarse_rate():
    est = RobustCpuRateEstimator(num_cores=8, coarse_window_s=0.5, corroboration_fraction=0.5)
    wall = 0.0
    cpu = 0.0
    # Quiet baseline: ~100% for several samples, giving the coarse window a
    # low-rate history to check the upcoming spike against.
    for _ in range(6):
        wall += 0.05
        cpu += 0.05
        est.update(wall, cpu)

    # Single-sample artifact: an OS counter "catch-up" implies ~700% for
    # this one sample, but the coarse window (still dominated by the
    # ~100% baseline) won't corroborate it.
    wall += 0.05
    cpu += 7 * 0.05
    result = est.update(wall, cpu)

    assert result["corroborated"] is False
    assert result["trusted_rate"] == pytest.approx(result["coarse_rate"])
    assert result["trusted_rate"] < result["fine_rate"]


def test_sustained_ramp_is_corroborated_and_trusted():
    est = RobustCpuRateEstimator(num_cores=8, coarse_window_s=0.5, corroboration_fraction=0.5)
    wall = 0.0
    cpu = 0.0
    for _ in range(6):
        wall += 0.05
        cpu += 0.05
        est.update(wall, cpu)

    # Sustained ramp to ~400%: every sample keeps the coarse window
    # elevated too, so it should be trusted as a real rate rather than
    # dismissed as a one-off artifact.
    result = None
    for _ in range(10):
        wall += 0.05
        cpu += 4 * 0.05
        result = est.update(wall, cpu)

    assert result["corroborated"] is True
    assert result["trusted_rate"] == pytest.approx(result["fine_rate"])
    assert result["trusted_rate"] == pytest.approx(400.0, rel=0.05)


def test_history_is_trimmed_to_coarse_window():
    est = RobustCpuRateEstimator(num_cores=8, coarse_window_s=0.2)
    wall = 0.0
    cpu = 0.0
    for _ in range(20):
        wall += 0.05
        cpu += 0.05
        est.update(wall, cpu)

    # coarse_window_s=0.2 with 0.05s steps should keep ~4-5 samples, not
    # all 20 fed in.
    assert len(est._history) <= 5
    assert wall - est._history[0][0] <= 0.2
