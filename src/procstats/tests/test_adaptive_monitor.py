import statistics

import pytest

from procstats.scripts.cpu_ram_monitoring import AdaptiveMonitor


@pytest.fixture
def monitor():
    return AdaptiveMonitor(pid=1)


def test_stability_score_needs_at_least_three_values(monitor):
    assert monitor.calculate_stability_score([1.0, 2.0]) == float("inf")


def test_stability_score_zero_mean_is_zero(monitor):
    assert monitor.calculate_stability_score([0.0, 0.0, 0.0]) == 0.0


def test_stability_score_matches_coefficient_of_variation(monitor):
    values = [10.0, 12.0, 8.0, 11.0]
    score = monitor.calculate_stability_score(values)
    expected = 100 * (statistics.stdev(values) / statistics.mean(values))
    assert score == pytest.approx(expected)


def test_adaptive_interval_uses_base_with_too_few_samples(monitor):
    assert monitor.adaptive_interval([1.0, 2.0]) == monitor.base_interval


def test_adaptive_interval_shrinks_for_high_variability(monitor):
    unstable = [10.0, 90.0, 5.0, 95.0, 2.0]
    assert monitor.calculate_stability_score(unstable) > 20
    interval = monitor.adaptive_interval(unstable)
    assert interval == max(monitor.min_interval, monitor.base_interval * 0.5)


def test_adaptive_interval_grows_for_low_variability(monitor):
    stable = [50.0, 51.0, 49.0, 50.5, 49.5]
    assert monitor.calculate_stability_score(stable) < 5
    interval = monitor.adaptive_interval(stable)
    assert interval == min(monitor.max_interval, monitor.base_interval * 2)


def test_adaptive_interval_keeps_base_for_medium_variability(monitor):
    medium = [50.0, 55.0, 48.0, 53.0, 47.0]
    score = monitor.calculate_stability_score(medium)
    assert 5 <= score <= 20
    assert monitor.adaptive_interval(medium) == monitor.base_interval


def test_outlier_filter_needs_at_least_three_values(monitor):
    values = [1.0, 2.0]
    assert monitor.outlier_filter(values) == values


def test_outlier_filter_no_op_when_all_identical(monitor):
    values = [5.0, 5.0, 5.0, 5.0]
    assert monitor.outlier_filter(values) == values


def test_outlier_filter_removes_far_outlier(monitor):
    values = [10.0, 11.0, 9.0, 10.5, 9.5, 100.0]
    filtered = monitor.outlier_filter(values)
    assert 100.0 not in filtered
    assert filtered == [10.0, 11.0, 9.0, 10.5, 9.5]
