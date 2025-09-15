"""
Tests for the satrain.baselines module.
"""
import pytest

from satrain.baselines import load_baseline_results


@pytest.mark.parametrize("base_sensor", ["gmi", "atms"])
def test_load_baseline_results(base_sensor):
    """
    Test that loading of baseline results works as expected.
    """
    results = load_baseline_results(base_sensor)
    assert "ERA5" in results.algorithm
    assert f"GPROF V7 ({base_sensor.upper()})" in results.algorithm

    results_austria = load_baseline_results(base_sensor, domain="austria")
    assert results["bias"].data[0] != results_austria["bias"].data[0]
    assert "ERA5" in results_austria.algorithm
    assert f"GPROF V7 ({base_sensor.upper()})" in results_austria.algorithm

    results = load_baseline_results(base_sensor, baselines=["era5"])
    assert "ERA5" in results.algorithm
    assert f"GPROF V7 ({base_sensor.upper()})" not in results.algorithm

    results = load_baseline_results(base_sensor, baselines=[f"gprof_v07"])
    assert "ERA5" not in results.algorithm
    assert f"GPROF V7 ({base_sensor.upper()})" in results.algorithm
