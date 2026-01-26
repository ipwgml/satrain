"""
Tests for the satrain.input module.
"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from satrain.data import get_local_files
from satrain.input import (
    normalize,
    InputConfig,
    GMI,
    Ancillary,
    calculate_input_features,
)


def test_normalize():
    """
    Test normalization of input data.
    """
    data = np.random.rand(128, 128)
    stats = xr.Dataset({"min": 0, "max": 1, "mean": 0.5, "std_dev": 1.0})

    data_n = normalize(data, stats, "standardize")
    assert (0.0 <= data).all()
    assert (data_n < 0.0).any()
    assert data.mean() > 0.0
    assert np.isclose(data_n.mean(), 0.0, atol=3e-2)

    data = np.random.rand(128, 128)
    stats = xr.Dataset({"min": 0, "max": 1, "mean": 0.5, "std_dev": 1.0})
    data_n = normalize(data, stats, "minmax")
    assert (0.0 <= data).all()
    assert (data_n < 0.0).any()
    assert data.mean() > 0.0
    assert np.isclose(data_n.mean(), 0.0, atol=3e-2)

    data = np.random.rand(128, 128)
    data[data > 0.5] = np.nan
    data_n = normalize(data, stats, "minmax", nan=-1.5)
    assert np.isclose(data_n.min(), -1.5)


def test_parsing():
    """
    Test parsing of input data configs.
    """
    inpt = "gmi"
    cfg = InputConfig.parse(inpt)
    assert isinstance(cfg, GMI)

    inpt = {"name": "GMI", "channels": [0, 1]}
    cfg = InputConfig.parse(inpt)
    assert isinstance(cfg, GMI)

    cfg = GMI(channels=[0, 1])
    assert isinstance(cfg, GMI)

    inpt = "ancillary"
    cfg = InputConfig.parse(inpt)
    assert isinstance(cfg, Ancillary)

    inpt = {"name": "ancillary", "variables": ["two_meter_temperature"]}
    cfg = InputConfig.parse(inpt)
    assert isinstance(cfg, Ancillary)

    cfg = Ancillary(variables=["two_meter_temperature"])
    assert isinstance(cfg, Ancillary)



@pytest.mark.parametrize("sensor_and_fixture", [["gmi", "satrain_gmi_gridded_train"], ["atms", "satrain_atms_gridded_train"]])
def test_pmw_input(request, sensor_and_fixture):
    """
    Test loading of PMW input data.
    """
    sensor, fixture = sensor_and_fixture
    data_path = request.getfixturevalue(fixture)

    files = get_local_files(
        dataset_name="satrain",
        base_sensor=sensor,
        split="training",
        geometry="gridded",
        subset="xl",
        data_path=data_path
    )
    pmw_files = files[sensor]
    target_files = files["target"]
    inpt = {"name": sensor, "channels": [0, 1]}
    cfg = InputConfig.parse(inpt)
    target_data = xr.load_dataset(target_files[0])
    inpt_data = cfg.load_data(pmw_files[0], target_time=target_data.time)

    assert inpt_data[f"obs_{sensor}"].shape[0] == cfg.features[f"obs_{sensor}"]
    assert inpt_data[f"eia_{sensor}"].shape[0] == cfg.features[f"obs_{sensor}"]

    assert f"obs_{sensor}" in inpt_data
    assert inpt_data[f"obs_{sensor}"].shape[0] == 2
    assert f"eia_{sensor}" in inpt_data

    assert cfg.stats is not None

    obs = inpt_data[f"obs_{sensor}"]
    valid = np.isfinite(obs)
    assert np.all(obs[valid] > 0.0)

    # Test replacement of NAN value
    inpt = {"name": f"{sensor}", "channels": [0, 1], "normalize": "minmax", "nan": -1.5}
    cfg = InputConfig.parse(inpt)
    target_data = xr.load_dataset(target_files[0])
    inpt_data = cfg.load_data(pmw_files[0], target_time=target_data.time)

    obs = inpt_data[f"obs_{sensor}"]
    assert np.isfinite(obs).all()
    valid = np.isfinite(obs)
    assert not np.all(obs[valid] > 0.0)


def test_ancillary_input(satrain_gmi_gridded_train):
    """
    Test loading of ancillary input data.
    """
    files = get_local_files(
        dataset_name="satrain",
        base_sensor="gmi",
        split="training",
        geometry="gridded",
        subset="xl",
        data_path=satrain_gmi_gridded_train
    )
    target_files = files["target"]
    anc_files = files["ancillary"]
    inpt = {"name": "ancillary", "variables": ["total_column_water_vapor"]}
    cfg = InputConfig.parse(inpt)

    target_data = xr.load_dataset(target_files[0])
    inpt_data = cfg.load_data(anc_files[0], target_time=target_data.time)

    assert "ancillary" in inpt_data
    assert inpt_data["ancillary"].shape[0] == 1
    assert inpt_data["ancillary"].shape[0] == cfg.features["ancillary"]


def test_geo_ir_input(satrain_gmi_gridded_train):
    """
    Test loading of GEO-IR input data.
    """
    files = get_local_files(
        dataset_name="satrain",
        base_sensor="gmi",
        split="training",
        geometry="gridded",
        subset="xl",
        data_path=satrain_gmi_gridded_train
    )
    geo_ir_files = files["geo_ir"]
    target_files = files["target"]

    inpt = {"name": "geo_ir"}
    cfg = InputConfig.parse(inpt)
    target_data = xr.load_dataset(target_files[0])
    inpt_data = cfg.load_data(geo_ir_files[0], target_time=target_data.time)
    assert "obs_geo_ir" in inpt_data
    assert inpt_data["obs_geo_ir"].shape[0] == 1
    assert inpt_data["obs_geo_ir"].shape[0] == cfg.features["obs_geo_ir"]

    assert cfg.stats is not None


def test_geo_input_gridded(satrain_gmi_gridded_train):
    """
    Test loading of GEO input data.
    """
    files = get_local_files(
        dataset_name="satrain",
        base_sensor="gmi",
        split="training",
        geometry="gridded",
        subset="xl",
        data_path=satrain_gmi_gridded_train
    )
    geo_files = files["geo"]
    target_files = files["target"]

    inpt = {"name": "geo", "channels": [0, 3, 9]}
    cfg = InputConfig.parse(inpt)
    target_data = xr.load_dataset(target_files[0])
    inpt_data = cfg.load_data(geo_files[0], target_time=target_data.time)
    assert "obs_geo" in inpt_data
    assert inpt_data["obs_geo"].shape[0] == len(cfg.channels)
    assert inpt_data["obs_geo"].shape[0] == cfg.features["obs_geo"]


def test_geo_input_on_swath(satrain_gmi_on_swath_train):
    """
    Test loading of GEO input data.
    """
    files = get_local_files(
        dataset_name="satrain",
        base_sensor="gmi",
        split="training",
        geometry="on_swath",
        subset="xl",
        data_path=satrain_gmi_on_swath_train
    )
    geo_files = files["geo"]
    target_files = files["target"]

    inpt = {"name": "geo", "channels": [0, 3, 9]}
    cfg = InputConfig.parse(inpt)
    target_data = xr.load_dataset(target_files[0])
    inpt_data = cfg.load_data(geo_files[0], target_time=target_data.time)
    assert "obs_geo" in inpt_data
    assert inpt_data["obs_geo"].shape[0] == len(cfg.channels)
    assert inpt_data["obs_geo"].shape[0] == cfg.features["obs_geo"]

    inpt = {"name": "geo"}
    cfg = InputConfig.parse(inpt)
    target_data = xr.load_dataset(target_files[0])
    inpt_data = cfg.load_data(geo_files[0], target_time=target_data.time)
    assert "obs_geo" in inpt_data
    assert inpt_data["obs_geo"].shape[0] == len(cfg.channels)
    assert inpt_data["obs_geo"].shape[0] == cfg.features["obs_geo"]
    assert (inpt_data["obs_geo"] > 100).any()

    # Ensure that input is normalized.
    inpt = {"name": "geo", "normalize": "minmax", "nan": -1.5}
    cfg = InputConfig.parse(inpt)
    target_data = xr.load_dataset(target_files[0])
    inpt_data = cfg.load_data(geo_files[0], target_time=target_data.time)
    assert "obs_geo" in inpt_data
    assert inpt_data["obs_geo"].shape[0] == len(cfg.channels)
    assert inpt_data["obs_geo"].shape[0] == cfg.features["obs_geo"]
    assert (inpt_data["obs_geo"] <= 1.1).all()


@pytest.mark.parametrize("sensor", ["gmi", "atms"])
def test_calculate_input_features(sensor):
    """
    Test calculation of input features.
    """
    inputs = [
        {"name": "gmi", "include_angles": True, "channels": [0, 3, 4]},
        {"name": "ancillary", "variables": ["two_meter_temperature", "surface_type"]},
        {"name": "geo_ir"},
        {"name": "geo", "channels": [0, 1, 2]},
    ]

    features = calculate_input_features(inputs, stack=False)
    assert features["obs_gmi"] == 3
    assert features["eia_gmi"] == 3
    assert features["ancillary"] == 2
    assert features["obs_geo_ir"] == 1
    assert features["obs_geo"] == 3

    features = calculate_input_features(inputs, stack=True)
    assert features == 12
