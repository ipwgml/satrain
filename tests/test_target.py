"""
Tests for the satrain.target module.
"""

import numpy as np
import xarray as xr

from satrain.utils import extract_samples
from satrain.data import get_local_files
from satrain.target import TargetConfig


def test_load_reference_precip(satrain_gmi_gridded_train):
    """
    Test loading of target data.
    """
    files = get_local_files(
        dataset_name="satrain",
        base_sensor="gmi",
        split="training",
        subset="xl",
        geometry="gridded",
        data_path=satrain_gmi_gridded_train
    )
    target_files = files["target"]

    target_data = xr.load_dataset(target_files[0])

    target_config = TargetConfig(
        target="surface_precip",
        min_rqi=1.0,
        no_snow=False,
        no_hail=False,
        min_gcf=None,
        max_gcf=None,
    )
    precip_data = target_config.load_reference_precip(target_files[0])
    valid = np.isfinite(precip_data)
    assert np.isclose(
        target_data["radar_quality_index"].data[valid].min(), 1.0, rtol=1e-3
    )

    # Test varying minimum RQI requirement
    target_config = TargetConfig(
        target="surface_precip",
        min_rqi=0.0,
        no_snow=False,
        no_hail=False,
        min_gcf=None,
        max_gcf=None,
    )
    precip_data = target_config.load_reference_precip(target_files[0])
    valid = np.isfinite(precip_data)
    assert target_data["radar_quality_index"].data[valid].min() < 1.0

    # Test no snow requirement.
    target_config = TargetConfig(
        target="surface_precip",
        min_rqi=1.0,
        no_snow=True,
        no_hail=False,
        min_gcf=None,
        max_gcf=None,
    )
    precip_data = target_config.load_reference_precip(target_files[0])
    valid = np.isfinite(precip_data)
    assert np.isclose(
        target_data["radar_quality_index"].data[valid].min(), 1.0, rtol=1e-3
    )
    assert (target_data["snow_fraction"].data[valid] == 0.0).all()


def test_load_reference_precip_tabular(satrain_gmi_gridded_train):
    """
    Test loading of target data.
    """
    files = get_local_files(
        dataset_name="satrain",
        base_sensor="gmi",
        split="training",
        subset="xl",
        geometry="gridded",
        data_path=satrain_gmi_gridded_train
    )
    target_files = files["target"]

    target_data = xr.load_dataset(target_files[0])
    valid = np.isfinite(target_data.surface_precip)
    target_data = extract_samples(target_data, xr.DataArray(valid, dims=target_data.surface_precip.dims))

    target_config = TargetConfig(
        target="surface_precip",
        min_rqi=1.0,
        no_snow=False,
        no_hail=False,
        min_gcf=None,
        max_gcf=None,
    )
    precip_data = target_config.load_reference_precip(target_data)
    valid = np.isfinite(precip_data)
    assert np.isclose(
        target_data["radar_quality_index"].data[valid].min(), 1.0, rtol=1e-3
    )

    # Test varying minimum RQI requirement
    target_config = TargetConfig(
        target="surface_precip",
        min_rqi=0.0,
        no_snow=False,
        no_hail=False,
        min_gcf=None,
        max_gcf=None,
    )
    precip_data = target_config.load_reference_precip(target_data)
    valid = np.isfinite(precip_data)
    assert target_data["radar_quality_index"].data[valid].min() < 1.0

    # Test no snow requirement.
    target_config = TargetConfig(
        target="surface_precip",
        min_rqi=1.0,
        no_snow=True,
        no_hail=False,
        min_gcf=None,
        max_gcf=None,
    )
    precip_data = target_config.load_reference_precip(target_data)
    valid = np.isfinite(precip_data)
    assert np.isclose(
        target_data["radar_quality_index"].data[valid].min(), 1.0, rtol=1e-3
    )
    assert (target_data["snow_fraction"].data[valid] == 0.0).all()


def test_load_precip_mask_spatial(satrain_gmi_gridded_train):
    """
    Test loading of target data.
    """
    files = get_local_files(
        dataset_name="satrain",
        base_sensor="gmi",
        split="training",
        subset="xl",
        geometry="gridded",
        data_path=satrain_gmi_gridded_train
    )
    target_files = files["target"]

    target_data = xr.load_dataset(target_files[0])

    thresh = np.random.uniform(0.5, 2.0)
    target_config = TargetConfig(precip_threshold=thresh)
    precip_data = target_config.load_reference_precip(target_data)
    precip_mask = 0 < target_config.load_precip_mask(target_data)
    mask = target_config.get_mask(target_data)
    assert (precip_data[~mask][precip_mask[~mask]] >= thresh).all()


def test_load_precip_mask_tabular(satrain_gmi_gridded_train):
    """
    Test loading of target data.
    """
    files = get_local_files(
        dataset_name="satrain",
        base_sensor="gmi",
        split="training",
        subset="xl",
        geometry="gridded",
        data_path=satrain_gmi_gridded_train
    )
    target_files = files["target"]

    target_data = xr.load_dataset(target_files[0])
    valid = np.isfinite(target_data.surface_precip)
    target_data = extract_samples(target_data, xr.DataArray(valid, dims=target_data.surface_precip.dims))

    thresh = np.random.uniform(0.5, 2.0)
    target_config = TargetConfig(precip_threshold=thresh)
    precip_data = target_config.load_reference_precip(target_data)
    precip_mask = 0 < target_config.load_precip_mask(target_data)
    mask = target_config.get_mask(target_data)
    assert (precip_data[~mask][precip_mask[~mask]] >= thresh).all()


def test_load_load_heavy_precip_mask_spatial(satrain_gmi_gridded_train):
    """
    Test loading of target data.
    """
    files = get_local_files(
        dataset_name="satrain",
        base_sensor="gmi",
        split="training",
        subset="xl",
        geometry="gridded",
        data_path=satrain_gmi_gridded_train
    )
    target_files = files["target"]

    target_data = xr.load_dataset(target_files[0])

    thresh = np.random.uniform(10.0, 20.0)
    target_config = TargetConfig(heavy_precip_threshold=thresh)
    precip_data = target_config.load_reference_precip(target_data)
    precip_mask = 0 < target_config.load_heavy_precip_mask(target_data)
    mask = target_config.get_mask(target_data)
    assert (precip_data[~mask][precip_mask[~mask]] >= thresh).all()


def test_load_load_heavy_precip_mask_tabular(satrain_gmi_gridded_train):
    """
    Test loading of target data.
    """
    files = get_local_files(
        dataset_name="satrain",
        base_sensor="gmi",
        split="training",
        subset="xl",
        geometry="gridded",
        data_path=satrain_gmi_gridded_train
    )
    target_files = files["target"]

    target_data = xr.load_dataset(target_files[0])
    valid = np.isfinite(target_data.surface_precip)
    target_data = extract_samples(target_data, xr.DataArray(valid, dims=target_data.surface_precip.dims))

    thresh = np.random.uniform(10.0, 20.0)
    target_config = TargetConfig(heavy_precip_threshold=thresh)
    precip_data = target_config.load_reference_precip(target_data)
    precip_mask = 0 < target_config.load_heavy_precip_mask(target_data)
    mask = target_config.get_mask(target_data)
    assert (precip_data[~mask][precip_mask[~mask]] >= thresh).all()
