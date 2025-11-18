"""
Tests for the satrain.pytorch.data module.
"""

import torch

from satrain.pytorch.datasets import SatRainTabular, SatRainSpatial


def test_dataset_satrain_tabular(satrain_gmi_on_swath_train):
    """
    Test loading of tabular data from the SatRain dataset.
    """
    data_path = satrain_gmi_on_swath_train
    dataset = SatRainTabular(
        base_sensor="gmi",
        geometry="on_swath",
        split="training",
        retrieval_input=["gmi", "geo", "geo_ir", "ancillary"],
        data_path=data_path,
        download=False,
    )

    assert len(dataset) > 0
    x, y = dataset[0]
    assert "obs_gmi" in x
    assert x["obs_gmi"].shape == (13,)
    assert "obs_geo_ir" in x
    assert x["obs_geo_ir"].shape == (1,)
    assert "obs_geo" in x
    assert x["obs_geo"].shape == (16,)
    assert "ancillary" in x
    assert y["surface_precip"].numel() == 1


def test_dataset_satrain_tabular_stacked(satrain_gmi_on_swath_train):
    """
    Test loading of tabular data from the SatRain dataset.
    """
    data_path = satrain_gmi_on_swath_train
    dataset = SatRainTabular(
        base_sensor="gmi",
        geometry="on_swath",
        split="training",
        retrieval_input=["gmi", "geo_ir", "ancillary"],
        data_path=data_path,
        stack=True,
        download=False,
    )

    assert len(dataset) > 0
    x, y = next(iter(dataset))
    assert isinstance(x, torch.Tensor)
    assert y["surface_precip"].numel() == 1


def test_dataset_satrain_tabular_batched(satrain_gmi_on_swath_train):
    """
    Test loading of tabular data from the SatRain dataset with batching.
    """
    batch_size = 1024
    data_path = satrain_gmi_on_swath_train
    dataset = SatRainTabular(
        base_sensor="gmi",
        geometry="on_swath",
        split="training",
        retrieval_input=["gmi", "geo", "geo_ir", "ancillary"],
        data_path=data_path,
        download=False,
        batch_size=batch_size,
    )

    assert len(dataset) > 0
    for ind, (x, y) in enumerate(dataset):
        if ind < len(dataset) - 1:
            assert "obs_gmi" in x
            assert x["obs_gmi"].shape == (batch_size, 13)
            assert x["obs_geo_ir"].shape == (batch_size, 1)
            assert x["obs_geo"].shape == (batch_size, 16)
            assert "ancillary" in x
            assert y["surface_precip"].numel() == batch_size

    dataset = SatRainTabular(
        base_sensor="gmi",
        geometry="on_swath",
        split="training",
        retrieval_input=[
            {"name": "gmi", "channels": [0, -2, -1]},
            {"name": "geo"},
            {"name": "geo_ir"},
            {"name": "ancillary", "variables": ["two_meter_temperature"]},
        ],
        data_path=data_path,
        download=False,
        batch_size=batch_size,
    )

    assert len(dataset) > 0
    for ind, (x, y) in enumerate(dataset):
        if ind < len(dataset) - 1:
            assert "obs_gmi" in x
            assert x["obs_gmi"].shape == (batch_size, 3)
            assert x["obs_geo_ir"].shape == (batch_size, 1)
            assert x["obs_geo"].shape == (batch_size, 16)
            assert "ancillary" in x
            assert x["ancillary"].shape == (batch_size, 1)
            assert y["surface_precip"].numel() == batch_size


def test_dataset_satrain_spatial(satrain_gmi_gridded_train):
    """
    Test loading of tabular data from the SatRain dataset.
    """
    data_path = satrain_gmi_gridded_train
    dataset = SatRainSpatial(
        base_sensor="gmi",
        geometry="gridded",
        split="training",
        retrieval_input=["gmi", "ancillary", "geo_ir"],
        data_path=data_path,
        download=False,
        augment=False
    )
    assert len(dataset) > 0
    x, y = next(iter(dataset))
    assert "obs_gmi" in x
    assert x["obs_gmi"].shape == (13, 256, 256)
    assert "ancillary" in x
    assert y["surface_precip"].shape == (256, 256)

    # Test augmentation
    retrieval_input = [
        {"name": "gmi", "nan": 0.0},
        {"name": "ancillary", "nan": 0.0},
    ]
    dataset = SatRainSpatial(
        base_sensor="gmi",
        geometry="gridded",
        split="training",
        retrieval_input=retrieval_input,
        data_path=data_path,
        download=False,
        augment=True
    )
    assert len(dataset) > 0
    x, y = next(iter(dataset))
    assert "obs_gmi" in x
    assert x["obs_gmi"].shape == (13, 256, 256)
    assert "ancillary" in x
    assert y["surface_precip"].shape == (256, 256)


def test_dataset_satrain_spatial_stacked(satrain_gmi_gridded_train):
    """
    Test loading of tabular data from the SatRain dataset.
    """
    data_path = satrain_gmi_gridded_train

    dataset = SatRainSpatial(
        base_sensor="gmi",
        geometry="gridded",
        split="training",
        retrieval_input=["gmi", "ancillary", "geo_ir"],
        stack=True,
        data_path=data_path,
        download=False,
        augment=False
    )
    assert len(dataset) > 0
    x, y = next(iter(dataset))
    assert isinstance(x, torch.Tensor)

    retrieval_input = [
        {"name": "gmi", "nan": 0.0},
        {"name": "ancillary", "nan": 0.0},
    ]
    dataset = SatRainSpatial(
        base_sensor="gmi",
        geometry="gridded",
        split="training",
        retrieval_input=retrieval_input,
        stack=True,
        data_path=data_path,
        download=False,
        augment=True
    )
    assert len(dataset) > 0
    x, y = next(iter(dataset))
    assert isinstance(x, torch.Tensor)
