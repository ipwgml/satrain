"""
Tests for the satrain.data module.
"""
import os

import pytest

from satrain.data import (
    enable_testing,
    get_files_in_dataset,
    get_local_files,
    get_files,
    load_tabular_data
)


def test_get_files_in_dataset():
    """
    Tests finding files from SatRain dataset and ensure that more than on file is found.
    """
    files = get_files_in_dataset("satrain")
    assert "gmi" in files
    assert len(files["gmi"]["training"]["xl"]["gridded"]["gmi"]) > 0
    assert len(files["gmi"]["training"]["xl"]["gridded"]["gmi"]) > len(files["gmi"]["training"]["l"]["gridded"]["gmi"])


@pytest.mark.parametrize("sensor_and_fixture", [["gmi", "satrain_gmi_gridded_train"], ["atms", "satrain_atms_gridded_train"]])
def test_download_files_satrain_gmi_gridded_train(request, sensor_and_fixture):
    """
    Ensure that fixture successfully downloaded files.
    """
    sensor, fixture = sensor_and_fixture
    data_path = request.getfixturevalue(fixture)
    for source in [sensor, "ancillary", "geo_ir", "target"]:
        files = get_local_files("satrain", sensor, "gridded", "training", data_path=data_path)
        assert len(files) == 7

@pytest.mark.parametrize("sensor_and_fixture", [["gmi", "satrain_gmi_gridded_train"], ["atms", "satrain_atms_gridded_train"]])
def test_get_files(request, sensor_and_fixture):
    """
    Ensure the get_files function returns locally available files.
    """
    sensor, fixture = sensor_and_fixture
    data_path = request.getfixturevalue(fixture)
    for source in [sensor, "ancillary", "geo_ir", "target"]:
        files = get_files(sensor, "training", input_data=[sensor, "ancillary"], geometry="gridded", data_path=data_path)
        assert sensor in files
        assert "target" in files
        assert len(files) == 3

@pytest.mark.parametrize("sensor_and_fixture", [["gmi", "satrain_gmi_on_swath_train"], ["atms", "satrain_atms_on_swath_train"]])
def test_download_files_satrain_gmi_on_swath_train(request, sensor_and_fixture):
    """
    Ensure that fixture successfully downloaded files.
    """
    sensor, fixture = sensor_and_fixture
    data_path = request.getfixturevalue(fixture)
    for source in [sensor, "ancillary", "geo_ir", "target"]:
        files = get_local_files("satrain", sensor, "on_swath", "training", data_path=data_path)
        assert len(files) == 7

@pytest.mark.parametrize("sensor_and_fixture", [["gmi", "satrain_gmi_testing"], ["atms", "satrain_atms_testing"]])
def test_download_files_satrain_gmi_testing(request, sensor_and_fixture):
    """
    Ensure that fixture successfully downloaded files.
    """
    sensor, fixture = sensor_and_fixture
    data_path = request.getfixturevalue(fixture)
    files = get_local_files("satrain", sensor, "on_swath", "testing", domain="conus", data_path=data_path)
    for source in [sensor, "ancillary", "geo_ir", "target"]:
        assert len(files[source]) == 1


@pytest.mark.parametrize("sensor_and_fixture", [["gmi", "satrain_gmi_on_swath_train_dataset"], ["atms", "satrain_atms_on_swath_train_dataset"]])
def test_download_satrain_gmi_dataset(request, sensor_and_fixture):
    """
    Ensure that download dataset function
    """
    sensor, fixture = sensor_and_fixture
    files = request.getfixturevalue(fixture)
    assert len(files[sensor]) == 5
    assert len(files["target"]) == 5


@pytest.mark.parametrize("sensor_and_fixture", [["gmi", "satrain_gmi_on_swath_train"], ["atms", "satrain_atms_on_swath_train"]])
def test_load_tabular_data(request, sensor_and_fixture):
    sensor, fixture = sensor_and_fixture
    data_path = request.getfixturevalue(fixture)

    inpt, target = load_tabular_data("satrain", sensor, "on_swath", "training", "xs", [sensor], data_path=data_path)
    assert sensor in inpt
    assert 0 < inpt[sensor].samples.size
    assert inpt[sensor].samples.size == target.samples.size
