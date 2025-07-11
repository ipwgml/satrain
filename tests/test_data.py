"""
Tests for the ipwgml.data module.
"""
import os


from ipwgml.data import (
    enable_testing,
    get_files_in_dataset,
    get_local_files
)


def test_get_files_in_dataset():
    """
    Tests finding files from SatRain dataset and ensure that more than on file is found.
    """
    files = get_files_in_dataset("satrain")
    assert "gmi" in files
    assert len(files["gmi"]["training"]["xl"]["gridded"]["gmi"]) > 0
    assert len(files["gmi"]["training"]["xl"]["gridded"]["gmi"]) > len(files["gmi"]["training"]["l"]["gridded"]["gmi"])


def test_download_files_satrain_gmi_gridded_train(satrain_gmi_gridded_train):
    """
    Ensure that fixture successfully downloaded files.
    """
    for source in ["gmi", "ancillary", "geo_ir", "target"]:
        files = get_local_files("satrain", "gmi", "gridded", "training", data_path=satrain_gmi_gridded_train)
        assert len(files) == 5

def test_download_files_satrain_gmi_on_swath_train(satrain_gmi_gridded_train):
    """
    Ensure that fixture successfully downloaded files.
    """
    for source in ["gmi", "ancillary", "geo_ir", "target"]:
        files = get_local_files("satrain", "gmi", "on_swath", "training", data_path=satrain_gmi_gridded_train)
        assert len(files) == 5

def test_download_files_satrain_gmi_evaluation(satrain_gmi_evaluation):
    """
    Ensure that fixture successfully downloaded files.
    """
    files = get_local_files("satrain", "gmi", "on_swath", "evaluation", domain="conus", data_path=satrain_gmi_evaluation)
    for source in ["gmi", "ancillary", "geo_ir", "target"]:
        assert len(files[source]) == 1


def test_download_satrain_gmi_dataset(satrain_gmi_on_swath_train_dataset):
    """
    Ensure that download dataset function
    """
    files = satrain_gmi_on_swath_train_dataset
    assert len(files["gmi"]) == 5
    assert len(files["target"]) == 5
