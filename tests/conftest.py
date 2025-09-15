import pytest

from satrain.data import (
    enable_testing,
    download_missing,
    download_dataset
)


enable_testing()


@pytest.fixture(scope="session")
def satrain_gmi_gridded_train(tmp_path_factory):
    """
    Fixture to download satellite-precipitation retrieval benchmark data for GMI with
    gridded geometry.
    """
    dest = tmp_path_factory.mktemp("satrain")
    for source in ["gmi", "target", "geo_ir", "geo", "ancillary"]:
        download_missing(
            dataset_name="satrain",
            base_sensor="gmi",
            geometry="gridded",
            split="training",
            source=source,
            destination=dest
        )
    return dest


@pytest.fixture(scope="session")
def satrain_atms_gridded_train(tmp_path_factory):
    """
    Fixture to download satellite-precipitation retrieval benchmark data for ATMS with
    gridded geometry.
    """
    dest = tmp_path_factory.mktemp("satrain")
    for source in ["atms", "target", "geo_ir", "geo", "ancillary"]:
        download_missing(
            dataset_name="satrain",
            base_sensor="atms",
            geometry="gridded",
            split="training",
            source=source,
            destination=dest
        )
    return dest


@pytest.fixture(scope="session")
def satrain_atms_on_swath_train(tmp_path_factory):
    """
    Fixture to download satellite-precipitation retrieval benchmark data for ATMS with
    on_swath geometry.
    """
    dest = tmp_path_factory.mktemp("satrain")
    for source in ["atms", "target", "geo_ir", "geo", "ancillary"]:
        download_missing(
            dataset_name="satrain",
            base_sensor="atms",
            geometry="on_swath",
            split="training",
            source=source,
            destination=dest
        )
    return dest


@pytest.fixture(scope="session")
def satrain_gmi_on_swath_train(tmp_path_factory):
    """
    Fixture to download satellite-precipitation retrieval benchmark data for GMI with
    on_swath geometry.
    """
    dest = tmp_path_factory.mktemp("satrain")
    for source in ["gmi", "target", "geo_ir", "geo", "ancillary"]:
        download_missing(
            dataset_name="satrain",
            base_sensor="gmi",
            geometry="on_swath",
            split="training",
            source=source,
            destination=dest
        )
    return dest


@pytest.fixture(scope="session")
def satrain_gmi_on_swath_train_dataset(tmp_path_factory):
    """
    Fixture to download satellite-precipitation retrieval benchmark data for GMI with
    on_swath geometry.
    """
    dest = tmp_path_factory.mktemp("satrain")
    return download_dataset(
        "satrain",
        "gmi",
        ["gmi"],
        split="training",
        geometry="on_swath",
        data_path=dest
    )


@pytest.fixture(scope="session")
def satrain_atms_on_swath_train_dataset(tmp_path_factory):
    """
    Fixture to download satellite-precipitation retrieval benchmark data for ATMS with
    on_swath geometry.
    """
    dest = tmp_path_factory.mktemp("satrain")
    return download_dataset(
        "satrain",
        "atms",
        ["atms"],
        split="training",
        geometry="on_swath",
        data_path=dest
    )


@pytest.fixture(scope="session")
def satrain_atms_testing(tmp_path_factory):
    """
    Fixture to download SatRain test data for ATMS.
    """
    dest = tmp_path_factory.mktemp("satrain")
    for source in ["atms", "target", "geo_ir", "geo", "ancillary"]:
        for geometry in ["gridded", "on_swath"]:
            download_missing(
                dataset_name="satrain",
                base_sensor="atms",
                geometry=geometry,
                split="testing",
                domain="conus",
                source=source,
                destination=dest
            )
    return dest


@pytest.fixture(scope="session")
def satrain_gmi_testing(tmp_path_factory):
    """
    Fixture to download SatRain test data for GMI.
    """
    dest = tmp_path_factory.mktemp("satrain")
    for source in ["gmi", "target", "geo_ir", "geo", "ancillary"]:
        for geometry in ["gridded", "on_swath"]:
            download_missing(
                dataset_name="satrain",
                base_sensor="gmi",
                geometry=geometry,
                split="testing",
                domain="conus",
                source=source,
                destination=dest
            )
    return dest
