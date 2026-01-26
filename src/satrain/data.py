"""
satrain.data
============

Provides functionality to access and download the SatRain data.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import cache
import gzip
import json
import logging
import multiprocessing
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import re

import click
import numpy as np
import requests
#from requests_cache import CachedSession
from requests import Session
from rich.progress import Progress
import xarray as xr

from satrain.definitions import (
    ALL_INPUTS,
    GEOMETRIES,
    BASE_SENSORS,
    SIZES,
    SPLITS,
)
from satrain.utils import get_median_time, extract_samples
from satrain import config
import satrain.logging


LOGGER = logging.getLogger(__name__)

_TESTING = False

def enable_testing() -> None:
    """
    Enable test mode.
    """
    global _TESTING
    _TESTING = True


def get_data_url(dataset_name: str) -> str:
    """
    Returns the URL from which the SatRain data can be downloaded.

    Args:
        dataset_name: The name of the dataset ('satrain').

    Return:
        A string containing the URL.
    """
    if dataset_name.lower() == "satrain":
        if _TESTING:
            return "https://rain.atmos.colostate.edu/ipwgml/.test"
        else:
            return "https://rain.atmos.colostate.edu/ipwgml/"
    raise ValueError(
        f"Unknown dataset name: {dataset_name}"
    )


FILE_REGEXP = re.compile(r'a href="([\w_]*\.nc)"')


def load_json_maybe_gzipped(path: Path):
    """
    Loads a JSON file, handling both plain and gzipped (.gz) files.

    Parameters:
        path: A Path object pointing to the file to read.

    Returns:
        The deserialized Python object.
    """
    filename = path.name
    open_fn = gzip.open if filename.endswith(".gz") else open
    mode = 'rt' if filename.endswith(".gz") else 'r'
    with open_fn(path, mode, encoding='utf-8') as f:
        return json.load(f)


@cache
def get_files_in_dataset(dataset_name: str) -> Dict[str, Any]:
    """
    Lists all available files for a given dataset.

    Args:
        dataset_name: The name of the dataset, i.e., 'satrain' for the Satellite
            Rain Estimation and Detection (SatRain) benchmar dataset.

    Return:
        A nested dictionary containing all files in the dataset.
    """
    if _TESTING:
        fname = f"files_{dataset_name.lower()}_test.json"
        path = Path(__file__).parent / "files" / fname
        if not path.exists():
            path = Path(__file__).parent / "files" / (fname + ".gz")
    else:
        fname = f"files_{dataset_name.lower()}.json"
        path = Path(__file__).parent / "files" / fname
        if not path.exists():
            path = Path(__file__).parent / "files" / (fname + ".gz")

    files = load_json_maybe_gzipped(path)
    return files


def download_file(url: str, destination: Path) -> None:
    """
    Download file from server.

    Args:
        url: A string containing the URL of the file to download.
        destination: The destination to which to write the file.
    """
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        with open(destination, "wb") as output:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    output.write(chunk)


@contextmanager
def progress_bar_or_not(progress_bar: bool) -> Progress | None:
    """
    Context manager for a optional progress bar.
    """
    if progress_bar:
        with Progress(console=satrain.logging.get_console()) as progress:
            yield progress
    else:
        yield None


def download_files(
        base_url: str,
        files: List[str],
        destination: Path,
        progress_bar: bool = True,
        retries: int = 3,
) -> List[str]:
    """
    Download files using multiple threads.

    Args:
        base_url: The URL from which the remote data is available.
        files: A list containing the relative paths of the files to download.
        destination: A Path object pointing to the local path to which to download the files.
        progress_bar: Whether or not to display a progress bar during download.
        retries: The number of retries to perform for failed files.

    Return:
        A list of the downloaded files.
    """
    n_threads = min(multiprocessing.cpu_count(), 8)
    pool = ThreadPoolExecutor(max_workers=n_threads)
    ctr = 0

    failed = []

    if progress_bar and len(files) > 0:
        progress = Progress(console=satrain.logging.get_console())
    else:
        progress = None

    while ctr < retries and len(files) > 0:

        tasks = []
        failed = []
        for path in files:
            *path, fname = path.split("/")
            path = "/".join(path)
            output_path = destination / path
            output_path.mkdir(parents=True, exist_ok=True)
            url = base_url + "/" + str(path) + "/" + fname
            tasks.append(pool.submit(download_file, url, output_path / fname))

        with progress_bar_or_not(progress_bar=progress_bar) as progress:
            if progress is not None:
                rel_path = "/".join(next(iter(files)).split("/")[:3])
                bar = progress.add_task(
                    f"Downloading files from {rel_path}:", total=len(files)
                )
            else:
                bar = None

            for path, task in zip(files, tasks):

                try:
                    task.result()
                    if progress is not None:
                        progress.advance(bar, advance=1)
                except Exception:
                    LOGGER.exception(
                        "Encountered an error when trying to download files %s.",
                        path.split("/")[-1],
                    )
                    failed.append(path)

        ctr += 1
        files = failed

    if len(failed) > 0:
        LOGGER.warning(
            "The download of the following files failed: %s. If the issue persists please consider "
            "submitting an issue at github.com/ipwgml/satrain.",
            failed,
        )

    return [fle for fle in files if fle not in failed]


def download_missing(
        dataset_name: str,
        base_sensor: str,
        geometry: str,
        split: str,
        source: str,
        subset: str = "xl",
        domain: str = "conus",
        destination: Path = None,
        progress_bar: bool = False,
) -> None:
    """
    Download missing file from dataset.

    Args:
        dataset_name: The name of the dataset, i.e., 'satrain' for the Satellite
            Rain Estimation and Detection (SatRain) dataset.
        base_sensor: The base sensor ('gmi' or 'atms')
        geometry: The viewing geometry ('on_swath', or 'gridded')
        split: The name of the data split, i.e., 'training', 'validation', or 'testing'.
        subset: The subset, i.e, 'xs', 's', 'm', 'l', or 'xl'; only relevant
            for 'training', 'validation', or 'testing' splits.
        domain: The name of the test domain. Only relevant if split='testing'.
        destination: Path pointing to the local directory containing the SatRain data.
        progress_base: Whether or not display a progress bar displaying the download progress.

    Return:
        A list containing the local paths of the downloaded files.
    """
    local_files = get_local_files(
        dataset_name,
        base_sensor,
        geometry,
        split,
        subset,
        domain,
        data_path=destination,
        relative_to=destination,
        check_consistency=False
    )
    local_files = map(str, local_files.get(source, []))
    all_files = get_files_in_dataset(dataset_name)
    if split.lower() == "testing":
        all_files = all_files[base_sensor][split][domain][geometry].get(source, [])
    else:
        all_files = all_files[base_sensor][split][subset][geometry].get(source, [])

    missing = set(all_files) - set(local_files)

    if 0 < len(missing):
        LOGGER.info(
            "Downloading %s %s files for base_sensor %s, split %s, and geometry %s.",
            len(missing),
            source,
            base_sensor,
            split,
            geometry
        )
        downloaded = download_files(
            get_data_url(dataset_name),
            missing,
            destination,
            progress_bar=progress_bar
        )
        return [destination / fle for fle in downloaded]
    return []


def download_dataset(
        dataset_name: str,
        base_sensor: str,
        input_data: Union[str, List[str]],
        split: str,
        geometry: str,
        domain: str = "conus",
        subset: str = "xl",
        data_path: Optional[Union[str, Path]] = None
) -> Dict[str, List[Path]]:
    """
    Download SatRain dataset and return list of local files.

    Args:
        dataset_name: The SatRain dataset to download.
        base_sensor: The base sensor of the dataset.
        input_data: The input data sources for which to download the data.
        split: Which split of the data to download.
        geometry: For which retrieval geometry to download the data.
        domain: Name of the test domain (optional).
        subset: The subset to download (xs, s, m, l, xl).
        data_path: Optional path pointing to the local data path.

    Return:
        A dictionary listing locally available files for each input data
        source and the target data.
    """
    if data_path is None:
        data_path = config.get_data_path()
    else:
        data_path = Path(data_path)

    download_missing(
        dataset_name,
        base_sensor,
        geometry,
        split,
        source="target",
        subset=subset,
        domain=domain,
        destination=data_path,
        progress_bar=True,
    )

    if not isinstance(input_data, list):
        input_data = [input_data]
    input_data = [inpt if isinstance(inpt, str) else inpt.name for inpt in input_data]

    for inpt in input_data:
        download_missing(
            dataset_name,
            base_sensor,
            geometry,
            split,
            source=inpt,
            subset=subset,
            domain=domain,
            destination=data_path,
            progress_bar=True,
        )

    paths = get_local_files(
        dataset_name=dataset_name,
        base_sensor=base_sensor,
        geometry=geometry,
        split=split,
        subset=subset,
        domain=domain,
        data_path=data_path
    )
    return paths


def get_local_files(
        dataset_name: str,
        base_sensor: str,
        geometry: str,
        split: str,
        subset: str = "xl",
        domain: str = "conus",
        relative_to: Optional[Path] = None,
        data_path: Optional[Path] = None,
        check_consistency: bool = True
) -> Dict[str, Path]:
    """
    Get all locally available files.

    Args:
        base_sensor: The name of the referene sensor.
        geometry: The viewing geometry.
        split: The split name.
        subset: The subset name (only relevant for training and validation splits).
        domain: The domain name (only relevant for testing split).
        relative_to: If given, file paths will be relative to the given path
            rather than absolute.
        data_path: The root directory containing IPWG data.
        check_consitency: Whether or not to check consistency of the found files.

    Return:
        A dictionary mapping data source names to the corresponding files.
    """
    if data_path is None:
        data_path = config.get_data_path()
    else:
        data_path = Path(data_path)
    files = {}
    sources = ["ancillary", "geo", "geo_t", "geo_ir", "geo_ir_t", "target"]
    for source in [base_sensor,] + sources:
        files[source] = []
        if split != "testing":
            for size_ind in range(SIZES.index(subset) + 1):
                rel_path = f"{dataset_name}/{base_sensor}/{split}/{SIZES[size_ind]}/{geometry}/"
                split_path = data_path / rel_path
                source_files = sorted(list(split_path.glob(f"**/{source}_??????????????.nc")))
                if relative_to is not None:
                    source_files = [path.relative_to(relative_to) for path in source_files]
                files[source] += source_files
        else:
            rel_path = f"{dataset_name}/{base_sensor}/{split}/{domain}/{geometry}/"
            split_path = data_path / rel_path
            source_files = sorted(list(split_path.glob(f"**/{source}_??????????????.nc")))
            if relative_to is not None:
                source_files = [path.relative_to(relative_to) for path in source_files]
            files[source] += source_files

    if check_consistency:
        ref_times = set([get_median_time(path) for path in files["target"]])
        for source in [base_sensor,] + sources:
            if len(ref_times) == 0 or len(files[source]) == 0:
                continue
            ref_times = set(ref_times)
            source_times = set([get_median_time(path) for path in files[source]])
            assert set(ref_times) == source_times

    return files


def get_files(
        base_sensor: str,
        split: str,
        input_data: Union[str, List[str]],
        geometry: str,
        domain: str = "conus",
        subset: str = "xl",
        data_path: Optional[Union[str, Path]] = None,
        download: bool = True
) -> Dict[str, List[Path]]:
    """
    Get files in SatRain dataset.

    Args:
        base_sensor: The base sensor of the dataset.
        split: Which split of the data to get (training, validation, testing).
        input_data: List of the input data sources ('gmi', 'atms', 'geo', 'geo_t', 'geo_ir', 'geo_ir_t', 'ancillary')
        geometry: For which retrieval geometry to download the data.
        domain: Name of the domain for the testing data ('austria', 'conus', 'korea')
        subset: The subset to download (xs, s, m, l, xl).
        data_path: Optional path pointing to the path to store the data.
        download: Download missing data.

    Return:
        A dictionary listing locally available files for each input data
        source and the target data.
    """
    if data_path is None:
        data_path = config.get_data_path()
    else:
        data_path = Path(data_path)

    if download:
        download_missing(
            "satrain",
            base_sensor,
            geometry,
            split,
            source="target",
            subset=subset,
            domain=domain,
            destination=data_path,
            progress_bar=True,
        )

    if not isinstance(input_data, list):
        input_data = [input_data]
    input_data = [inpt if isinstance(inpt, str) else inpt.name for inpt in input_data]

    for inpt in input_data:
        if download:
            download_missing(
                "satrain",
                base_sensor,
                geometry,
                split,
                source=inpt,
                subset=subset,
                domain=domain,
                destination=data_path,
                progress_bar=True,
            )

    paths = get_local_files(
        dataset_name="satrain",
        base_sensor=base_sensor,
        geometry=geometry,
        split=split,
        subset=subset,
        domain=domain,
        data_path=data_path
    )
    return {inpt: files for inpt, files in paths.items() if inpt in input_data + ["target"]}


def load_tabular_data(
        dataset_name: str,
        base_sensor: str,
        geometry: str,
        split: str,
        subset: str,
        retrieval_input: List[str | Dict[str, Any] | "InputConfig"],
        target_config: Optional["TargetConfig"] = None,
        data_path: Optional[Path] = None
):
    """
    Load data in tabular format.

    Args:
        dataset_name: The name of the dataset.
        base_sensor: The base sensor.
        geometry: The geometry, i.e., 'on_swath' or 'gridded'.
        split: Training or validation.
        subset: The subset: 'xs', 's', 'm', 'l', 'xl'
        retrieval_input: A list specifying the retrieval input.
        target_config: A config dict or object defining the target data configuration.
        data_path: Optional path pointing to the local data path.

    Return:
        A tuple ``input_data, target`` with ``input_data`` being a dictionary containing
        the retrieval input as separate xarray.Datasets and ``target`` containing the target
        data.
    """
    from .input import parse_retrieval_inputs
    from .target import TargetConfig

    if not base_sensor.lower() in ["gmi", "atms"]:
        raise ValueError("Base_Sensor must be one of ['gmi', 'atms'].")
    base_sensor = base_sensor.lower()

    if not geometry.lower() in ["gridded", "on_swath"]:
        raise ValueError("Geomtry must be one of ['gridded', 'on_swath'].")
    geometry = geometry.lower()

    if not split.lower() in ["training", "validation", "testing"]:
        raise ValueError(
            "Split must be one of ['training', 'validation', 'testing']"
        )

    retrieval_input = parse_retrieval_inputs(retrieval_input)
    if target_config is None:
        target_config = TargetConfig()
    if isinstance(target_config, dict):
        target_config = TargetConfig(**target_config)

    files = download_dataset(
        dataset_name, base_sensor, retrieval_input, split, geometry, subset=subset, data_path=data_path
    )

    target_files = files["target"]

    target_data = []
    input_data = {inpt.name: [] for inpt in retrieval_input}

    from tqdm import tqdm
    for ind, target_file in tqdm(enumerate(target_files), total=len(target_files)):
        data = xr.load_dataset(target_file)
        valid = ~target_config.get_mask(data)
        valid = xr.DataArray(
            data=valid,
            dims=data.surface_precip.dims
        )
        data = extract_samples(data, valid)
        if "time" in data.coords:
            data = data.reset_index("time")
        target_data.append(data)

        ref_time = get_median_time(target_file)

        for inpt in retrieval_input:
            input_time = get_median_time(files[inpt.name][ind])
            if ref_time != input_time:
                raise ValueError(
                    "Encountered an input files %s that is inconsistent with the corresponding "
                    "reference file %s. This indicates that the dataset has not been downloaded "
                    "properly."
                )
            data = extract_samples(xr.load_dataset(files[inpt.name][ind]), valid)
            if "time" in data.coords:
                data = data.reset_index("time")
            input_data[inpt.name].append(data)

    target_data = xr.concat(target_data, dim="samples")
    input_data = {name: xr.concat(data, dim="samples") for name, data in input_data.items()}
    return input_data, target_data


def list_local_files_rec(path: Path) -> Dict[str, Any]:
    """
    Recursive listing of SatRain data files.

    Args:
        path: A path pointing to a directory containing SatRain files.

    Return:
        A dictionary containing all sub-directories

    """
    netcdf_files = sorted(list(path.glob("*.nc")))
    if len(netcdf_files) > 0:
        return netcdf_files

    files = {}
    for child in path.iterdir():
        if child.is_dir():
            files[child.name] = list_local_files_rec(child)
    return files


def list_local_files() -> Dict[str, Any]:
    """
    List available SatRain files.
    """
    data_path = config.get_data_path()
    files = list_local_files_rec(data_path / "satrain")
    return files


@click.command()
@click.option("--data_path", type=str, default=None, help="The local directory in which to store the SatRain dataset.")
@click.option(
    "--base_sensors",
    type=str,
    default=None,
    help="Comma-separated list of he base sensors to download."
)
@click.option(
    "--geometries",
    type=str,
    default=None,
    help="Comma-separated list of the geometries to download ('on_swath', 'gridded' or both)."
)
@click.option(
    "--splits",
    type=str,
    default=None,
    help="Comma-separated list of the splits to ddownload ('training', 'validation', 'testing')"
)
@click.option(
    "--subset",
    type=str,
    default=None,
    help="The subset to download."
)
@click.option(
    "--inputs",
    type=str,
    default=None,
    help="Comma-separated list of the input sources to download ('gmi', 'atms', 'geo', 'geo_ir', 'geo_t', 'geo_ir_t', 'ancillary')"
)
def cli(
    data_path: Optional[str] = None,
    base_sensors: Optional[str] = None,
    geometries: Optional[str] = None,
    splits: Optional[str] = None,
    subset: Optional[str] = None,
    inputs: Optional[str] = None,
):
    """
    Download the SatRain benchmark dataset.
    """
    dataset = "satrain"

    if data_path is None:
        data_path = config.get_data_path()
    else:
        data_path = Path(data_path)
        if not data_path.exists():
            LOGGER.error("The provided 'data_path' does not exist.")
            return 1

    if base_sensors is None:
        base_sensors = BASE_SENSORS
    else:
        base_sensors = [sensor.strip() for sensor in base_sensors.split(",")]
        for sensor in base_sensors:
            if sensor not in BASE_SENSORS:
                LOGGER.error(
                    "The sensor '%s' is currently not supported. Currently supported base_sensors "
                    f"are {BASE_SENSORS}."
                )
                return 1

    if geometries is None:
        geometries = GEOMETRIES
    else:
        geometries = [geometry.strip() for geometry in geometries.split(",")]
        for geometry in geometries:
            if geometry not in GEOMETRIES:
                LOGGER.error(
                    "The geometry '%s' is currently not supported. Currently supported geometries"
                    f" are {GEOMETRIES}."
                )
                return 1

    if splits is None:
        splits = SPLITS
    else:
        splits = [split.strip() for split in splits.split(",")]
        for split in splits:
            if split not in SPLITS:
                LOGGER.error(
                    "The split '%s' is currently not supported. Currently supported splits"
                    f" are {SPLITS}."
                )
                return 1

    if subset is None:
        subset = "xl"
    else:
        subset = subset.lower()
        if subset not in SIZES:
            LOGGER.error(
                "%s is not a valid subset. Valid subsets are {SPLITS}."
            )
            return 1

    if inputs is None:
        inputs = ALL_INPUTS
    else:
        inputs = [inpt.strip() for inpt in inputs.split(",")]
        for inpt in inputs:
            if inpt not in ALL_INPUTS + ["target"]:
                LOGGER.error(
                    f"The input 'inpt' is currently not supported. Currently supported inputs"
                    f" are {ALL_INPUTS}."
                )
                return 1

    LOGGER.info(f"Starting data download to {data_path}.")


    for sensor in base_sensors:
        for geometry in geometries:
            for inpt in inputs + ["target"]:
                for split in splits:
                    if sensor == "gmi" and inpt == "atms":
                        continue
                    if sensor == "atms" and inpt == "gmi":
                        continue

                    if split == "testing":
                        domains = ["conus", "korea", "austria"]
                    else:
                        domains = [None]

                    for domain in domains:
                        try:
                            download_missing(
                                dataset_name=dataset,
                                base_sensor=sensor,
                                geometry=geometry,
                                split=split,
                                source=inpt,
                                subset=subset,
                                domain=domain,
                                destination=data_path,
                                progress_bar=True,
                            )
                        except Exception:
                            LOGGER.exception(
                                f"An  error was encountered when downloading dataset '{dataset}'."
                            )

    config.set_data_path(data_path)

