"""
ipwgml.data
===========

Provides functionality to access IPWG ML datasets.
"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import gzip
import json
import logging
import multiprocessing
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import re

import click
import requests
#from requests_cache import CachedSession
from requests import Session
from rich.progress import Progress

from ipwgml.definitions import (
    ALL_INPUTS,
    GEOMETRIES,
    REFERENCE_SENSORS,
    SIZES,
    SPLITS,
)
from ipwgml.utils import get_median_time
from ipwgml import config
import ipwgml.logging


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
    Returns the URL from which the IPWGML data can be downloaded.

    Args:
        dataset_name: The name of the dataset ('spr').

    Return:
        A string containing the URL.
    """
    if dataset_name.lower() == "spr":
        if _TESTING:
            return "https://rain.atmos.colostate.edu/gprof_nn/ipwgml/.test"
        else:
            return "https://rain.atmos.colostate.edu/gprof_nn/ipwgml/"
    raise ValueError(
        f"Unknown dataset name: {dataset_name}"
    )


FILE_REGEXP = re.compile('a href="([\w_]*\.nc)"')


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


def get_files_in_dataset(dataset_name: str) -> Dict[str, Any]:
    """
    Lists all available files for a given dataset.

    Args:
        dataset_name: The name of the dataset, i.e., 'spr' for the satellite
            precipitation retrieval (SPR) dataset..

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
        with Progress(console=ipwgml.logging.get_console()) as progress:
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
        progress = Progress(console=ipwgml.logging.get_console())
        rel_path = "/".join(next(iter(files)).split("/")[:-1])
        bar = progress.add_task(f"Downloading files from {rel_path}:", total=len(files))
    else:
        progress = None
        bar = None

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
                rel_path = "/".join(next(iter(files)).split("/")[:-1])
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
            "submitting an issue at github.com/simonpf/ipwgml.",
            failed,
        )

    return [fle for fle in files if fle not in failed]


def download_missing(
        dataset_name: str,
        reference_sensor: str,
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
        dataset_name: The name of the dataset, i.e., 'spr' for the satellite
            precipitation retrieval (SPR) dataset..
        reference_sensor: The reference sensor ('gmi' or 'atms')
        geometry: The viewing geometry ('on_swath', or 'gridded')
        split: The name of the data split, i.e., 'training', 'validation',
            'testing', or 'evaluation'
        subset: The subset, i.e, 'xs', 's', 'm', 'l', or 'xl'; only relevant
            for 'training', 'validation', or 'testing' splits.
        domain: The name of the domain for the 'evaluation' split.
        destination: Path pointing to the local directory containing the IPWGML data.
        progress_base: Whether or not display a progress bar displaying the download progress.

    Return:
        A list containing the local paths of the downloaded files.
    """
    local_files = get_local_files(
        dataset_name,
        reference_sensor,
        geometry,
        split,
        subset,
        domain,
        data_path=destination,
        relative_to=destination
    )
    local_files = map(str, local_files.get(source, []))
    all_files = get_files_in_dataset(dataset_name)
    if split.lower() == "evaluation":
        all_files = all_files[reference_sensor][split][domain][geometry][source]
    else:
        all_files = all_files[reference_sensor][split][subset][geometry][source]

    missing = set(all_files) - set(local_files)
    downloaded = download_files(
        get_data_url(dataset_name),
        missing,
        destination,
        progress_bar=progress_bar
    )
    return [destination / fle for fle in downloaded]


def download_dataset(
        dataset_name: str,
        reference_sensor: str,
        input_data: Union[str, List[str]],
        split: str,
        geometry: str,
        domain: str = "conus",
        subset: str = "xl",
) -> Dict[str, List[Path]]:
    """
    Download IPWGML dataset and return list of local files.

    Args:
        dataset_name: The IPWGML dataset to download.
        reference_sensor: The reference sensor of the dataset.
        input_data: The input data sources for which to download the data.
        split: Which split of the data to download.
        geometry: For which retrieval geometry to download the data.

    Return:
        A dictionary listing locally available files for each input data
        source and the target data.
    """
    data_path = config.get_data_path()

    download_missing(
        dataset_name,
        reference_sensor,
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
            reference_sensor,
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
        reference_sensor=reference_sensor,
        geometry=geometry,
        split=split,
        subset=subset,
        domain=domain,
        data_path=data_path
    )
    return paths


def get_local_files(
        dataset_name: str,
        reference_sensor: str,
        geometry: str,
        split: str,
        subset: str = "xl",
        domain: str = "conus",
        relative_to: Optional[Path] = None,
        data_path: Optional[Path] = None
) -> Dict[str, Path]:
    """
    Get all locally available files.

    Args:
        reference_sensor: The name of the referene sensor.
        geometry: The viewing geometry.
        split: The split name.
        subset: The subset name (only relevant for training, validation,
             and testing splits).
        domain: The domain name (only relevant for evaluation split).
        relative_to: If given, file paths will be relative to the given path
            rather than absolute.
        data_path: The root directory containing IPWG data.

    Return:
        A dictionary mapping data source names to the corresponding files.
    """
    if data_path is None:
        data_path = config.get_data_path()
    else:
        data_path = Path(data_path)

    files = {}
    sources = ["ancillary", "geo", "geo_ir", "target"]
    for source in [reference_sensor,] + sources:
        files[source] = []
        if split != "evaluation":
            for size_ind in range(SIZES.index(subset) + 1):
                rel_path = f"{dataset_name}/{reference_sensor}/{split}/{SIZES[size_ind]}/{geometry}/"
                split_path = data_path / rel_path
                source_files = sorted(list(split_path.glob(f"**/{source}_??????????????.nc")))
                if relative_to is not None:
                    source_files = [path.relative_to(relative_to) for path in source_files]
                files[source] += source_files
        else:
            rel_path = f"{dataset_name}/{reference_sensor}/{split}/{domain}/{geometry}/"
            split_path = data_path / rel_path
            source_files = sorted(list(split_path.glob(f"**/{source}_??????????????.nc")))
            if relative_to is not None:
                source_files = [path.relative_to(relative_to) for path in source_files]
            files[source] += source_files

    ref_times = [get_median_time(path) for path in files["target"]]
    for source in [reference_sensor,] + sources:
        if len(ref_times) == 0 or len(files[source]) == 0:
            continue
        assert set(ref_times) == set([get_median_time(path) for path in files[source]])

    return files


@click.command()
@click.option("--data_path", type=str, default=None)
@click.option("--reference_sensors", type=str, default=None)
@click.option("--geometries", type=str, default=None)
@click.option("--formats", type=str, default=None)
@click.option("--splits", type=str, default=None)
@click.option("--inputs", type=str, default=None)
def cli(
    data_path: Optional[str] = None,
    reference_sensors: Optional[str] = None,
    geometries: Optional[str] = None,
    formats: Optional[str] = None,
    splits: Optional[str] = None,
    inputs: Optional[str] = None,
):
    """
    Download the SPR benchmark dataset.
    """
    dataset = "spr"

    if data_path is None:
        data_path = config.get_data_path()
    else:
        data_path = Path(data_path)
        if not data_path.exists():
            LOGGER.error("The provided 'data_path' does not exist.")
            return 1

    if reference_sensors is None:
        reference_sensors = REFERENCE_SENSORS
    else:
        reference_sensors = [sensor.strip() for sensor in reference_sensors.split(",")]
        for sensor in reference_sensors:
            if sensor not in REFERENCE_SENSORS:
                LOGGER.error(
                    "The sensor '%s' is currently not supported. Currently supported reference_sensors "
                    f"are {REFERENCE_SENSORS}."
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

    if formats is None:
        formats = FORMATS
    else:
        formats = [format.strip() for format in formats.split(",")]
        for format in formats:
            if format not in formats:
                LOGGER.error(
                    "The format '%s' is currently not supported. Currently supported formats"
                    f" are {FORMATS}."
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

    if inputs is None:
        inputs = ALL_INPUTS
    else:
        inputs = [inpt.strip() for inpt in inputs.split(",")]
        for inpt in inputs:
            if inpt not in ALL_INPUTS:
                LOGGER.error(
                    "The input '%s' is currently not supported. Currently supported inputs"
                    f" are {ALL_INPUTS}."
                )
                return 1

    LOGGER.info(f"Starting data download to {data_path}.")

    for sensor in reference_sensors:
        for geometry in geometries:
            for inpt in inputs + ["target"]:
                for fmt in formats:
                    for split in splits:
                        if split == "evaluation":
                            dataset = f"spr/{sensor}/{split}/{geometry}/{inpt}"
                        else:
                            dataset = f"spr/{sensor}/{split}/{geometry}/{fmt}/{inpt}"
                        try:
                            download_missing(dataset, data_path, progress_bar=True)
                        except Exception:
                            LOGGER.exception(
                                f"An  error was encountered when downloading dataset '{dataset}'."
                            )

    config.set_data_path(data_path)


def list_local_files_rec(path: Path) -> Dict[str, Any]:
    """
    Recursive listing of ipwgml data files.

    Args:
        path: A path pointing to a directory containing ipwgml files.

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
    List available ipwgml files.
    """
    data_path = config.get_data_path()
    files = list_local_files_rec(data_path)
    return files
