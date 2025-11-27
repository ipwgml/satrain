"""
satrain.utils
=============

Defines helper functions used throught the satrain package.
"""

from contextlib import contextmanager
from datetime import datetime
import gc
from pathlib import Path
from typing import Union

import hdf5plugin
import xarray as xr


@contextmanager
def open_if_required(path_or_dataset: str | Path | xr.Dataset) -> xr.Dataset:
    """
    Open and close an xarray.Dataset or do nothing if data is already loaded.

    Args:
         path_or_dataset: A Path pointing to a NetCDF4 to open of an already
             loaded dataset.

    Return:
         An xarray.Dataset providing access to the loaded data.
    """
    try:
        handle = None
        if isinstance(path_or_dataset, (str, Path)):
            handle = xr.open_dataset(path_or_dataset, engine="h5netcdf", chunks=None, cache=False)
            yield handle
        else:
            yield path_or_dataset
    finally:
        if handle is not None:
            handle.close()
        del path_or_dataset
        del handle


def get_median_time(path: Union[Path, str]) -> datetime:
    """
    Extract median time from filename.
    """
    if isinstance(path, Path):
        path = path.name
    date = datetime.strptime(path.split("_")[-1][:-3], "%Y%m%d%H%M%S")
    return date

def cleanup_files(path: Path, no_action: bool = False) -> None:
    """
    Removes all files that do not have matching files in all input and target files.

    Args:
        path: A Path object pointing to the folder containing the
            SatRain training scenes.
        no_action: Just print filename, don't remove any files.
    """
    path = Path(path)

    all_times = None
    for fldr in ["gmi", "target", "ancillary", "geo_ir", "geo"]:
        if not (path / "on_swath" / fldr).exists():
            continue
        files = sorted(list((path / "on_swath" / fldr).glob("*.nc")))
        times = set(map(get_median_time, files))
        if all_times is None:
            all_times = times
        else:
            all_times = all_times.intersection(times)
        if not (path / "gridded" / fldr).exists():
            continue
        files = sorted(list((path / "gridded" / fldr).glob("*.nc")))
        times = set(map(get_median_time, files))
        all_times = all_times.intersection(times)

    for fldr in ["target", "gmi", "ancillary", "geo_ir", "geo"]:
        if not (path / "on_swath" / fldr).exists():
            continue
        files = sorted(list((path / "on_swath" / fldr).glob("*.nc")))
        for fle in files:
            if get_median_time(fle) not in all_times:
                print("Extra file: ", fle)
                if not no_action:
                    fle.unlink()

        files = sorted(list((path / "on_gridded" / fldr).glob("*.nc")))
        for fle in files:
            if get_median_time(fle) not in all_times:
                print("Extra file: ", fle)
                if not no_action:
                    fle.unlink()


def extract_samples(dataset: xr.Dataset, mask: xr.DataArray):
    """
    Extract tabular data from spatial scenes based on reference data availability.

    Args:
        dataset: The spatial input data.
        mask: A mask identifying the pixels with valid reference data.

    Return:
        A new dataset containing only samples with valid reference data.
    """
    dataset = dataset.transpose(*mask.dims, ...)
    extracted = xr.Dataset()
    for name in dataset:
        var = dataset[name]
        if var.dims[:2] == mask.dims:
            var_e = var.data[mask.data]
            extracted[name] = (("samples",) + var.dims[2:], var_e)
        else:
            extracted[name] = var
    for dim in dataset.dims:
        extracted[dim] = dataset[dim].copy()
    return extracted
