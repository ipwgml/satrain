"""
ipwg.pytorch.data
=================

This module provides PyTorch dataset classes for loading the SatRain data.
The :class:`SatRainTabular` will load data in tabular format while the
:class:`SatRainSpatial` will load data in spatial format.

"""
from datetime import datetime
from functools import cache, cached_property, partial
import gc
import logging
from math import ceil
import multiprocessing
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import tracemalloc
import sys

import numpy as np
import torch
from torch.utils.data import Dataset
import hdf5plugin
import xarray as xr
from rich.progress import Progress
from torchvision.transforms import v2

import satrain
from satrain.data import download_missing, get_local_files, progress_bar_or_not
from satrain.definitions import ALL_INPUTS
from satrain import config
from satrain.input import InputConfig, parse_retrieval_inputs
from satrain.target import TargetConfig
from satrain.utils import get_median_time, extract_samples


LOGGER = logging.getLogger(__name__)


class SatRainTabular(Dataset):
    """
    Dataset class for SatRain data in tabular format.

    For efficiency, the SatRainTabular data loads all of the training data into memory
    upon creation and provides the option to perform batching within the dataset
    instead of in the data loader.
    """

    def __init__(
        self,
        base_sensor: str,
        geometry: str,
        split: str,
        subset: str = "xl",
        batch_size: Optional[int] = None,
        shuffle: Optional[bool] = True,
        retrieval_input: List[str | Dict[str, Any] | InputConfig] = None,
        target_config: Optional[TargetConfig] = None,
        stack: bool = False,
        subsample: Optional[float] = None,
        data_path: Optional[Path] = None,
        download: bool = True,
    ):
        """
        Args:
            base_sensor: The base_sensor for which to load the benchmark dataset.
            geometry: Whether to load on_swath or regridded observations.
            split: Whether to load training ('training'), validation ('validation'), or
                 test ('testing') splits.
            batch_size: If given will return batched input data.
            shuffle: Whether or not to shuffle the samples in the dataset.
            retrieval_input: List of the retrieval inputs to load. The list should contain
                names of retrieval input sources ("pmw", "geo", "geo_ir", "ancillary"), dictionaries
                defining the input name and additional input options, or InputConfig. If not explicitly
                specified all available input data is loaded.
            target_config: An optional TargetConfig specifying quality requirements for the retrieval
                target data to load.
            stack: If 'False', the input will be loaded as a dictionary containing the input tensors
                from all input dataset. If 'True', the tensors will be concatenated along the
                feature axis and only a single tensor is loaded instead of dictionary.
            subsample: An optional fraction specifying how much of the dataset to load per epoch.
            satrain_path: Path containing or to which to download the SatRain data.
            download: If 'True', missing data will be downloaded upon dataset creation. Otherwise, only
                locally available files will be used.
        """
        super().__init__()

        if data_path is None:
            data_path = config.get_data_path()
        else:
            data_path = Path(data_path)

        if not base_sensor.lower() in ["gmi", "atms"]:
            raise ValueError("Base_Sensor must be one of ['gmi', 'atms'].")
        self.base_sensor = base_sensor.lower()

        if not geometry.lower() in ["gridded", "on_swath"]:
            raise ValueError("Geomtry must be one of ['gridded', 'on_swath'].")
        self.geometry = geometry.lower()

        if not split.lower() in ["training", "validation", "testing"]:
            raise ValueError(
                "Split must be one of ['training', 'validation', 'testing']"
            )
        self.split = split
        self.subset = subset
        self.batch_size = batch_size
        self.shuffle = shuffle

        if retrieval_input is None:
            retrieval_input = ALL_INPUTS
        self.retrieval_input = np.array(parse_retrieval_inputs(retrieval_input))

        if target_config is None:
            target_config = TargetConfig()
        elif isinstance(target_config, dict):
            target_config = TargetConfig(**target_config)
        self.target_config = target_config

        self.stack = stack
        self.subsample = subsample

        self.geo_data = None
        self.geo_ir_data = None
        self.ancillary_data = None
        self.target_data = None

        # Load target data and mask
        if download:
            sources = set([inpt.name for inpt in self.retrieval_input] + ["target"])
            for source in sources:
                download_missing(
                    dataset_name="satrain",
                    base_sensor=self.base_sensor,
                    geometry=self.geometry,
                    source=source,
                    split=self.split,
                    subset=self.subset,
                    progress_bar=True,
                    destination=data_path
                )
        files = get_local_files(
            dataset_name="satrain",
            base_sensor=self.base_sensor,
            geometry=self.geometry,
            split=self.split,
            subset=self.subset,
            data_path=data_path,
        )
        if len(files["target"]) == 0:
            raise ValueError(
                f"Couldn't find any target data files. "
                " Please make sure that the satrain data path is correct or "
                "set 'download' to True to download the file."
            )
        self._load_training_data(files, progress_bar=True)

        self.rng = np.random.default_rng(seed=42)
        if self.shuffle:
            self.indices = self.rng.permutation(self.target_data.samples.size)
        else:
            self.indices = np.arange(self.target_data.samples.size)


    def _load_training_data(
            self,
            files: Dict[str, Path],
            progress_bar: bool = False
    ) -> None:
        """
        Load tabular training data into memory.

        This function extract all pixels with valid reference data from the SatRain input and reference files
        and stores them as xarray.Datasets in the attributes of the object.

        Args:
            files: A dictionary mapping data source names to lists containing the paths of the SatRain scenes
                 from which to load the data.
            progress_bar: Flag indicating whether or not to display a progress bar when loading the data.
        """
        target_files = files["target"]
        self.target_data = []
        for inpt in self.retrieval_input:
            setattr(self, inpt.name + "_data", [])

        LOGGER.info("Loading %s data from %s training scenes.", self.split, len(target_files))

        if progress_bar and len(files) > 0:
            progress = Progress(console=satrain.logging.get_console())
        else:
            progress = None

        with progress_bar_or_not(progress_bar=progress_bar) as progress:
            if progress is not None:
                bar = progress.add_task(
                    f"Loading training samples:", total=len(target_files)
                )
            else:
                bar = None

            for ind, target_file in enumerate(target_files):
                target_data = xr.load_dataset(target_file, engine="h5netcdf")
                valid = ~self.target_config.get_mask(target_data)
                valid = xr.DataArray(
                    data=valid,
                    dims=target_data.surface_precip.dims
                )
                target_data = extract_samples(target_data, valid)
                if "time" in target_data.coords:
                    target_data = target_data.reset_index("time")

                for var in target_data:
                    if target_data[var].dtype == np.float64:
                        target_data[var] = target_data[var].astype(np.float32)

                self.target_data.append(target_data)
                del target_data

                ref_time = get_median_time(target_file)

                for inpt in self.retrieval_input:
                    input_time = get_median_time(files[inpt.name][ind])
                    if ref_time != input_time:
                        raise ValueError(
                            "Encountered an input files %s that is inconsistent with the corresponding "
                            "reference file %s. This indicates that the dataset has not been downloaded "
                            "properly."
                        )
                    input_data = extract_samples(xr.load_dataset(files[inpt.name][ind], engine="h5netcdf"), valid)
                    for var in input_data:
                        if input_data[var].dtype == np.float64:
                            input_data[var] = input_data[var].astype(np.float32)

                    if "scan_time" in input_data:
                        input_data = input_data.drop_vars(("scan_time"))
                    if "time" in input_data.coords:
                        input_data = input_data.reset_index("time")
                    getattr(self, inpt.name + "_data").append(input_data)

                    del input_data

                if progress is not None:
                    progress.advance(bar, advance=1)

        self.target_data = xr.concat(self.target_data, dim="samples")
        for inpt in self.retrieval_input:
            input_data = xr.concat(getattr(self, inpt.name + "_data"), dim="samples")
            setattr(self, inpt.name + "_data", input_data)


    def __len__(self) -> int:
        """
        The number of samples in the dataset.
        """
        n_samples = self.target_data.samples.size
        if self.subsample is not None:
            n_samples = self.subsample * n_samples

        if self.batch_size is None:
            return self.target_data.samples.size

        n_batches = ceil(n_samples / self.batch_size)
        return n_batches

    def __getitem__(
        self, ind: int
    ) -> Tuple[Union[torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor]:
        """
        Return sample from dataset.

        Args:
                ind: The index identifying the sample.

        Return:
            A tuple ``input, target`` containing a the retrieval input data in ``input`` and
            the target data in ``target``. If ``stack`` is 'True', ``input`` is a tensor containing
            all input data, otherwise ``input`` is dictionary mapping the separate input names
            to separate tensors and it is up to the user to combine them.
        """
        if ind >= len(self):
            raise IndexError("Dataset is exhausted.")

        if ind == 0:
            if self.shuffle:
                self.indices = self.rng.permutation(self.target_data.samples.size)
            else:
                self.indices = np.arange(self.target_data.samples.size)

        if self.batch_size is None:
            samples = self.indices[ind]
        else:
            batch_start = ind * self.batch_size
            batch_end = batch_start + self.batch_size
            samples = self.indices[batch_start:batch_end]

        target_data = self.target_data[{"samples": samples}]
        surface_precip = self.target_config.load_reference_precip(target_data).astype(
            np.float32
        )
        precip_mask = self.target_config.load_precip_mask(target_data).astype(np.float32)
        heavy_precip_mask = self.target_config.load_heavy_precip_mask(target_data).astype(np.float32)
        target = {
            "surface_precip": torch.tensor(surface_precip),
            "precip_mask": torch.tensor(precip_mask),
            "heavy_precip_mask": torch.tensor(heavy_precip_mask),

        }

        target_time = target_data.time

        input_data = {}

        for inpt in self.retrieval_input:
            data = getattr(self, inpt.name + "_data", None)
            if data is None:
                continue
            data = inpt.load_data(data[{"samples": samples}], target_time=target_time)
            for key, arr in data.items():
                if self.batch_size is not None:
                    arr = arr.reshape(-1, arr.shape[-1]).transpose().copy()
                else:
                    arr = arr.ravel()
                input_data[key] = torch.tensor(arr.astype(np.float32))

        if self.stack:
            input_data = torch.cat(list(input_data.values()), -1)

        return input_data, target


def apply(tensors: Any, transform: torch.Tensor) -> torch.Tensor:
    """
    Apply transformation to any container containing torch.Tensors.

    Args:
        tensors: An arbitrarily nested list, dict, or tuple containing
            torch.Tensors.
        transform:

    Return:
        The same containiner but with the given transformation function applied to
        all tensors.
    """
    if isinstance(tensors, tuple):
        return tuple([apply(tensor, transform) for tensor in tensors])
    if isinstance(tensors, list):
        return [apply(tensor, transform) for tensors in tensors]
    if isinstance(tensors, dict):
        return {key: apply(tensor, transform) for key, tensor in tensors.items()}
    if isinstance(tensors, torch.Tensor):
        return transform(tensors)
    raise ValueError("Encountered an unsupported type %s in apply.", type(tensors))


class SatRainSpatial:
    """
    Dataset class providing access to the spatial variant of the satellite precipitation retrieval
    benchmark dataset.
    """

    def __init__(
        self,
        base_sensor: str,
        geometry: str,
        split: str,
        subset: str = "xl",
        retrieval_input: List[str | dict[str | Any] | InputConfig] = None,
        target_config: TargetConfig = None,
        stack: bool = False,
        augment: bool = True,
        data_path: Optional[Path] = None,
        download: bool = True,
    ):
        """
        Args:
            base_sensor: The base_sensor for which to load the benchmark dataset.
            geometry: Whether to load on_swath or regridded observations.
            split: Whether to load 'training', 'validation', or
                 'testing' splits.
            retrieval_input: List of the retrieval inputs to load. The list should contain
                names of retrieval input sources ("pmw", "geo", "geo_ir", "ancillary"), dictionaries
                defining the input name and additional input options, or InputConfig. If not explicitly
                specified all available input data is loaded.
            target_config: An optional TargetConfig specifying quality requirements for the retrieval
                target data to load.
            stack: If 'False', the input will be loaded as a dictionary containing the input tensors
                from all input dataset. If 'True', the tensors will be concatenated along the feature axis
                and only a single tensor is loaded instead of dictionary.
            augment: If 'True' will apply random horizontal and vertical flips to the input data.
            satrain_path: Path containing or to which to download the SatRain data.
            download: If 'True', missing data will be downloaded upon dataset creation. Otherwise, only
                locally available files will be used.
        """
        super().__init__()

        if data_path is None:
            data_path = config.get_data_path()
        else:
            data_path = Path(data_path)

        if not base_sensor.lower() in ["gmi", "atms"]:
            raise ValueError("Base_Sensor must be one of ['gmi', 'atms'].")
        self.base_sensor = base_sensor.lower()

        if not geometry.lower() in ["gridded", "on_swath"]:
            raise ValueError("Geomtry must be one of ['gridded', 'on_swath'].")
        self.geometry = geometry.lower()

        if not split.lower() in ["training", "validation", "testing"]:
            raise ValueError(
                "Split must be one of ['training', 'validation', 'testing']"
            )
        self.split = split
        self.subset = subset

        if retrieval_input is None:
            retrieval_input = ALL_INPUTS
        self._retrieval_input = retrieval_input
        retrieval_input = parse_retrieval_inputs(retrieval_input)

        if target_config is None:
            target_config = TargetConfig()
        elif isinstance(target_config, dict):
            target_config = TargetConfig(**target_config)
        self.target_config = target_config

        self.stack = stack
        self.augment = augment

        self.pmw = None
        self.geo = None
        self.geo_ir = None
        self.ancillary = None
        self.target = None
        self.data_path = data_path

        dataset = f"satrain/{self.base_sensor}/{self.split}/{self.geometry}/spatial/"

        if download:
            sources = set([inpt.name for inpt in retrieval_input] + ["target"])
            for source in sources:
                download_missing(
                    dataset_name="satrain",
                    base_sensor=self.base_sensor,
                    geometry=self.geometry,
                    source=source,
                    split=self.split,
                    subset=self.subset,
                    progress_bar=True,
                    destination=data_path
                )
        files = get_local_files(
            dataset_name="satrain",
            base_sensor=self.base_sensor,
            geometry=self.geometry,
            split=self.split,
            subset=self.subset,
            data_path=data_path,
        )
        if len(files["target"]) == 0:
            raise ValueError(
                f"Couldn't find any target data files. "
                " Please make sure that the satrain data path is correct or "
                "set 'download' to True to download the file."
            )

        for source, source_files in files.items():
            setattr(self, source, np.array([str(path) for path in source_files]))

        self.check_consistency()
        self.worker_init_fn(0)

        self.fill_values = None
        if self.augment:
            self.fill_values = []
            for inpt in self.retrieval_input:
                fill_value = getattr(inpt, "nan", None)
                if fill_value is None:
                    raise ValueError(
                        "For data augmentation all retrieval inputs must have the 'nan' attribute set."
                    )
                self.fill_values.append(fill_value)


    def worker_init_fn(self, w_id: int) -> None:
        """
        Seeds the dataset loader's random number generator.
        """
        seed = int.from_bytes(os.urandom(4), "big") + w_id
        self.rng = np.random.default_rng(seed)

    def check_consistency(self):
        """
        Check consistency of training files.

        Raises:
            RuntimeError when the training scenes for any of the inputs is inconsistent with those
            available for the target.
        """
        target_times = set(map(get_median_time, self.target))
        for inpt in parse_retrieval_inputs(self._retrieval_input):
            inpt_times = set(map(get_median_time, getattr(self, inpt.name)))
            if target_times != inpt_times:
                raise RuntimeError(
                    f"Available target times are inconsistent with input files for input {inpt}."
                )

    @cached_property
    def retrieval_input(self):
        return parse_retrieval_inputs(self._retrieval_input)

    @cache
    def get_source_files(self, source: str) -> np.ndarray:
        """
        Get list of source files.

        """
        files = get_local_files(
            dataset_name="satrain",
            base_sensor=self.base_sensor,
            geometry=self.geometry,
            split=self.split,
            subset=self.subset,
            data_path=self.data_path,
        )
        return files[source]

    @cache
    def get_target_files(self) -> np.ndarray:
        """
        Get list of target files.
        """
        files = get_local_files(
            dataset_name="satrain",
            base_sensor=self.base_sensor,
            geometry=self.geometry,
            split=self.split,
            subset=self.subset,
            data_path=self.data_path,
        )
        return files["target"]


    @cache
    def __len__(self) -> int:
        """
        The number of samples in the dataset.
        """
        return len(self.get_target_files())

    def __getitem__(self, ind: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Load sample from dataset.
        """
        if self.augment:
            degrees = self.rng.uniform(-180, 180)
            scale = self.rng.uniform(0.8, 1.2)
            shear = self.rng.uniform(-30.0, 30.0)
            kwargs = {"angle": degrees, "scale": scale, "shear": shear, "translate": [0.0, 0.0]}
            input_transforms = [
                lambda x: v2.functional.affine(x, **kwargs, fill=fill)
                for inpt, fill in zip(self.retrieval_input, self.fill_values)
            ]
            target_transform = lambda tensor: v2.functional.affine(tensor[None], **kwargs, fill=torch.nan)[0]
        else:
            input_transforms = [lambda x: x for _ in self.retrieval_input]
            target_transform = lambda x: x

        with xr.open_dataset(self.get_target_files()[ind], engine="h5netcdf", chunks=None, cache=False) as data:
            target_time = data.time.data.copy()
            surface_precip = self.target_config.load_reference_precip(data).copy()
            precip_mask = self.target_config.load_precip_mask(data).copy()
            heavy_precip_mask = self.target_config.load_heavy_precip_mask(data).copy()
            target = {
                "surface_precip": target_transform(torch.tensor(surface_precip.astype(np.float32))),
                "precip_mask": target_transform(torch.tensor(precip_mask.astype(np.float32))),
                "heavy_precip_mask": target_transform(torch.tensor(heavy_precip_mask.astype(np.float32))),
            }
        data.close()
        del data

        input_data = {}
        for inpt, transform in zip(self.retrieval_input, input_transforms):
            files = self.get_source_files(inpt.name)
            if files is None:
                continue
            data = inpt.load_data(
                files[ind],
                target_time=target_time,
            )
            for name, arr in data.items():
                input_data[name] = transform(torch.tensor(arr.astype(np.float32).copy()))

            del files
            del data

        if self.stack:
            input_data = torch.cat(list(input_data.values()), axis=0)

        del target_time

        return input_data, target
