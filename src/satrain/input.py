"""
ipwgml.input
============

The ``ipwgml.input`` module provides configuration classes to represent and
configure the retrieval input data for the SPR dataset. The currently supported
input datasets are GMI observations, ATMS observations, ancillary data,
geostationary observations, and geostationary IR observations. These retrieval
inputs are represented by the classes :class:`Ancillary`, :class:`GMI`,
``ATMS``, :class:`Geo`, :class:`GeoIR`. Each of these classes provides
configuration options for the data that is actually loaded.

Usage
-----

The above-named input config classes can be used everywhere that retrieval input
data is specified, most notably in the :class:`ipwgml.evaluation.Evaluator`
and the dataset classes.

Alternatively, the retrieval input datasets can be specified using strings. In
this case, the strings ``gmi``, ``atms``, ``ancillary`` ``geo``, ``geo_ir`` map
to the classes :class:`Ancillary`, :class:`GMI`, ``ATMS``, :class:`Geo`,
:class:`GeoIR`, which will be instantiated using their default configuration.

Finally, it is also possible to specify an input using a dictionary. The
dictionary must in this case have an entry ``name``, which should correspond the
string representation of the input. The remaining key-value pairs in the
dictionary will then be passed to the constructor of the corresponding input
config class.

As an example, the following three ways of repsenting GMI retrieval input are
equivalent:

.. code-block:: Python

  retrieval_input = GMI(channels=None, include_angles=True, normalize=None, nan=None)
  retrieval_input = ["gmi"]
  retrieval_input = {
      "name": "gmi",
      "channels": None,
      "include_angles": True,
      "normalize": None,
      "nan": None
  }

Members
-------

"""
from abc import ABC, abstractproperty
from copy import copy
from functools import cached_property
import gc
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Union

import hdf5plugin
import numpy as np
import xarray as xr

from satrain.definitions import ANCILLARY_VARIABLES
from satrain.utils import open_if_required


def normalize(
        data: np.ndarray,
        stats: xr.Dataset,
        how: Optional[str] = None,
        nan: Optional[float] = None
) -> np.ndarray:
    """
    Normalize input data and replace missing values.

    Args:
        data: An numpy.ndarray containing the data to normalize.
        stats: An xarray.Dataset containing the summary statistics of the data.
        how: A string specifying how to normalize the data. Should be one of
            ['standardize', 'minmax']>
        nan: If given, use this value to replace NAN values in the input.

    Return:
        The give array 'data' normalized according to the given statistics and
        chosen normalization method and, if 'nan' is not None, with NAN values
        replaced with 'nan'.
    """
    if how is not None:
        pad_dims = data.ndim - 1
        if how.lower() == "standardize":
            mu = stats["mean"].data.__getitem__((...,) + (None,) * pad_dims)
            sigma = stats["std_dev"].data.__getitem__((...,) + (None,) * pad_dims)
            data = (data - mu) / (sigma + 1e-6)
        elif how.lower() == "minmax":
            x_max = stats["max"].data.__getitem__((...,) + (None,) * pad_dims)
            x_min = stats["min"].data.__getitem__((...,) + (None,) * pad_dims)
            data = 2.0 * (data - x_min) / (x_max - x_min + 1e-6) - 1.0
        else:
            raise ValueError(
                f"The normalization strategy '{how}' is not supported. Supported strategies are "
                "'standardize' and 'minmax'."
            )

    if nan is not None:
        data = np.nan_to_num(data, nan=nan, copy=True)
    return data



class InputConfig(ABC):
    """
    Base class for input data records used to define what input data to load.
    """
    @classmethod
    def parse(self, inpt: Union[str, Dict[str, Any], "InputConfig"]) -> "InputConfig":
        """
        Parse InputConfig object from an argument that can be either a string, a dictionary
        or an InputConfig object.

        If 'inpt' is a string, this method will simply instantiate the InputConfig sub-class
        of the corresponding name, which will be instantiated with the default settings.
        If 'inpt' is a dictionary, it must have field 'name' specifying the name of the
        InputConfig sub-class to instantiate. All other keys in the dictionary will be
        passed to the constructor call of this class. Finally, if 'inpt' is allready
        an InputConfig sub-class object, it is returned as-is.

        Args:
            inpt: The inpt to parse as a InputConfig object.

        Return:
            An object of an InputConfig sub-class.
        """
        if isinstance(inpt, InputConfig):
            return inpt
        elif isinstance(inpt, str):
            name = inpt
            kwargs = {}
        elif isinstance(inpt, dict):
            inpt = copy(inpt)
            name = inpt.pop("name", None)
            if name is None:
                raise ValueError(
                    "If a retrieval input is specified using a dict, it must have an entry "
                    "'name'."
                )
            kwargs = inpt
        else:
            raise ValueError(
                f"Unsupported input for parsing an InputConfig: {inpt}"
            )

        if name.lower() == "gmi":
            return GMI(**kwargs)
        if name.lower() == "atms":
            return ATMS(**kwargs)
        elif name.lower() == "ancillary":
            return Ancillary(**kwargs)
        elif name.lower() == "geo":
            return Geo(**kwargs)
        elif name.lower() == "seviri":
            return Seviri(**kwargs)
        elif name.lower() == "geo_t":
            return GeoT(**kwargs)
        elif name.lower() == "geo_ir":
            return GeoIR(**kwargs)
        elif name.lower() == "geo_ir_t":
            return GeoIRT(**kwargs)
        raise RuntimeError(
            f"Provided retrieval input name '{name}' is not known."
        )

    @abstractproperty
    def name(self) -> str:
        """
        String representation of the input.
        """

    @abstractproperty
    def stats(self) -> xr.Dataset:
        """
        xarray.Dataset containing summary statistics for the input.
        """

    def to_dict(self) -> Dict[str, Any]:
        """
        .toml compatible dictionary representation of input config.
        """
        dct = asdict(self)
        dct["name"] = self.name
        return {
            name: val for name, val in dct.items() if val is not None
        }

    def __hash__(self):
        """
        Use class name as hash to allow building dictionaries with InputConfigs.
        """
        return hash(self.__class__.__name__)


@dataclass
class PMW(InputConfig):
    """
    InputData record class representing passive-microwave (PMW) observations.
    """
    def __init__(
            self,
            channels: Optional[List[int]] = None,
            include_angles: bool = True,
            normalize: Optional[str] = None,
            nan: Optional[str] = None
    ):
        """
        Args:
            channels: An optional list of zero-based indices identifying channels to
                load. If 'None', all channels will be loaded.
            include_angles: Wether or not to include the eart-incidence angles of the
                observations in the input.
            normalize: An optional string specifying how to normalize the input data.
            nan: An optional float value that will be used to replace missing values
                in the input data.
        """
        self.channels = None if channels is None else np.array(channels)
        self.include_angles = include_angles
        self.normalize = normalize
        self.nan = nan
        self._obs_stats = None
        self._ang_stats = None

    @cached_property
    def stats(self) -> xr.Dataset:
        """
        xarray.Dataset containing summary statistics for the input.
        """
        stats_file = Path(__file__).parent / "files" / "stats" / f"obs_{self.name}.nc"
        stats = xr.load_dataset(stats_file, engine="h5netcdf")
        if self.channels is not None:
                stats = stats[{"features": self.channels}]
        return stats

    @property
    def ang_stats(self) -> xr.Dataset:
        """
        xarray.Dataset containing summary statistics for the viewing angles.
        """
        if self._ang_stats is None:
            stats_file = Path(__file__).parent / "files" / "stats" / f"eia_{self.name}.nc"
            self._ang_stats = xr.load_dataset(stats_file, engine="h5netcdf")
            if self.channels is not None:
                self._ang_stats = self._ang_stats[{"features": self.channels}]
        return self._ang_stats

    def load_data(self, pmw_data_file: Path, target_time: xr.DataArray) -> Dict[str, np.ndarray]:
        """
        Load PMW observations from NetCDF file.

        Args:
            pmw_data_file: A Path object pointing to the file from which to load the input data.
            target_time: Not used.

        Return:
            A dictionary mapping the keys 'obs_<sensor_name>' the loaded PMW observations. If 'include_angles'
            is 'True' the dictionary will also containg the earth-incidence angles with the
            key 'eia_<sensor_name>'.
        """
        with open_if_required(pmw_data_file) as pmw_data:
            pmw_data = pmw_data[["observations", "earth_incidence_angle"]].compute().transpose("channel", ...)
            obs = pmw_data["observations"].compute().transpose("channel", ...)
            if self.channels is not None:
                obs = obs[{"channel": self.channels}]
            else:
                obs = obs[{"channel": slice(0, None)}]

            obs = obs.data
            obs = normalize(obs, self.stats, how=self.normalize, nan=self.nan)

            inpt_data = {
                f"obs_{self.name}": obs
            }
            if self.include_angles:
                angs = pmw_data["earth_incidence_angle"].compute().transpose("channel", ...)
                if self.channels is not None:
                    angs = angs[{"channel": self.channels}]
                else:
                    angs = angs[{"channel": slice(0, None)}]
                angs = normalize(angs.data, self.ang_stats, how=self.normalize, nan=self.nan)
                inpt_data[f"eia_{self.name}"] = angs

        return inpt_data


@dataclass
class ATMS(PMW):
    """
    Retrieval input data from the Advanced Technology Microwave Sounder (ATMS).

    The ATMS class represents observations from the Advanced Technology Microwave Sounder as
    retrieval input data. It allows for selecting subsets of the available ATMS
    channels and including or excluding the earth-incidence angles in the input data.

    The ATMS input will load tensors 'obs_atms' containing the ATMS passive microwave
    observations and, if 'include_angles' is set to 'True', 'eia_atms' containing the
    earth incidence angles corresponding to the observations in 'obs_atms'.
    """
    def __init__(
            self,
            channels: Optional[List[int]] = None,
            include_angles: bool = True,
            normalize: Optional[str] = None,
            nan: Optional[str] = None
    ):
        """
        Args:
            channels: An optional list of zero-based indices identifying channels to
                load. If 'None', all channels will be loaded.
            include_angles: Wether or not to include the earth-incidence angles of the
                observations in the input.
            normalize: An optional string specifying how to normalize the input data.
            nan: An optional float value that will be used to replace missing values
                in the input data.
        """
        self.channels = channels
        self.include_angles = include_angles
        self.normalize = normalize
        self.nan = nan
        self._obs_stats = None
        self._ang_stats = None

    @property
    def name(self) -> str:
        return "atms"

    @property
    def features(self) -> Dict[str, int]:
        """
        Dictionary mapping the input names from the ATMS input to the corresponding
        number of channels.
        """
        n_chans = 9
        if self.channels is not None:
            n_chans = len(self.channels)

        features = {"obs_atms": n_chans}
        if self.include_angles:
            features["eia_atms"] = n_chans

        return features


@dataclass
class GMI(PMW):
    """
    Retrieval input data from the GPM Microwave Imager (GMI).

    The GMI class represents observations from the GPM Microwave Imager (GMI) as
    retrieval input data. It allows for selecting subsets of the available GMI
    channels and including or excluding the earth-incidence angles in the input data.

    The GMI input will load tensors 'obs_gmi' containing the GMI passive microwave
    observations and, if 'include_angles' is set to 'True', 'eia_gmi' containing the
    earth incidence angles corresponding to the observations in 'obs_gmi'.
    """
    def __init__(
            self,
            channels: Optional[List[int]] = None,
            include_angles: bool = True,
            normalize: Optional[str] = None,
            nan: Optional[str] = None
    ):
        """
        Args:
            channels: An optional list of zero-based indices identifying channels to
                load. If 'None', all channels will be loaded.
            include_angles: Wether or not to include the earth-incidence angles of the
                observations in the input.
            normalize: An optional string specifying how to normalize the input data.
            nan: An optional float value that will be used to replace missing values
                in the input data.
        """
        self.channels = channels
        self.include_angles = include_angles
        self.normalize = normalize
        self.nan = nan
        self._obs_stats = None
        self._ang_stats = None

    @property
    def name(self) -> str:
        return "gmi"

    @property
    def features(self) -> Dict[str, int]:
        """
        Dictionary mapping the input names from the GMI input to the corresponding
        number of channels.
        """
        n_chans = 13
        if self.channels is not None:
            n_chans = len(self.channels)

        features = {"obs_gmi": n_chans}
        if self.include_angles:
            features["eia_gmi"] = n_chans

        return features




@dataclass
class Ancillary(InputConfig):
    """
    This InputConfig class will load ancillary data as retrieval input. The class
    allows for configuration, which variables will be loaded.

    Including the 'Ancillary' input config in the list of retrieval inputs will
    load the ancillary data and include it in the retrieval input data as a
    variable named 'ancillary'.
    """
    def __init__(
            self,
            variables: Optional[List[str]] = None,
            normalize: Optional[str] = None,
            nan: Optional[float] = None
    ):
        """
        Args:
            variable: A list of strings specifying the ancillary data to load.
            normalize: An optional string specifying how to normalize the input data.
            nan: An optional float value that will be used to replace missing values
                in the input data.
        """
        if variables is None:
            variables = ANCILLARY_VARIABLES
        invalid = [var for var in variables if var not in ANCILLARY_VARIABLES]
        if len(invalid) > 0:
            raise ValueError(
                "'variables' must contain a subset of the available ancillary variables but "
                f"'{invalid}' are not."
            )
        self.variables = variables
        self.normalize = normalize
        self.nan = nan

    @property
    def name(self) -> str:
        return "ancillary"

    @cached_property
    def stats(self) -> xr.Dataset:
        """
        xarray.Dataset containing summary statistics for the input.
        """
        stats_file = Path(__file__).parent / "files" / "stats" / "ancillary.nc"
        inds = [ind for ind, var in enumerate(ANCILLARY_VARIABLES) if var in self.variables]
        stats = xr.load_dataset(stats_file, engine="h5netcdf")[{"features": inds}]
        return stats

    def load_data(self, ancillary_data_file: Path, target_time: xr.DataArray) -> xr.Dataset:
        """
        Load ancillary data from NetCDF file.

        Args:
            ancillary_data_file: A Path object pointing to the file from which to load the input data.
            targete_time: Not used.

        Return:
            A dicitonary mapping the single key 'ancillary' to an array containing the data from
            all ancillary variables stacked along the first axis.
        """
        with open_if_required(ancillary_data_file) as ancillary_data:
            data = []
            for var in self.variables:
                data.append(ancillary_data[var].data)

        data = np.stack(data)
        data = normalize(np.stack(data), self.stats, how=self.normalize, nan=self.nan)
        return {"ancillary": data}

    @property
    def features(self) -> Dict[str, int]:
        """
        Dictionary mapping names of the input data variables loaded by the
        ancillary data input class to the corresponding number of features.
        """
        return {"ancillary": len(self.variables)}


@dataclass
class GeoIR(InputConfig):
    """
    The GeoIR loads input data from IR-window channel observations interpolated in time to
    be closest to the nominal time of the precipitation estimates.
    """
    def __init__(
            self,
            normalize: Optional[str] = None,
            nan: Optional[float] = None
    ):
        """
        Args:
            normalize: An optional string specifying how to normalize the input data.
            nan: An optional float value that will be used to replace missing values
                in the input data.
        """
        self.normalize = normalize
        self.nan = nan

    @property
    def name(self) -> str:
        return "geo_ir"

    @cached_property
    def stats(self) -> xr.Dataset:
        """
        xarray.Dataset containing summary statistics for the input.
        """
        stats_file = Path(__file__).parent / "files" / "stats" / "obs_geo_ir.nc"
        stats = xr.load_dataset(stats_file, engine="h5netcdf")[{"features": 8}]
        return stats

    def load_data(self, geo_data_file: Path, target_time: xr.DataArray) -> xr.Dataset:
        """
        Load GEO IR data from NetCDF file.

        Args:
            geo_data_file: A Path object pointing to the file from which to load the input data.
            target_time: An xarray.DataArray containing the target times, which will be used to
                to interpolate the input observations to the nearest time step if 'self.nearest'
                is 'True'.

        Return:
            A dicitonary mapping the single key 'obs_geo' to an array containing the GEO IR
            observation from the desired time steps.
        """
        with open_if_required(geo_data_file) as geo_data:
            obs = geo_data.observations.data[None]

        obs = normalize(obs, self.stats, how=self.normalize, nan=self.nan)
        return {"obs_geo_ir": obs}

    @property
    def features(self) -> Dict[str, int]:
        """
        Dictionary mapping names of the input data variables loaded by the
        GeoIR input class to the corresponding number of features.
        """
        n_features = 1
        return {"obs_geo_ir": n_features}

@dataclass
class GeoIRT(InputConfig):
    """
    The GeoIR class represents IR-window channel observations from geostationary
    satellites in the retrieval input. The full IR input comprises 8
    half-hourly observations before the median overpass time and 8 after the
    median overpass time. The GeoIR class allows selecting subsets of these time
    steps as well as only loading the nearest observations for every reference
    data pixel.
    """
    time_steps: List[int]

    def __init__(
            self,
            time_steps: Optional[List[int]] = None,
            normalize: Optional[str] = None,
            nan: Optional[float] = None
    ):
        """
        Args:
            time_steps: Optional list of time steps to load. The time steps are identified
                using zero-based indices with steps 0-7 to the eight time steps prior
                to the median overpass time and steps 8-15 to the eight time steps after
                the overpass time.
            normalize: An optional string specifying how to normalize the input data.
            nan: An optional float value that will be used to replace missing values
                in the input data.
        """
        if time_steps is None:
            time_steps = list(range(16))
        for time_step in time_steps:
            if (time_step < 0) or (15 < time_step):
                raise RuntimeError(
                    "Time steps for GeoIR input must be within [0, 15]."
                )
        self.time_steps = time_steps
        self.normalize = normalize
        self.nan = nan

    @property
    def name(self) -> str:
        return "geo_ir_t"

    @cached_property
    def stats(self) -> xr.Dataset:
        """
        xarray.Dataset containing summary statistics for the input.
        """
        stats_file = Path(__file__).parent / "files" / "stats" / "obs_geo_ir.nc"
        stats = xr.load_dataset(stats_file, engine="h5netcdf")[{"features": self.time_steps}]
        return stats

    def load_data(self, geo_data_file: Path, target_time: xr.DataArray) -> xr.Dataset:
        """
        Load GEO IR data from NetCDF file.

        Args:
            geo_data_file: A Path object pointing to the file from which to load the input data.
            target_time: An xarray.DataArray containing the target times, which will be used to
                to interpolate the input observations to the nearest time step if 'self.nearest'
                is 'True'.

        Return:
            A dicitonary mapping the single key 'obs_geo' to an array containing the GEO IR
            observation from the desired time steps.
        """
        with open_if_required(geo_data_file) as geo_data:
            geo_data = geo_data.transpose("time", ...)
            obs = geo_data.observations[{"time": self.time_steps}].data

        obs = normalize(obs, self.stats, how=self.normalize, nan=self.nan)
        return {"obs_geo_ir": obs}

    @property
    def features(self) -> Dict[str, int]:
        """
        Dictionary mapping names of the input data variables loaded by the
        GeoIR input class to the corresponding number of features.
        """
        n_features = len(self.time_steps)
        return {"obs_geo_ir": n_features}


@dataclass
class GeoT(InputConfig):
    """
    The Geo class represents GOES-16 ABI  observations in the retrieval input.
    The full IR input comprises 2 10-minute observations before the median
    overpass time and 2 after the median overpass time.
    """
    def __init__(
            self,
            channels: Optional[List[int]] = None,
            time_steps: Optional[List[int]] = None,
            normalize: Optional[str] = None,
            nan: Optional[float] = None
    ):
        """
        Args:
            channels: Optional list of zero-based indices identifying the GOES channels
                to load.
            time_steps: Optional zero-based indices of the time steps to load. Indices
                0 and 1 correspond to 30 and 15 minutes before the median overpass time
                and indices 2 and 3 to 15 minutes after the median overpass time.
            normalize: An optional string specifying how to normalize the input data.
            nan: An optional float value that will be used to replace missing values
                in the input data.
        """
        if channels is None:
            channels = range(16)
        self.channels = channels

        if time_steps is None:
            time_steps = list(range(7))
        for time_step in time_steps:
            if (time_step < 0) or (6 < time_step):
                raise RuntimeError(
                    "Time steps for Geo input must be within [0, 6]."
                )
        self.time_steps = time_steps
        self.normalize = normalize
        self.nan = nan

    @property
    def name(self) -> str:
        return "geo_t"

    @cached_property
    def stats(self) -> xr.Dataset:
        """
        xarray.Dataset containing summary statistics for the input.
        """
        stats_file = Path(__file__).parent / "files" / "stats" / "obs_geo.nc"
        stats = xr.load_dataset(stats_file, engine="h5netcdf")
        mask = np.zeros((4, 16), dtype=bool)
        for time_ind in self.time_steps:
            mask[time_ind, self.channels] = True
        mask = mask.ravel()
        stats = stats[{"features": mask}]
        return stats

    def load_data(self, geo_data_file: Path, target_time: xr.DataArray) -> xr.Dataset:
        """
        Load GEO data from NetCDF file.

        Args:
            geo_data_file: A Path object pointing to the file from which to load the input data.
            target_time: An xarray.DataArray containing the target times, which will be used to
                to interpolate the input observations to the nearest time step if 'self.nearest'
                is 'True'.

        Return:
            A dicitonary mapping the single key 'obs_geo' to an array containing the GEO
            observation from the desired time steps. The returned array will have the
            time and channel dimensions along the leading axes of the array.
        """
        with open_if_required(geo_data_file) as geo_data:
            geo_data = geo_data.compute()
            geo_data = geo_data.transpose("time", "channel", ...)[{"channel": self.channels}]
            obs = geo_data.observations[{"time": self.time_steps}].data
            obs = np.reshape(obs, (-1,) + obs.shape[2:])

            if self.normalize is not None:
                obs = normalize(obs, self.stats, how=self.normalize, nan=self.nan)

        del geo_data

        return {"obs_geo": obs}

    @property
    def features(self) -> Dict[str, int]:
        """
        Dictionary mapping names of the input data variables loaded by the
        Geo input class to the corresponding number of features.
        """
        n_chans = len(self.channels)
        n_features = len(self.time_steps) * n_chans
        return {"obs_geo": n_features}


@dataclass
class Geo(InputConfig):
    """
    The Geo class represents GOES-16 ABI  observations in the retrieval input.
    The full IR input comprises 2 10-minute observations before the median
    overpass time and 2 after the median overpass time.
    """
    def __init__(
            self,
            channels: Optional[List[int]] = None,
            normalize: Optional[str] = None,
            nan: Optional[float] = None
    ):
        """
        Args:
            channels: Optional list of zero-based indices identifying the GOES channels
                to load.
            normalize: An optional string specifying how to normalize the input data.
            nan: An optional float value that will be used to replace missing values
                in the input data.
        """
        if channels is None:
            channels = list(range(16))
        self._channels = np.array(channels)
        self.normalize = normalize
        self.nan = nan

    @property
    def name(self) -> str:
        return "geo"

    @cached_property
    def channels(self):
        return self._channels

    @cached_property
    def stats(self) -> xr.Dataset:
        """
        xarray.Dataset containing summary statistics for the input.
        """
        stats_file = Path(__file__).parent / "files" / "stats" / "obs_geo.nc"
        stats = xr.load_dataset(stats_file, engine="h5netcdf")
        mask = np.zeros((4, 16), dtype=bool)
        mask[3, self.channels] = True
        mask = mask.ravel()
        stats = stats[{"features": mask}]
        return stats

    def load_data(self, geo_data_file: Path, target_time: xr.DataArray) -> xr.Dataset:
        """
        Load GEO data from NetCDF file.

        Args:
            geo_data_file: A Path object pointing to the file from which to load the input data.
            target_time: An xarray.DataArray containing the target times, which will be used to
                to interpolate the input observations to the nearest time step if 'self.nearest'
                is 'True'.

        Return:
            A dicitonary mapping the single key 'obs_geo' to an array containing the GEO
            observation from the desired time steps. The returned array will have the
            time and channel dimensions along the leading axes of the array.
        """
        with open_if_required(geo_data_file) as geo_data:
            obs = geo_data.observations[{"channel": self.channels}].load()
            obs = obs.transpose("channel", ...).data.copy()
        del geo_data
        obs = normalize(obs, self.stats, how=self.normalize, nan=self.nan)
        return {"obs_geo": obs.copy()}

    @property
    def features(self) -> Dict[str, int]:
        """
        Dictionary mapping names of the input data variables loaded by the
        Geo input class to the corresponding number of features.
        """
        n_chans = len(self.channels)
        return {"obs_geo": n_chans}


@dataclass
class Seviri(InputConfig):
    """
    Special instance of the Geo class load observations from the SEVIRI sensor of the 'austria' domain.
    """
    def __init__(
            self,
            channels: Optional[List[int]] = None,
            normalize: Optional[str] = None,
            nan: Optional[float] = None
    ):
        """
        Args:
            channels: Optional list of zero-based indices identifying the GOES channels
                to load.
            normalize: An optional string specifying how to normalize the input data.
            nan: An optional float value that will be used to replace missing values
                in the input data.
        """
        self.all_goes_channels = [0, 1, 2, 4, 6, 7, 9, 10, 11, 13, 14, 15]
        if channels is None:
            channels = list(range(12))
        self._channels = np.array(channels)
        self.normalize = normalize
        self.nan = nan

    @property
    def name(self) -> str:
        return "geo"

    @cached_property
    def channels(self):
        return self._channels

    @cached_property
    def goes_channels(self):
        return [self.all_goes_channels[ind] for ind in self._channels]

    @cached_property
    def stats(self) -> xr.Dataset:
        """
        xarray.Dataset containing summary statistics for the input.
        """
        stats_file = Path(__file__).parent / "files" / "stats" / "obs_geo.nc"
        stats = xr.load_dataset(stats_file, engine="h5netcdf")
        mask = np.zeros((4, 16), dtype=bool)
        mask[3, self.goes_channels] = True
        mask = mask.ravel()
        stats = stats[{"features": mask}]
        return stats

    def load_data(self, geo_data_file: Path, target_time: xr.DataArray) -> xr.Dataset:
        """
        Load GEO data from NetCDF file.

        Args:
            geo_data_file: A Path object pointing to the file from which to load the input data.
            target_time: An xarray.DataArray containing the target times, which will be used to
                to interpolate the input observations to the nearest time step if 'self.nearest'
                is 'True'.

        Return:
            A dicitonary mapping the single key 'obs_geo' to an array containing the GEO
            observation from the desired time steps. The returned array will have the
            time and channel dimensions along the leading axes of the array.
        """
        with open_if_required(geo_data_file) as geo_data:
            obs = geo_data.observations[{"channel": self.channels}].load()
            obs = obs.transpose("channel", ...).data.copy()
        del geo_data
        obs = normalize(obs, self.stats, how=self.normalize, nan=self.nan)
        return {"obs_geo": obs.copy()}

    @property
    def features(self) -> Dict[str, int]:
        """
        Dictionary mapping names of the input data variables loaded by the
        Geo input class to the corresponding number of features.
        """
        n_chans = len(self.channels)
        return {"obs_geo": n_chans}


def parse_retrieval_inputs(
        inputs: List[str | Dict[str, Any] | InputConfig]
) -> List[InputConfig]:
    """
    Parse retrieval inputs.

    Args:
        inputs: A list specifying retrieval inputs. Each element in the list can
            be a string, a dictionary defining the retrieval input configuration
            or an InputConfig.

    Return:
        A list containing the retrieval input configuration represented using
        InputConfig objects.
    """
    return [InputConfig.parse(inpt) for inpt in inputs]


def calculate_input_features(
        inputs: List[str | Dict[str, Any] | InputConfig],
        stack: bool = True
) -> int | Dict[str, int]:
    """
    Calculates the number of input features given a list of inputs.

    Args:
        inputs: A list specifying the retrieval input.
        stack: If 'True', returns a single integer representing the total number of
            features of all inputs stacked along the channel/feature dimension. If 'False',
            returns a dict mapping input names to the corresponding number of features.
    """
    inputs = parse_retrieval_inputs(inputs)
    features = {}
    for inpt in inputs:
        features.update(inpt.features)

    if stack:
        return sum(features.values())

    return features
