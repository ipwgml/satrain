"""
satrain
=======

The satrain Python package provides access and evaluation functionality for the
Satellite-Based Estimation and Detection of Rain (SatRain) benchmark dataset
developed by the machine-learning working group of the International
Precipitation Working Group (IPWG).
"""
import hdf5plugin

from .data import get_files

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("satrain")
except PackageNotFoundError:
    __version__ = "0.1.dev0+unknown"
