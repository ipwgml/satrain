# SatRain: A machine-learning-ready benchmark dataset for satellite precipitation estimation

The Satellite Precipitation Retrieval (SatRain) benchmark dataset developed by the
International Precipitation Working Group (IPWG) is a benchmark dataset for
machine-learning-based satellite precipitation retrievals, i.e., algorithms for
detecting and quantifying precipitation from satellite imagery.

## Features

 - Training, validation, and test splits derived from four years of overpasses of passive-microwave sensors
   over the conterminous united states (CONUS)
 - Collocated satellite observations from visible, infrared, and microwave sensors
 - Comprehensive ancillary data covering atmospheric parameters and surface properties
 - Independent *testing* datasets derived from different regions and measurement techniques
 - Flexible evaluation framework
 
## Example

The figure shows retrieved precipitation from three retrievals trained using the
SatRain training applied to observations from Typhoon Khanun during landfall on
the Korean peninsula. Each of three SatRain-based retrievals use one of the
three different types of input observations available in the SatRain dataset:
single IR window channel (Panel (a)), Himawari-9 observations (Panel (b)), and
GMI observations (Panel (c)). The retrievals are compared to the reference
measurements from ground-based radar (Panel (a)), and baseline estimates from
ERA5 (Panel (b)) and GPROF V7 (Panel (c)).

![Precipitation estimates from three SatRain-based retrievals  applied to observations from Typhoon Khanun during landfall on
the Korean peninsula](docs/figures/retrieval_example.png)]


## Documentation

For instructions on how to get started using the dataset refer to the documentation available [here](https://satrain.readthedocs.io/en/latest/intro.html).
