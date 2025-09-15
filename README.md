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
 

![Precipitation estimates from three SatRain-based retrievals  applied to observations from Typhoon Khanun during landfall on
the Korean peninsula](docs/figures/retrieval_example.png)]

Retrieved precipitation from three SatRain-trained retrievals applied to
observations of Typhoon Khanun during landfall on the Korean peninsula. Each
retrieval relies on a different type of input from the SatRain dataset: a single
IR window channel (panel d), Himawari-9 observations (panel e), and GMI
observations (panel f). The results are compared with reference ground-based
radar measurements (panel a) and baseline estimates from ERA5 (panel b) and
GPROF V7 (panel c).

## Documentation

For instructions on how to get started using the dataset refer to the documentation available [here](https://satrain.readthedocs.io/en/latest/intro.html).
