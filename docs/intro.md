# SatRain: A Benchmark Dataset for Machine-Learning–Based Satellite Rain Estimation and Detection

The `ipwgml` Python package provides tools for accessing and working with **SatRain: A machine-learning benchmark dataset for satellite-based estimation and detection of rain**.  
SatRain was developed by the Machine Learning Working Group of the International Precipitation Working Group (IPWG) to provide a standardized reference for training, testing, and comparing machine-learning precipitation retrieval algorithms.  

By combining high-quality satellite observations with ground-based radar reference data, SatRain offers a machine-learning–ready benchmark that supports both model development and rigorous evaluation across diverse retrieval scenarios.

## The SatRain Benchmark Dataset

SatRain integrates satellite imagery from multiple platforms and sensors with gauge-corrected, ground-based radar precipitation measurements. This combination creates a comprehensive resource for advancing precipitation retrievals with machine learning.  

The dataset enables:  
- **Algorithm development** by providing collocated, multi-sensor input data.  
- **Robust benchmarking** of existing retrieval methods in a reproducible setting.  
- **Exploration of advanced techniques** such as sensor synergy, temporal fusion, and the use of ancillary datasets.  

```{figure} /figures/example_scene.png
---
height: 400px
name: example_scene
---
Example scene from the SatRain benchmark dataset. Panels (a)-(c) show collocated observations from passive microwave (a), geostationary visible (b), and infrared (c) sensors, which together form the retrieval inputs. Panel (d) shows the radar-based precipitation estimates used as reference targets. Grey dashed and dash-dotted lines mark training sample outlines for gridded and on-swath observations. Panels (e) and (f) display extracted training samples corresponding to the highlighted regions.
```


## Features

The key features provided by SatRain and the ipwgml package are:

- **ML-ready dataset**: SatRain delivers collocated two- and three-dimensional observation tensors paired with reference precipitation estimates. The ipwgml package includes ready-to-use PyTorch dataloaders for both image and tabular formats, along with cloud-enabled example notebooks demonstrating training and evaluation of fully connected and convolutional neural networks.

- **Flexible evaluation framework:** The package provides functionality for evaluating any precipitation retrieval against SatRain’s test data, enabling rapid development–evaluation cycles and direct comparison of machine-learning and conventional retrieval algorithms.

- **Multi-sensor and temporal coverage:** SatRain incorporates observations from multiple satellite sensors, time-resolved geostationary platforms, and ancillary reanalysis data. This rich input space supports studies of sensor synergy, temporal fusion, and the added value of auxiliary datasets.
