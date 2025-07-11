# The Satellite-based Rain Estimation and Detection (SatRain) benchmark dataset created by the International Precipitation Working Group

The ipwgml Python package provides tools for accessing and using the Satellite-Based Rain Estimation and Detection (SatRain) benchmark dataset developed by the Machine Learning Working Group of the International Precipitation Working Group (IPWG). The dataset provides a machine-learning-ready benchmark for assessing  machine-learning-based  satellite precipitation estimation and detection algorithms.

## The SatRain benchmark dataset

The SatRain benchmark dataset combines satellite imagery from multiple platforms and sensors with gauge-corrected ground-based radar precipitation measurements. It serves as a comprehensive resource for developing and benchmarking precipitation retrieval algorithms using machine learning techniques. The dataset supports various retrieval scenarios by offering multi-sensor collocated observations, making it suitable not only for the evaluation of existing ML techniques but also the development of advanced ML methods for multi-sensor and temporal fusion.


```{figure} /figures/example_scene.png
---
height: 400px
name: example_scene
---
Retrieval input and target data of the SatRain benchmark dataset. Panels (a), (b), and (c) show selected, collocated observations from the passive microwave (PMW) observations (Panel (a)) and geostationary visible (Panel (b)) and infrared observations that make up the input data of the SatRain data. Panel (d) shows the precipitation-radar-based precipitation estimates that are the retrieval targets. Grey dashed, and dash-dotted lines mark the outlines of the training samples extracted from this collocation scene for the gridded and on-swath observations. Black dashed and dash-dotted lines mark the sample training scenes displayed in Panel (e) and (f).

```

### Features

The principal features provided by the SatRain dataset and the ``ipwgml`` package are:


1. **ML-ready dataset****: The SatRain dataset provides collocated two- and three-dimensional observation tensors and corresponding reference precipitation estimates that can be used to train various machine-learning models. Additionally, the ``ipwgml`` package provides PyTorch dataloaders for loading the data in image and tabular format as well as cloud-enabled example notebooks demonstrating the training and evaluations of fully-connected and convolutional neural network models. 

2. **Flexible evaluation framework:** The ``ipwgml`` package provides functionality to evaluate any precipitation retrieval against the SatRain test data, thus supporting fast development-evaluation cycles and allowing direct comparison of ML-based and conventional retrievals.

3. **Multi-sensor and multi-time-step observations:** The SatRain input data comprises observations from multiple satellite sensors, time-resolved observations from geostationary platforms, as well ancillary data from a reanlysis dataset. The dataset thus provides a comprehensive base for exploring sensor synergies, temporal fusion, and the benefits of ancillary data.
