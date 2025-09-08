# SatRain: A Benchmark Dataset for Satellite-Based Estimation and Detection of Rain 

The `satrain` Python package provides tools for accessing and working with
**SatRain: A Benchmark Dataset for Satellite-Based Estimation
and Detection of Rain**. SatRain was developed by the Machine Learning Working
Group of the [International Precipitation Working
Group](https://cgms-info.org/about-cgms/international-precipitation-working-group/)
(IPWG) to provide a standardized reference for comparing machine-learning
precipitation retrieval algorithms. SatRain combines multi-sensor satellite
observations with high-quality reference data from gauge-corrected ground-based
radars providing an AI-ready benchmark dataset that supports both model
development and evaluation across diverse retrieval scenarios.


```{figure} /figures/example_scene.png
---
name: example_scene
---
Satellite observations and reference precipitation estimates of the SatRain benchmark dataset. Panels (a), (b), and (c) show selected channels from the passive microwave (PMW), visible, and infrared observations that make up the input data of the SatRain dataset. Panel (d) shows the ground-radar-based precipitation reference used as training targets. All training samples are available in gridded format on a 0.036° regular latitude-longitude grid and in on-swath format corresponding to the native sampling of the PMW observations. Black lines mark the sample training scenes displayed in Panel (e) and (f).
```

## Features

The key features of the SatRain dataset are:

- **AI-ready dataset**: SatRain delivers collocated observations  paired with reference precipitation estimates. The ``satrain`` package includes ready-to-use PyTorch dataloaders for both image and tabular formats, along with cloud-enabled example notebooks demonstrating training and evaluation of fully connected and convolutional neural networks.

- **Flexible evaluation framework:** The package provides functionality for evaluating any precipitation retrieval against SatRain’s test data, enabling rapid development-evaluation cycles and direct comparison of machine-learning and conventional retrieval algorithms.

- **Multi-sensor and temporal coverage:** SatRain incorporates observations from multiple satellite sensors, time-resolved geostationary platforms, and ancillary reanalysis data. This rich input space supports studies of sensor synergy, temporal fusion, and the added value of auxiliary datasets.

## Applications

The SatRain dataset was designed to enable a systematic approach to the
development of AI-based satellite precipitation estimation and detection
algorithms. The dataset provides a comprehensive yet flexible basis to assess
the impact of advanced ML models and tackle important challenges for advancing
satellite precipitation estimation and detection.

Example applications:

- **Model benchmarking:**  Systematic comparison of different ML-models will help the community move towards stronger baseline retrievals.
- **Sensor fusion:**  The collocated multi-sensor observations in the SatRain dataset provide an ideal starting point for developing algorithms exploiting synergies between different satellite sensors.
- **Temporal fusion:** Similarly, the availability of time-resolved geostationary observations provides a basis for developing novel AI-based algorithms that exploit this information for improved precipitation retrievals.
- **Future retrieval concepts** A variety of sensor types included in the SatRain dataset makes it suitable for exploring advanced retrieval techniques such as sensor-agnostic retrievals designed to be applied across sensor types.
