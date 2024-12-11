# The Satellite Precipitation Retrieval (SPR) benchmark dataset created by the International Precipitation Working Group

The ipwgml Python package provides tools for accessing and using the Satellite Precipitation Retrieval (SPR) benchmark dataset developed by the Machine Learning Working Group of the International Precipitation Working Group (IPWG). This dataset is designed for evaluating machine-learning-based precipitation retrieval algorithms by providing a standardized, ML-ready dataset for testing and comparison.

## The SPR benchmark dataset

The SPR benchmark dataset combines satellite imagery from multiple platforms and sensors with gauge-corrected ground-based radar precipitation measurements. It serves as a comprehensive resource for developing and benchmarking precipitation retrieval algorithms using machine learning techniques. The dataset supports various retrieval scenarios by offering multi-sensor collocated observations, making it suitable for testing sensor fusion and advanced ML methods.


```{figure} /figures/example_scene.png
---
height: 400px
name: example_scene
---
Retrieval input and target data of the SPR benchmark dataset. Panels (a), (b), and (c) show selected, collocated observations from the passive microwave (PMW) observations (Panel (a)) and geostationary visible (Panel (b)) and infrared observations that make up the input data of the SPR data. Panel (d) shows the precipitation-radar-based precipitation estimates that are the retrieval targets. Grey dashed, and dash-dotted lines mark the outlines of the training samples extracted from this collocation scene for the gridded and on-swath observations. Black dashed and dash-dotted lines mark the sample training scenes displayed in Panel (e) and (f).

```

### Features

The principal features provided by the SPR dataset and the ``ipwgml`` package are:


1. **ML-ready dataset****: The SPR dataset combines passive microwave observations and visible and infrared observations from geostationary satellites with gauge-corrected precipitation estimates from ground-based precipitation radars. The input data of the SPR benchmark dataset also comprises various environmental variables, so-called *ancillary data*, and multiple time steps from the geostationary sensors, thus providing a comprehensive base for exploring sensor synergies, temporal fusion, and the benefits of ancillary data.
   
2. **Flexible evaluation framework:** The ``ipwgml`` package provides functionality to evaluate any precipitation retrieval against the SPR test data, thus supporting fast development-evaluation cycles and allowing direct comparison of ML-based and conventional retrievals.
