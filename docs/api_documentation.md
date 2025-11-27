# API Documentation

This section contains the API documentation for the user-facing components of
the ``satrain`` package.

## Core Modules

- [Data Access and Management](api/data) - Dataset downloading, file management, and data loading
- [Evaluation Framework](api/evaluation) - Model evaluation and benchmarking tools  
- [Metrics](api/metrics) - Statistical metrics for precipitation retrieval assessment
- [Input Configuration](api/input) - Input data processing and configuration
- [Target Configuration](api/target) - Target precipitation data handling

## PyTorch Integration

- [PyTorch Utilities](api/pytorch/pytorch) - PyTorch integration helpers
- [PyTorch Datasets](api/pytorch/datasets) - Dataset classes for tabular and spatial data

## Utilities and Configuration

- [Configuration](api/config) - Package configuration management
- [Command Line Interface](api/cli) - CLI tools for data management
- [Utilities](api/utils) - General utility functions  
- [Logging](api/logging) - Logging configuration and utilities
- [Tiling](api/tiling) - Dataset tiling functionality
- [Plotting](api/plotting) - Visualization functions
- [Baselines](api/baselines) - Baseline retrieval algorithms

## Quick Reference

### Data Access
- `{py:func}satrain.data.get_files` - Get dataset files matching criteria
- `{py:func}satrain.data.download_missing` - Download missing data files
- `{py:func}satrain.data.load_tabular_data` - Load tabular data for ML

### Evaluation
- `{py:class}satrain.evaluation.Evaluator` - Main evaluation framework
- `{py:class}satrain.evaluation.InputFiles` - Input file management

### Configuration
- `{py:mod}satrain.config` - Package configuration management
- `{py:mod}satrain.cli` - Command-line interface

### PyTorch Datasets
- `{py:class}satrain.pytorch.datasets.SatRainTabular` - Tabular dataset loader
- `{py:class}satrain.pytorch.datasets.SatRainSpatial` - Spatial dataset loader

