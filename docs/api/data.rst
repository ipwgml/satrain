Data Access and Management
==========================

.. currentmodule:: satrain.data

The :py:mod:`satrain.data` module provides functionality for accessing, downloading, and loading SatRain datasets.

Key Functions
-------------

.. autofunction:: get_files
.. autofunction:: download_missing
.. autofunction:: load_tabular_data
.. autofunction:: list_local_files

Dataset Subsets
---------------

The SatRain dataset is available in multiple subset sizes to accommodate different use cases:

* ``xs`` - Extra small (~1-5 GB): Quick testing and tutorials
* ``s`` - Small (~10-20 GB): Development and prototyping  
* ``m`` - Medium (~50-100 GB): Algorithm validation
* ``l`` - Large (~200-500 GB): Model development
* ``xl`` - Extra large (~1-2 TB): Full-scale training

Usage Examples
--------------

Basic file access::

    from satrain.data import get_files
    
    # Get files for small subset
    files = get_files(
        base_sensor='gmi',
        split='training',
        subset='s',  # Small subset for development
        geometry='gridded'
    )

Download missing data::

    from satrain.data import download_missing
    
    # Download small subset for quick start
    download_missing(
        base_sensor='gmi',
        split='training',
        subset='xs'  # Start with extra small
    )

Load tabular data efficiently::

    from satrain.data import load_tabular_data
    
    # Load progressively larger subsets
    
    # Start small for development
    data_small = load_tabular_data(
        base_sensor='gmi',
        subset='xs',
        inputs=['gmi']
    )
    
    # Scale up for training
    data_large = load_tabular_data(
        base_sensor='gmi', 
        subset='l',  # Large subset for serious training
        inputs=['gmi', 'geo', 'ancillary']
    )

Subset Selection Guidelines::

    # For learning and quick experiments
    subset = 'xs'  # 1-5 GB, ~1K-5K scenes
    
    # For algorithm development
    subset = 's'   # 10-20 GB, ~10K-20K scenes
    
    # For validation and comparison
    subset = 'm'   # 50-100 GB, ~50K-100K scenes
    
    # For model development
    subset = 'l'   # 200-500 GB, ~200K-500K scenes
    
    # For production training
    subset = 'xl'  # 1-2 TB, ~1M+ scenes

All Functions
-------------

.. automodule:: satrain.data
    :members:
    :undoc-members:
    :show-inheritance:
    :no-index:
