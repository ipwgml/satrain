PyTorch Datasets
================

.. currentmodule:: satrain.pytorch.datasets

The :py:mod:`satrain.pytorch.datasets` module provides PyTorch-compatible dataset classes for both tabular and spatial data.

Dataset Classes
---------------

.. autoclass:: SatRainTabular
   :members:
   :show-inheritance:

.. autoclass:: SatRainSpatial
   :members:
   :show-inheritance:

Subset Size Considerations
--------------------------

When working with PyTorch datasets, subset size affects memory usage and training time:

**Memory-efficient subsets:**

* ``xs``, ``s`` - Can typically fit in memory, good for development
* ``m`` - May require data chunking depending on available RAM
* ``l``, ``xl`` - Require careful memory management and data streaming

**Recommended workflow:**

1. Prototype with ``xs`` or ``s``
2. Validate with ``m`` 
3. Train production models with ``l`` or ``xl``

Usage Examples
--------------

Tabular dataset for pixel-based models::

    from satrain.pytorch.datasets import SatRainTabular
    from torch.utils.data import DataLoader
    
    # Start with small subset for development
    dataset = SatRainTabular(
        base_sensor='gmi',
        split='training',
        subset='s',  # Small subset for development
        inputs=['gmi'],
        transforms=None
    )
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )

Spatial dataset for image-based models::

    from satrain.pytorch.datasets import SatRainSpatial
    
    # Use medium subset for spatial models
    dataset = SatRainSpatial(
        base_sensor='gmi', 
        split='training',
        subset='m',  # Medium subset for spatial work
        inputs=['gmi', 'geo'],
        tile_size=128,
        transforms=None
    )

Progressive scaling workflow::

    # 1. Start development with extra small
    dev_dataset = SatRainTabular(subset='xs')
    
    # 2. Validate with small to medium
    val_dataset = SatRainTabular(subset='s')
    
    # 3. Train final model with large
    train_dataset = SatRainTabular(subset='l')

Memory management for large subsets::

    from torch.utils.data import DataLoader
    
    # For large subsets, use smaller batch sizes
    if subset in ['l', 'xl']:
        batch_size = 16  # Smaller batches for large datasets
        num_workers = 2  # Fewer workers to reduce memory usage
    else:
        batch_size = 32  # Standard batch size for smaller subsets
        num_workers = 4
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

Complete Module Reference
-------------------------

.. automodule:: satrain.pytorch.datasets
    :members:
    :undoc-members:
    :show-inheritance:
    :no-index:
