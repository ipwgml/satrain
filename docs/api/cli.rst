Command Line Interface
======================

.. currentmodule:: satrain.cli

The :py:mod:`satrain.cli` module provides the command-line interface for the SatRain package, enabling data management and configuration from the terminal.

.. automodule:: satrain.cli
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

Configure data path::

    satrain config set_data_path /path/to/data

Show current configuration::

    satrain config show

Download dataset components::

    satrain download --sensors gmi --subset s --splits training,validation,testing --geometries gridded

List locally available files::

    satrain list_files