# Getting started

The easiest way to get started using the SatRain dataset is through the ``satrain`` Python package. While the full dataset can be
manually downloaded from [here](https://rain.atmos.colostate.edu/satrain_2/satrain/), the ``satrain`` package provides functionality to automate the data download and the loading of the training and testing data.


## Installing the ``satrain`` package


To install the latest version of satrain, use the following command:

```
pip install satrain[complete]@git+https://github.com/satrain/satrain
```

> **Note**: The above command installs all dependencies required to run the examples included here. If
this is a concern, use ``pip install git+https://github.com/satrain/satrain`` for a minimal installation.


Installing the ``satrain`` package is all that is required to start using the package. If you just want to quickly test the dataset feel free to skip ahead to the neural net [examples](examples). Read on, to learn how to configure where the SatRain data is stored on your machine.

## Optional: Data download using the ``satrain`` command

```{note}
Most functionality of the ``satrain`` dataset that requires access to the SatRain dataset will automatically download the required files. Therefore, the following steps are not
strictly required to get started using the package.
```

Use the satrain command-line interface to download parts or all of the SatRain dataset:

```
satrain download --data_path /path/to/store/data --sensors gmi --subset s --splits training,validation,testing --geometries gridded
```

This will download the gridded SatRain training, validation, and testing data. The ``satrain download`` command takes the following options:

 - ``--sensors`` A comma-separated lists of the sensors for which to download the data.
 - ``--subset`` The size of the subset to download. Choose ``xl`` for the full dataset.
 - ``--splits`` A comma-separated lists of the data splits to download. Available options are ``training``, ``validation``, ``testing``, and ``evaluation``.
 - ``--geometries`` A comma-separated lists of the data geometries. Available options are ``gridded`` for gridded
   observations and ``on_swath`` for the data on the PMW swath.
 - ``--inputs`` A comma-separated list of the input data to download. Available options are ``ancillary``, ``geo``, ``geo_ir`` and ``pmw``.
 - ``--formats`` A comma-separated list of the data formats to download. Available options are ``spatial`` for 2D
   training scenes and ``tabular`` for tabular data.
   
   
## Listing available files

 The ``iwpgml list`` command can be used to list the files on the local machine
 that ``satrain`` is aware of. After a successful download, it should show a
 table listing relative locations of each dataset and how many files it
 comprises.
 
## Configuring the data path

The ``satrain`` package expects data to be located in a path called the ``satrain`` data path.
``satrain`` does its best to keep track of the data path between subsequent
invocations to avoid downloading data multiple times.

After a fresh install, the ``data_path`` points to the current working directory.
To set an explicit ``data_path``, you can use the ``satrain config
set_data_path`` command. This will create a ``satrain`` configuration file in the
current user's configuration directory, which will allow the setting to persist
for subsequent use of the ``satrain`` package. A configuration file storing the
``data_path`` will also be created when the ``satrain download`` command is
invoked with the ``--data_path`` option.

Alternatively, the ``data_path`` can be set using the ``SATRAIN_DATA_PATH`` environment
variable. The path in ``SATRAIN_DATA_PATH`` will overwrite the setting in the user's configuration
file.

The ``satrain config show`` command can be used to find out the value of the ``satrain`` data path
and how it is derived:

```
satrain config show
```

