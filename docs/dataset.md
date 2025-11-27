# Dataset Overview

The SatRain dataset provides paired satellite observations with corresponding
surface precipitation rate estimates derived from ground-based radar and rain
gauges. Training and validation sets are composed of fixed-size scenes and are
available on two spatial grids: a 0.036° regular latitude–longitude grid
(*gridded*) and the native sampling of the passive microwave sensors
(*on-swath*).

Testing data is also provided in both gridded and on-swath formats, but instead
of fixed-size scenes it consists of full overpass scenes of irregular extent.

All training, validation, and testing data are stored in separate files for each
input type (ATMS, GMI, Geo, Geo-IR, ancillary data, time-resolved Geo,
time-resolved Geo-IR) and for the precipitation reference. File names follow the
pattern ``<prefix>_YYYYmmddHHMMSS.nc``, where ``<prefix>`` identifies the data
type (``gmi``, ``atms``, ``geo``, ``geo_t``, ``geo_ir``, ``geo_ir_t``,
``ancillary``, ``target``) and ``YYYYmmddHHMMSS`` denotes the median scan time
of the corresponding PMW observation. Files belonging to the same scene can be
matched by their shared timestamp.

## Organization

SatRain is organized to balance **ease of use** with **flexibility** for a wide
range of retrieval scenarios. The dataset hierarchy is defined along several
dimensions (base sensor, split, subset, geometry, and data source) allowing
users to access only the data needed for their use case.

```{table} SatRain data organization
:name: data_organization

| Configuration name | Possible values               | Significance                                                                                                             |   |
|--------------------|-------------------------------|--------------------------------------------------------------------------------------------------------------------------|---|
| Base sensor   | ``gmi``, ``atms``                     | The PMW sensor whose CONUS overpasses form the dataset's foundation                                                        |   
| Split     | ``training``, ``validation``, ``testing``    | Partitioning into training, validation, and testing data. |
| Subset     | ``xs``, ``s``, ``m``, ``l``, ``xl``    | The training and validation datasets are split into size-based subsets for users who wish to get started with smaller datasets or assess the scaling of ML models. |
| Domain     | ``austria``, ``conus``, ``korea``| The testing data is available from three domains: Austria, CONUS, and Korea |
| Geometry           | ``on_swath``, ``gridded``             | Native spatial sampling (on-swath) or regridded to a regular 0.036° latitude-longitude grid                                        |  
| Data source | ``gmi``, ``atms``, ``geo``, ``geo_t``, ``geo_ir``, ``geo_ir_t``, ``ancillary``, ``target`` | Different input data sources and precipitation reference data (``target``). |
```

### Base sensor

The SatRain dataset comprises two independent sub-datasets: the first one
generated from the GPM Microwave Imager (GMI) sensor, and the second one
generated from overpasses of the Advanced Technology Microwave Sounder (ATMS).
The GMI and ATMS-based subsets are completely independent subsets. The two
sensors are included here to allow testing of algorithms on both a dedicated
precipitation sensor (GMI) and a microwave sounding instrument not primarily
designed for precipitation remote sensing (ATMS).

Due to the larger swath width of ATMS, the gridded dataset is considerably
larger. Users who wish to train retrievals based on geostationary sensors on as much
data as possible, are therefore advised to use the ``atms`` subset.

### Gridded and on-swath geometries

The SatRain data are provided on both *on-swath* and *gridded* coordinate
systems. Here, on-swath refers to the native, 2D scan pattern of the sensor,
which is organized into scans and pixels, wheras *gridded* designates the
observations remappted to a regular longitude-latitude grid with a resolution of
0.036°.

SatRain supports both of these geometries to allow greater flexibility in the
design of the retrieval algorithm. Traditionally, many currently operational
algorithms operate on single pixels, which makes the on-swath geometry a natural
choice. However, a gridded geometry may be a more natural choice for image based
retrievals, particularly for those combining observations from multiple sensors.


### Data splits

Following machine-learning best practices, the SatRain dataset provides separate
training, validation, and testing splits. The training and validation data are
extracted from the collocations from 2018-2021 over CONUS. The validation data
uses the collocations from each month's first five days, while the remaining
days are assigned to the training data.

The testing data is separated into data extracted over CONUS and two additional,
independent testing datasets from Austria and Korea. As opposed to the training
and validation data, the testing data is not split into fixed-size scenes. The
testing data retains the structure of the original PMW overpasses to make it
easy to compare SatRain retrievals against existing retrievals. To simplify evaluating
retrieval on the testing data, the ``satrain.evaluation`` module provides functionality
to tile and batch the input data from the testing data. See the documentation available
[here](evaluation).


### Subsets

The data is split up into subsets to provide a hierarchy of dataset sizes. This
is to allow users to get started using a smaller dataset but also provide a
dataset large enough to train complex models. The subsets should be understood 
cumulatively meaning that, for example, the 'xl' dataset includes all files in 
'xs', 's', 'm', and 'l' folders. 

```{table} Dataset Subset Sizes
:name: subset_sizes

| Subset | Approximate Size | Number of Scenes (Gridded)     |
|--------|------------------|--------------------------------|
| ``xs`` | 1 GB             | ~500                           |
| ``s``  | 7 GB             | ~2, 000                        |
| ``m``  | 10 GB            | ~5, 000                        |
| ``l``  | 70 GB            | ~20, 000                       |
| ``xl`` | 1-2 TB           | ~50, 000                       |
```


#### Subset Selection Guidelines

**For beginners and development:**
- Start with ``xs`` for learning the API and data structure
- Use ``s`` for initial algorithm development and testing
- Progress to ``m`` for validation and comparison

**For research and production:**
- Use ``l`` for serious model development and hyperparameter optimization
- Use ``xl`` for final training runs and publication-quality results
- Consider computational resources when selecting larger subsets

#### Cumulative Nature of Subsets

Subsets are inclusive, meaning:
- ``s`` contains all data from ``xs`` plus additional scenes
- ``m`` contains all data from ``xs`` + ``s`` plus additional scenes  
- ``l`` contains all data from ``xs`` + ``s`` + ``m`` plus additional scenes
- ``xl`` contains all available training/validation data

This design allows you to:
1. Start development with smaller subsets
2. Scale up gradually as needed
3. Ensure reproducibility when moving between subset sizes
4. Compare results across different data volumes

## File Structure

### Training and Validation Data
The data is organzied into a folder structure following the hierarchy explained above. For the training data the folder structure looks as follows.

```
<satrain_data_path>
└── satrain
    └── <gmi/atms>
        └── <training/validation>
            ├── xs
            │   ├── gridded
            │   │   └── <year>/<month>/<day>/
            │   └── on_swath
            │       └── <year>/<month>/<day>/
            ├── ...
            ├── ...
            └── xl
                ├── gridded
                │   └── <year>/<month>/<day>/
                └── on_swath
                    └── <year>/<month>/<day>/
```

### Testing Data

For the testing data, the size-based subsets are replaced by the three domains: ``austria``, ``conus``, and ``korea``.

```
<satrain_data_path>
    satrain
    └── <gmi/atms>
        └── testing
            ├── austria
            │   ├── gridded
            │   │   └── <year>/<month>/<day>/
            │   └── on_swath
            │       └── <year>/<month>/<day>/
            ├── conus
            │   ├── gridded
            │   │   └── <year>/<month>/<day>/
            │   └── on_swath
            │       └── <year>/<month>/<day>/
            └── korea
                ├── gridded
                │   └── <year>/<month>/<day>/
                └── on_swath
                    └── <year>/<month>/<day>/
```





