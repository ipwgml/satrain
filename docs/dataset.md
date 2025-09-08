# Overview and Organization

The **Satellite-based Rain Estimation and Detection (SatRain) benchmark dataset** provides paired satellite observations and corresponding surface precipitation rate estimates derived from ground-based radar. It is designed to support the development and evaluation of machine-learning (ML) precipitation retrieval algorithms.

SatRain is built from collocations of passive microwave (PMW) sensor overpasses from the [Global Precipitation Measurement (GPM)](https://gpm.nasa.gov/) mission. These PMW observations are augmented with time-resolved visible and infrared measurements from [GOES-16](https://www.star.nesdis.noaa.gov/GOES/conus.php?sat=G16), the [CPC Merged-IR dataset](https://www.cpc.ncep.noaa.gov/products/global_precip/html/wpage.merged_IR.html), and a range of environmental **ancillary data**.

---

## Data Organization

SatRain is organized to balance **ease of use** with **flexibility** for a wide range of retrieval scenarios. The dataset hierarchy is defined along several dimensions (base sensor, split, subset, geometry, and data source) allowing users to access only the data needed for their use case. Th

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
Since GMI and ATMS are different sensors with different viewing geometries and
spectral sampling, and their overpasses at different times, these two subsets
are treated as independent. The reference sensor constitutes the top level of
the organization of the SatRain dataset.

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
days are assigned to the training data. The testing data is separated into data
extracted over CONUS and two additional, independent testing datasets from
Austria and Korea.

### Subsets

The data is split up into subsets to provide a hierarchy of dataset sizes. This
is to allow users to get started using a smaller dataset but also provide a
dataset large enough to train complex models. The subsets are 'xs', 's', 'm',
'l', 'xl'. While the files are not duplicated across those different subsets in
the folder hierarchy, the subsets should be understood cumulatively meaning
that, for example, the 'xl' dataset includes all files in 'xs', 's', 'm', and
'l' folder.

### File Organization
 


#### Training and Validation Data


#### Testing data

