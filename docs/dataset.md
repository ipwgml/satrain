# The Satellite-based Rain Estimation and Detection (SatRain) Benchmark Dataset

The **Satellite-based Rain Estimation and Detection (SatRain) benchmark dataset** provides paired satellite observations and corresponding surface precipitation rate estimates derived from ground-based radar. It is designed to support the development and evaluation of machine-learning (ML) precipitation retrieval algorithms.  

SatRain is built from collocations of passive microwave (PMW) sensor overpasses from the [Global Precipitation Measurement (GPM)](https://gpm.nasa.gov/) mission. These PMW observations are augmented with time-resolved visible and infrared measurements from [GOES-16](https://www.star.nesdis.noaa.gov/GOES/conus.php?sat=G16), the [CPC Merged-IR dataset](https://www.cpc.ncep.noaa.gov/products/global_precip/html/wpage.merged_IR.html), and a range of environmental **ancillary data**.

---

## Data Organization

SatRain is organized to balance **ease of use** with **flexibility** for a wide range of retrieval scenarios. The dataset hierarchy is defined along several dimensions (data source, geometry, split, and subset) so that users can tailor data access to their workflow.  

```{table} SatRain data organization
:name: data_organization

| Configuration name | Possible values               | Significance                                                                                                             |   |
|--------------------|-------------------------------|--------------------------------------------------------------------------------------------------------------------------|---|
| Base sensor   | GMI, ATMS                     | The PMW sensor whose CONUS overpasses form the dataset foundation                                                        |   |
| Geometry           | on-swath, gridded             | Native spatial sampling (swath) or regridded to a regular latitude–longitude grid                                        |   |
| Training split     | training, validation, testing    | Partitioning for ML workflows                                                                                           |   |
| Format             | spatial, tabular              | 2D spatial scenes or flattened pixel samples                                                                            |   |
| Data source | gmi, atms, geo, geo_t, geo_ir, geo_ir_t, ancillary, target| Different input data sources and precipitation reference data. ||

### Base sensor

The SatRain dataset comprises two independent sub-datasets: the first one generated from CONUS overpasses of the GPM Microwave Imager (GMI) sensor, and the second one generated from overpasses of the Advanced Technology Microwave Sounder (ATMS). Since GMI and ATMS are different sensors with different viewing geometries and spectral sampling, and their overpasses occur at other times of the day and with different spatial sampling, these two subsets are treated as independent. The reference sensor constitutes the top level of the organization of the SatRain dataset.

### Gridded and on-swath geometries

The SatRain data are provided on both *on-swath* and *gridded* coordinate systems. Here, on-swath refers to the native, 2D scan pattern of the sensor, which is organized into scans and pixels, wheras *gridded* designates the observations remappted to a regular longitude-latitude grid with a resolution of 0.036 degree.

SatRain supports both of these geometries to allow greater flexibility in the design of the retrieval algorithm. Traditionally, many currently operational algorithms operate on single pixels, which makes the on-swath geometry a natural choice. However, a gridded geometry may be a more natural choice for ML-based retrievals, particularly for those combining observations from multiple sensors.


### Data splits

Following machine-learning best practices, the SatRain dataset provides separate training, validation, and testing splits. The training and
validation data are extracted from the collocations from 2018-2021. The validation data uses the collocations from each first five days, while the remaining days are assigned to the training data. Collocations from the year 2022 are used for the testing of the retrievals.

### Subsets

The data is split up into subsets to provide a hierarchy of dataset sizes. This is to allow users to get started using a smaller dataset but also provide a dataset large enough to train complex models. The subsets are 'xs', 's', 'm', 'l', 'xl'. While the files are not duplicated across those different subsets in the folder hierarchy, the subsets should be understood cumulatively meaning that, for example, the 'xl' dataset includes all files in 'xs', 's', 'm', and 'l' folder.


# Dataset generation

The basis of the SatRain dataset is formed by overpasses of the GMI and ATMS sensors
over CONUS. An example of such an overpass is shown in the figure below. The
figure shows an overpass of GMI over CONUS. Panel (a) shows the 89-H channels of
the GMI sensor. Panel (b) and (c) show visible observations from the GOES-16 ABI
and IR-window observations from the CPCIR merged-IR datset. Finally, panel (d)
shows the collocated precipitation measurements from the NOAA
[Multi-Radar/Multi-Sensor System](https://www.nssl.noaa.gov/projects/mrms/).

These ovepasses form the basis for the extraction of the SatRain data. For the
generation of the tabular data, all pixels with valid MRMS estimates are extracted
from the scene. The data in spatial format is extracted by extracting random
sub-scenes from the overpass. The scene sizes are 256x256 for the gridded data and
64x64 for the swath-based data. The sub-scenes are allowed to overlap by about 50%, to maximize the available training scenes. The scenes extracted for gridded and on-swath geometries are shown in the figure below using the grey, dashed and grey, dot-dashed lines respectively. Panels (e) and (f) showcase the extracted scenes highlighted in black in panel (b) and (d).

```{figure} /figures/example_scene.png
---
height: 400px
name: example_scene
---
Retrieval input and target data of the SatRain benchmark dataset. Panels (a), (b), and (c) show selected, collocated observations from the passive microwave (PMW) observations (Panel (a)) and geostationary visible (Panel (b)) and infrared observations that make up the input data of the SatRain data. Panel (d) shows the precipitation-radar-based precipitation estimates that are the retrieval targets. Grey dashed and dash-dotted lines mark the outlines of the training samples extracted from this collocation scene for the gridded and on-swath observations. Black dashed and dash-dotted lines mark the sample training scenes displayed in Panel (e) and (f).
```

## Preprocessing

The SatRain dataset is generated in two steps: The first one consists of extracting the collocation scenes from overpasses of GPM sensors over the MRMS domain. Each collocation scene is stored in the on-swath coordinate system following the spatial sampling of the GPM sensor and regridded to a uniform latitude-longitude grid.

The processing flow is illustrated in the figure below. The PMW observations
from the GPM sensor are augmented with ancillary data. MRMS measurements are
matched with the PMW observations by loading all two-minute measurements that
cover the overpass time and interpolating them to the nearest scan time of the
PMW sensor. The MRMS measurements are then downsampled spatially by smoothing
them with a Gaussian filter with a full-width at half-maximum of 0.036 degree
and interpolating them to the 0.036-degree resolution grid used by SatRain. During
the downsampling the MRMS precipitation classes are transformed to
precipitation-type fractions by calculating the fractional occurrence of each
class within the Gaussian smoothing kernel.

The MRMS measurements and PMW observations are combined into the gridded collocation scene by interpolating the PMW observations to the regular 0.036-degree grid using nearest-neighbor interpolation. Similarly, the observations from the geostationary sensors are added to the scene by nearest-neighbor interpolation.

To create the on-swath collocations the MRMS measurements as well as the goestationary observations are interpolated to the PMW sensor observations using nearest-neightbor interpolation. 


```{figure} /figures/processing.svg
---
height: 400px
name: data_processing
---
Preprocessing flow for generating the collocation scenes upon which the SatRain dataset is based.
```

## Training file generation

The collocation scenes form the basis for the generation for the training files that make up the SatRain dataset. The data in tabular format is generated by extract all pixel that have valid MRMS estimates from all scenes separately for the on-swath and gridded geometries.

The training data in spatial format is generated by randomly extracting training scenes from the collocation files. The scenes are required to have at least 75% of valid MRMS pixels and are allowed to overlap by 50%. Scenes of size 256x256 pixels are extract from the gridded collocation files, while scenes of size 64x64 are extracted from the on-swath collocation files.

## Dataset structure

Physically, the SatRain dataset is oganized following the hierarchy defined in the table [above](data_organization). Note that the directory tree contains an additional directory ``evaluation``. This folder holds the full collocation scenes, which are used for the general retrieval evaluation, as described in [](evaluation.md). 



Note that the ``satrain`` package downloads data as required and local copies of the SatRain dataset may thus contain only a subset of the folders below.

