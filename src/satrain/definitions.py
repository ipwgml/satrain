"""
satrain.definitions
===================

This module defines shared attributes for the satrain package.
"""

ANCILLARY_VARIABLES = [
    "ten_meter_wind_u",
    "ten_meter_wind_v",
    "two_meter_dew_point",
    "two_meter_temperature",
    "cape",
    "sea_ice_concentration",
    "sea_surface_temperature",
    "skin_temperature",
    "snow_depth",
    "snowfall",
    "surface_pressure",
    "total_column_cloud_ice_water",
    "total_column_cloud_liquid_water",
    "total_column_water_vapor",
    "total_precipitation",
    "convective_precipitation",
    "leaf_area_index",
    "surface_type",
    "elevation",
]

N_CLASSES = {
    "surface_type": 18,
}


ALL_INPUTS = ["gmi", "geo", "geo_ir", "ancillary", "geo_t", "geo_ir_t"]

BASE_SENSORS = ["gmi", "atms"]
GEOMETRIES = ["gridded", "on_swath"]
SPLITS = ["training", "validation", "testing"]
SIZES = ["xs", "s", "m", "l", "xl"]
DOMAINS = ["austria", "conus", "korea"]
