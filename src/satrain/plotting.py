"""
satrain.plotting
================

Provides plotting-related functionality.
"""
from pathlib import Path
from typing import List
import urllib

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
import xarray as xr


def set_style():
    """
    Set the SATRAIN matplotlib style.
    """
    plt.style.use(Path(__file__).parent / "files" / "satrain.mplstyle")

def add_ticks(
        ax: plt.Axes,
        lons: List[float],
        lats: list[float],
        left=True,
        bottom=True
) -> None:
    import cartopy.crs as ccrs
    """
    Add tick to cartopy Axes object.

    Args:
        ax: The Axes object to which to add the ticks.
        lons: The longitude coordinate at which to add ticks.
        lats: The latitude coordinate at which to add ticks.
        left: Whether or not to draw ticks on the y-axis.
        bottom: Whether or not to draw ticks on the x-axis.
    """
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0, color='none')
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = left
    gl.bottom_labels = bottom
    gl.xlocator = FixedLocator(lons)
    gl.ylocator = FixedLocator(lats)

cmap_precip = sns.cubehelix_palette(start=1.50, rot=-0.9, as_cmap=True, hue=0.8, dark=0.2, light=0.9)
cmap_tbs = sns.cubehelix_palette(start=2.2, rot=0.9, as_cmap=True, hue=1.3, dark=0.2, light=0.8, reverse=True)
cmap_tbs = sns.color_palette("rocket", as_cmap=True)


def scale_bar(
        ax,
        length,
        location=(0.5, 0.05),
        linewidth=3,
        height=0.01,
        border=0.05,
        border_color="k",
        parts=4,
        zorder=50,
        textcolor="k"
):
    """
    Draw a scale bar on a cartopy map.

    Args:
        ax: The matplotlib.Axes object to draw the axes on.
        length: The length of the scale bar in meters.
        location: A tuple ``(h, w)`` defining the fractional horizontal
            position ``h`` and vertical position ``h`` in the given axes
            object.
        linewidth: The width of the line.
    """
    import cartopy.crs as ccrs
    lon_min, lon_max, lat_min, lat_max = ax.get_extent(ccrs.PlateCarree())

    lon_c = lon_min + (lon_max - lon_min) * location[0]
    lat_c = lat_min + (lat_max - lat_min) * location[1]
    transverse_merc = ccrs.TransverseMercator(lon_c, lat_c)

    x_min, x_max, y_min, y_max = ax.get_extent(transverse_merc)

    x_c = x_min + (x_max - x_min) * location[0]
    y_c = y_min + (y_max - y_min) * location[1]

    x_left = x_c - length / 2
    x_right = x_c  + length / 2

    def to_axes_coords(point):
        crs = ax.projection
        p_data = crs.transform_point(*point, src_crs=transverse_merc)
        return ax.transAxes.inverted().transform(ax.transData.transform(p_data))

    def axes_to_lonlat(point):
        p_src = ax.transData.inverted().transform(ax.transAxes.transform(point))
        return ccrs.PlateCarree().transform_point(*p_src, src_crs=ax.projection)


    left_ax = to_axes_coords([x_left, y_c])
    right_ax = to_axes_coords([x_right, y_c])

    l_ax = right_ax[0] - left_ax[0]
    l_part = l_ax / parts



    left_bg = [
        left_ax[0] - border,
        left_ax[1] - height / 2 - border
    ]

    background = Rectangle(
        left_bg,
        l_ax + 2 * border,
        height + 2 * border,
        facecolor="none",
        transform=ax.transAxes,
        zorder=zorder
    )
    ax.add_patch(background)

    for i in range(parts):
        left = left_ax[0] + i * l_part
        bottom = left_ax[1] - height / 2

        color = "k" if i % 2 == 0 else "w"
        rect = Rectangle(
            (left, bottom),
            l_part,
            height,
            facecolor=color,
            edgecolor=border_color,
            transform=ax.transAxes,
            zorder=zorder
        )
        ax.add_patch(rect)

    x_bar = [x_c - length / 2, x_c + length / 2]
    x_text = 0.5 * (left_ax[0] + right_ax[0])
    y_text = left_ax[1] + 0.05 * height + 0.75 * border
    ax.text(x_text,
            y_text,
            f"{length / 1e3:g} km",
            transform=ax.transAxes,
            horizontalalignment='center',
            verticalalignment='center',
            color=textcolor
    )


def download_blue_marble(texture_file: Path) -> Path:
    """
    Download the texture file if it doesn't exist.

    Args:
        url: String specifying the URL of the texture file.
        texture_file: Path object pointing to the file to which to download the texture image.
    """
    url = "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73751/world.topo.bathy.200407.3x21600x10800.jpg"
    if not texture_file.exists():
        urllib.request.urlretrieve(url, texture_file)
    return texture_file


def get_blue_marble(area: "pyresample.AreaDefinition") -> Image:
    """
    Get NASA Blue Marble background image.
    """
    from pansat.utils import resample_data
    Image.MAX_IMAGE_PIXELS = None   # disables the warning
    texture_file = Path(".") / "blue_marble_hires.png"
    img = np.array(Image.open(download_blue_marble(texture_file)))
    lats = np.linspace(90, -90, img.shape[0])
    lons = np.linspace(-180, 180, img.shape[1])
    blue_marble = xr.Dataset({
        "longitude": (("longitude"), lons),
        "latitude": (("latitude"), lats),
        "img": (("latitude", "longitude", "channels"), img / 256)
    })
    blue_marble_r = resample_data(blue_marble, area)
    return Image.fromarray((blue_marble_r.img.data * 255).astype("uint8"))
