"""
satrain.baselines
=================

This module provides access to results from baseline retrievals.
"""

from pathlib import Path
from typing import List, Optional

from satrain.definitions import BASE_SENSORS
import xarray as xr

BASELINES_GMI = {
    "era5": "ERA5",
    "gprof_v07": "GPROF V7 (GMI)",
}

BASELINES_ATMS = {
    "era5": "ERA5",
    "gprof_v07": "GPROF V7 (ATMS)",
}


def load_baseline_results(
    base_sensor: str,
    domain: str = "conus",
    baselines: Optional[List[str]] = None,
) -> xr.Dataset:
    """
    Load baseline results.

    Args:
        baselines: An optional list containing the baseline names to load. If not givne,
            results from all baselines are loaded.

    Return:
        An xarray.Dataset containing the merged baseline results.
    """
    if base_sensor.lower() == "gmi":
        BASELINES = BASELINES_GMI
    elif base_sensor.lower() == "atms":
        BASELINES = BASELINES_ATMS
    else:
        raise ValueError(
            f"Unsupport base sensor '{base_sensor}'. Shoule be one of {BASE_SENSORS}"
        )

    if baselines is None:
        baselines = BASELINES.keys()

    data_path = Path(__file__).parent / "files" / "baselines"
    results = []

    for baseline in baselines:
        if baseline not in BASELINES:
            raise ValueError(f"Encountered unsupported baseline name '{baseline}'.")

        results.append(xr.load_dataset(data_path / (f"{baseline}_{base_sensor.lower()}_{domain}.nc")))

    results = xr.concat(results, dim="algorithm")
    results["algorithm"] = (("algorithm",), [BASELINES[name] for name in baselines])
    return results
