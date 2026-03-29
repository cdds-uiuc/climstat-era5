"""
Module 1: Acquire raw climate data (ERA5 + IFS deterministic forecast).

Downloads (or loads from cache) ERA5 reanalysis and IFS gap-fill +
optional deterministic forecast data for the requested metrics.
"""

import pandas as pd

from .era5_extract import extract_era5
from .ifs_extract import extract_ifs
from ..data_process.metrics import METRIC_REGISTRY


def acquire_data(
    metrics: list[str],
    year_start: int,
    year_end: int,
    cache_dir: str = "data/cache",
    forecast_days: int = 0,
) -> dict:
    """
    Download ERA5 and IFS raw data (Module 1).

    Determines which ERA5 variables are needed for the requested metrics,
    downloads (or loads from cache) the ERA5 data, and fills the gap with
    IFS data.  Optionally extends IFS into the future as a deterministic
    forecast.

    Parameters
    ----------
    metrics : list[str]
        Metric names to compute (must be keys in METRIC_REGISTRY).
    year_start, year_end : int
        Year range for ERA5 extraction (inclusive).
    cache_dir : str
        Directory for raw NetCDF cache files.
    forecast_days : int
        Number of days of IFS deterministic forecast to include beyond
        yesterday (default 0).  Set to e.g. 7 for a 1-week forecast.

    Returns
    -------
    dict with keys:
        "era5_ds"       — xr.Dataset of ERA5 hourly data
        "ifs_ds"        — xr.Dataset of IFS hourly data, or None
        "era5_end_date" — datetime.date of last valid ERA5 day
    """
    # Determine which ERA5 variables are needed
    needed_vars = set()
    for m in metrics:
        needed_vars.update(METRIC_REGISTRY[m]["era5_vars"])
    needed_vars = sorted(needed_vars)
    print(f"ERA5 variables needed: {needed_vars}")

    # Extract ERA5
    era5_ds = extract_era5(
        variables=needed_vars,
        year_start=year_start,
        year_end=year_end,
        cache_dir=cache_dir,
    )
    print(era5_ds)

    # Save grid info for IFS gap-fill
    era5_lats = era5_ds.lat.values.copy()
    era5_lons = era5_ds.lon.values.copy()
    era5_end_date = pd.Timestamp(era5_ds.time.values[-1]).date()
    print(f"ERA5 data ends: {era5_end_date}")

    # Extract IFS to fill the gap + optional forecast
    ifs_ds = extract_ifs(
        era5_end_date=era5_end_date,
        grid_lats=era5_lats,
        grid_lons=era5_lons,
        forecast_days=forecast_days,
        cache_dir=cache_dir,
        variables=needed_vars,
    )
    if ifs_ds is not None:
        print(f"IFS data: {pd.Timestamp(ifs_ds.time.values[0]).date()} -> "
              f"{pd.Timestamp(ifs_ds.time.values[-1]).date()}")
    else:
        print("No IFS data needed (ERA5 is up to date).")

    return {
        "era5_ds": era5_ds,
        "ifs_ds": ifs_ds,
        "era5_end_date": era5_end_date,
    }
