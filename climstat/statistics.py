"""
Step 3: Compute summary statistics for derived climate metrics.

What this module does
---------------------
After Step 2 produces hourly metric values on the ERA5 grid (e.g. hourly
Heat Index at each grid cell), we need to distill those into useful
summaries.  This module provides two levels of aggregation:

1. **Daily summary** — one value per day per grid cell.
   Useful for time-series analysis.  Produces:
     - daily_mean, daily_max, daily_min
     - hours_above_{threshold} for each "above" threshold
     - hours_below_{threshold} for each "below" threshold

2. **Averages** — one value per grid cell, no time dimension.
   Useful for county-level maps and reports.  Produces:
     - avg_daily_max, avg_daily_mean, avg_daily_min
     - avg_days_per_year_above_{threshold}, avg_hours_per_year_above_{threshold}
     - avg_days_per_year_below_{threshold}, avg_hours_per_year_below_{threshold}

Key xarray concepts used here
------------------------------
- **.resample(time="1D")**: Groups hourly data into daily bins (like
  pandas groupby, but along the time dimension).  After resampling you
  can apply .mean(), .max(), .min(), .sum(), etc.
- **.groupby("time.year")**: Groups data by calendar year — useful for
  computing per-year totals that we then average.
- **.astype(int)**: Converts True/False to 1/0, so we can sum them to
  count how many hours a threshold was exceeded.
"""

import numpy as np
import xarray as xr


# ===========================================================================
# Daily summaries  (one value per day per grid cell — a time series)
# ===========================================================================

def daily_summary(
    metric_hourly: xr.DataArray,
    thresholds: list[float] | None = None,
    thresholds_below: list[float] | None = None,
) -> xr.Dataset:
    """
    Compute daily summary statistics from hourly metric data.

    For each day and each grid cell, this computes the mean, max, and min
    of the metric, plus the number of hours above/below each threshold.

    Parameters
    ----------
    metric_hourly : xr.DataArray
        Hourly metric values with dimensions (time, lat, lon).
        Units should match the thresholds (e.g. both in °F).
    thresholds : list[float], optional
        Values to check exceedance against.  For example, [90, 104] for
        Heat Index means we count hours above 90°F and above 104°F each day.
    thresholds_below : list[float], optional
        Values to check sub-threshold against.  For example, [0, 32] for
        Wind Chill means we count hours below 0°F and below 32°F each day.

    Returns
    -------
    xr.Dataset with variables:
        daily_mean              — average value for each day
        daily_max               — highest value for each day
        daily_min               — lowest value for each day
        hours_above_{threshold} — hours the metric exceeded the threshold (0-24)
        hours_below_{threshold} — hours the metric was below the threshold (0-24)
    """
    if thresholds is None:
        thresholds = []
    if thresholds_below is None:
        thresholds_below = []

    ds = xr.Dataset()

    # .resample(time="1D") groups the 24 hourly values in each day,
    # then .mean()/.max()/.min() collapses those 24 values into one.
    ds["daily_mean"] = metric_hourly.resample(time="1D").mean()
    ds["daily_max"] = metric_hourly.resample(time="1D").max()
    ds["daily_min"] = metric_hourly.resample(time="1D").min()

    for t in thresholds:
        label = f"hours_above_{t}"
        # (metric_hourly > t) produces a boolean array (True/False at each hour)
        # .astype(int) converts True->1, False->0
        # .resample(time="1D").sum() adds up the 1s in each day = number of hours
        exceeded = (metric_hourly > t).astype(int)
        ds[label] = exceeded.resample(time="1D").sum(dim="time")

    for t in thresholds_below:
        label = f"hours_below_{t}"
        # Same logic as above, but counting hours *below* the threshold
        below = (metric_hourly < t).astype(int)
        ds[label] = below.resample(time="1D").sum(dim="time")

    return ds


# ===========================================================================
# Averages  (one value per grid cell — climatological averages)
# ===========================================================================

def averages_summary(
    metric_hourly: xr.DataArray,
    thresholds: list[float] | None = None,
    thresholds_below: list[float] | None = None,
) -> xr.Dataset:
    """
    Compute climatological (time-collapsed) summary statistics.

    These are long-term averages over the entire period — one number per
    grid cell, with no time dimension.  They answer questions like:
    "On average, how many days per year does WBGT exceed 85°F in this
    grid cell?"

    Parameters
    ----------
    metric_hourly : xr.DataArray
        Hourly metric values with dimensions (time, lat, lon).
    thresholds : list[float], optional
        Threshold values for above-exceedance statistics.
    thresholds_below : list[float], optional
        Threshold values for below-threshold statistics (e.g. wind chill).

    Returns
    -------
    xr.Dataset with variables (no time dimension, just lat/lon):
        avg_daily_max                    — mean of daily maxima across all days
        avg_daily_mean                   — mean of daily means across all days
        avg_daily_min                    — mean of daily minima across all days
        avg_days_per_year_above_{t}      — for each above threshold
        avg_hours_per_year_above_{t}     — for each above threshold
        avg_days_per_year_below_{t}      — for each below threshold
        avg_hours_per_year_below_{t}     — for each below threshold
    """
    if thresholds is None:
        thresholds = []
    if thresholds_below is None:
        thresholds_below = []

    # First compute daily aggregates (these still have a time dimension,
    # one value per day)
    daily_max = metric_hourly.resample(time="1D").max()
    daily_mean = metric_hourly.resample(time="1D").mean()
    daily_min = metric_hourly.resample(time="1D").min()

    ds = xr.Dataset()

    # Average the daily values over the entire period to get one number
    # per grid cell.  For example, avg_daily_max is "the average of all
    # daily maximum temperatures over the 10-year period".
    ds["avg_daily_max"] = daily_max.mean(dim="time")
    ds["avg_daily_mean"] = daily_mean.mean(dim="time")
    ds["avg_daily_min"] = daily_min.mean(dim="time")

    for t in thresholds:
        # ── Days per year above threshold (averaged across years) ─────
        # Step 1: For each day, check if any hour exceeded the threshold
        daily_exceeded = (metric_hourly > t).resample(time="1D").max()
        # Step 2: Convert True/False to 1/0
        daily_exceeded_int = daily_exceeded.astype(int)
        # Step 3: Sum up exceedance days within each calendar year
        days_per_year = daily_exceeded_int.groupby("time.year").sum(dim="time")
        # Step 4: Average across years to get "avg days/year above threshold"
        ds[f"avg_days_per_year_above_{t}"] = days_per_year.mean(dim="year")

        # ── Hours per year above threshold (averaged across years) ────
        # Same idea, but counting individual hours instead of days
        hourly_exceeded = (metric_hourly > t).astype(int)
        hours_per_year = hourly_exceeded.groupby("time.year").sum(dim="time")
        ds[f"avg_hours_per_year_above_{t}"] = hours_per_year.mean(dim="year")

    for t in thresholds_below:
        # ── Days per year below threshold (averaged across years) ─────
        # Same logic as above, but counting days where the metric dropped
        # below the threshold (e.g. wind chill below 0°F)
        daily_below = (metric_hourly < t).resample(time="1D").max()
        daily_below_int = daily_below.astype(int)
        days_per_year = daily_below_int.groupby("time.year").sum(dim="time")
        ds[f"avg_days_per_year_below_{t}"] = days_per_year.mean(dim="year")

        # ── Hours per year below threshold (averaged across years) ────
        hourly_below = (metric_hourly < t).astype(int)
        hours_per_year = hourly_below.groupby("time.year").sum(dim="time")
        ds[f"avg_hours_per_year_below_{t}"] = hours_per_year.mean(dim="year")

    return ds
