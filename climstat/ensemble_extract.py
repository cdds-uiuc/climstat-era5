"""
Step 1c: Extract ECMWF IFS ensemble forecast data from Open-Meteo.

What this module does
---------------------
Downloads 14-day (configurable) ensemble forecasts from the ECMWF IFS model
via the Open-Meteo Ensemble API.  The ensemble has 51 members (1 control +
50 perturbed), providing uncertainty estimates for all forecast variables.

This module is fully independent of the deterministic IFS gap-fill module
(ifs_extract.py).  It reuses helper functions from ifs_extract but maintains
its own cache files, API endpoint, and data structure.

Data source
-----------
- Open-Meteo Ensemble API: https://open-meteo.com/en/docs/ensemble-api
- Model: ECMWF IFS 0.25° (ecmwf_ifs025), 51 ensemble members
- Max forecast horizon: 15 days, hourly (interpolated from 3-hourly)
- Variables: temperature_2m, dew_point_2m, wind_speed_10m, wind_direction_10m,
  precipitation

Caching strategy
----------------
Ensemble data is cached as one NetCDF file per variable, with a `member`
dimension (0-50):

    ensemble_{variable}_{start_YYYYMMDD}_{end_YYYYMMDD}.nc

Cache staleness: 6 hours (same as deterministic IFS).
"""

import os
import glob
import json
import time as _time
import datetime
import urllib.request
import urllib.parse

import numpy as np
import xarray as xr
import pandas as pd

from .ifs_extract import (
    _lon_360_to_180,
    _lon_180_to_360,
    _speed_direction_to_uv,
    _safe_array,
    ERA5_TO_OPENMETEO,
    _expand_era5_vars,
    _era5_vars_to_openmeteo,
    MAX_RETRIES,
    RETRY_BACKOFF_SECONDS,
    IFS_CACHE_STALENESS_HOURS,
)


# ── Open-Meteo Ensemble API endpoint ─────────────────────────────────────
OPENMETEO_ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"

# ── Batching ──────────────────────────────────────────────────────────────
# Smaller than deterministic IFS because ensemble responses are ~51x larger
ENSEMBLE_BATCH_SIZE = 20

# ── Number of ensemble members ────────────────────────────────────────────
N_MEMBERS = 51  # 1 control + 50 perturbed


# ===========================================================================
# API fetching
# ===========================================================================

def _fetch_ensemble_batch(
    lats: list[float],
    lons: list[float],
    forecast_days: int,
    hourly_vars: list[str] | None = None,
) -> list[dict]:
    """
    Fetch ensemble forecast data from Open-Meteo for a batch of locations.

    Parameters
    ----------
    lats, lons : list[float]
        Coordinate pairs in -180:180 convention.
    forecast_days : int
        Number of forecast days to request.
    hourly_vars : list[str], optional
        Open-Meteo hourly variable names.  Defaults to all.

    Returns
    -------
    list[dict]
        One dict per location with ensemble member data in 'hourly'.
    """
    if hourly_vars is None:
        hourly_vars = sorted({
            v for vars_list in ERA5_TO_OPENMETEO.values() for v in vars_list
        })
    params = {
        "latitude": ",".join(f"{lat:.2f}" for lat in lats),
        "longitude": ",".join(f"{lon:.2f}" for lon in lons),
        "hourly": ",".join(hourly_vars),
        "models": "ecmwf_ifs025",
        "wind_speed_unit": "ms",
        "forecast_days": str(forecast_days),
    }
    url = OPENMETEO_ENSEMBLE_URL + "?" + urllib.parse.urlencode(params)

    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            if isinstance(data, dict):
                if "error" in data and "hourly" not in data:
                    raise RuntimeError(
                        f"Open-Meteo Ensemble API error: {data.get('reason', data)}"
                    )
                return [data]
            return data

        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            if attempt < MAX_RETRIES - 1:
                is_rate_limit = isinstance(e, urllib.error.HTTPError) and e.code == 429
                wait = 30 if is_rate_limit else RETRY_BACKOFF_SECONDS * (2 ** attempt)
                print(f"[ensemble_extract]   Retry {attempt + 1}/{MAX_RETRIES} "
                      f"after {wait}s ({'rate limited' if is_rate_limit else e})")
                _time.sleep(wait)
            else:
                raise RuntimeError(
                    f"Open-Meteo Ensemble API failed after {MAX_RETRIES} attempts: {e}"
                ) from e
    return []  # unreachable


# ===========================================================================
# Response parsing
# ===========================================================================

def _ensemble_responses_to_dataset(
    all_responses: list[dict],
    grid_lats: np.ndarray,
    grid_lons_360: np.ndarray,
    era5_vars: list[str],
) -> xr.Dataset:
    """
    Convert per-point ensemble responses into a gridded xr.Dataset with
    a member dimension.

    Parameters
    ----------
    all_responses : list[dict]
        Flat list of API responses, ordered lat-major.
    grid_lats : array
        Latitude coordinates (descending), matching ERA5.
    grid_lons_360 : array
        Longitude coordinates in 0-360 convention.
    era5_vars : list[str]
        ERA5 variable names to include.

    Returns
    -------
    xr.Dataset
        Dimensions: (member, time, lat, lon).
    """
    n_lat = len(grid_lats)
    n_lon = len(grid_lons_360)
    n_points = n_lat * n_lon

    if len(all_responses) != n_points:
        raise ValueError(
            f"Expected {n_points} responses, got {len(all_responses)}"
        )

    # Parse time axis from the first response
    time_strings = all_responses[0]["hourly"]["time"]
    times = pd.to_datetime(time_strings)
    n_times = len(times)

    # Detect actual number of members from the response
    first_hourly = all_responses[0]["hourly"]
    member_keys = sorted(
        k for k in first_hourly.keys()
        if k.startswith("temperature_2m_member")
    )
    n_perturbed = len(member_keys)
    n_members = 1 + n_perturbed  # control + perturbed
    print(f"[ensemble_extract] Detected {n_members} ensemble members "
          f"(1 control + {n_perturbed} perturbed)")

    def to_grid(flat_arr):
        """Reshape (n_members, n_points, n_times) -> (n_members, n_times, n_lat, n_lon)."""
        return flat_arr.reshape(n_members, n_lat, n_lon, n_times).transpose(0, 3, 1, 2)

    era5_var_set = set(era5_vars)
    need_temp = "2m_temperature" in era5_var_set
    need_dewp = "2m_dewpoint_temperature" in era5_var_set
    need_wind = ("10m_u_component_of_wind" in era5_var_set or
                 "10m_v_component_of_wind" in era5_var_set)
    need_precip = "total_precipitation" in era5_var_set

    data_vars = {}
    members = np.arange(n_members)

    def _extract_member_data(responses, base_var, n_pts, n_t, n_mem):
        """Extract all member data for a variable across all grid points.

        Returns array of shape (n_mem, n_pts, n_t).
        """
        arr = np.full((n_mem, n_pts, n_t), np.nan)
        for i, resp in enumerate(responses):
            hourly = resp["hourly"]
            # Member 0 = control (the base variable name)
            arr[0, i] = _safe_array(hourly[base_var], n_t)
            # Members 1..N = perturbed
            for m in range(1, n_mem):
                key = f"{base_var}_member{m:02d}"
                arr[m, i] = _safe_array(hourly[key], n_t)
        return arr

    if need_temp:
        temp_c = _extract_member_data(
            all_responses, "temperature_2m", n_points, n_times, n_members,
        )
        data_vars["2m_temperature"] = (
            ["member", "time", "lat", "lon"],
            to_grid(temp_c + 273.15),
        )

    if need_dewp:
        dewp_c = _extract_member_data(
            all_responses, "dew_point_2m", n_points, n_times, n_members,
        )
        data_vars["2m_dewpoint_temperature"] = (
            ["member", "time", "lat", "lon"],
            to_grid(dewp_c + 273.15),
        )

    if need_wind:
        wspd = _extract_member_data(
            all_responses, "wind_speed_10m", n_points, n_times, n_members,
        )
        wdir = _extract_member_data(
            all_responses, "wind_direction_10m", n_points, n_times, n_members,
        )
        u10, v10 = _speed_direction_to_uv(wspd, wdir)
        data_vars["10m_u_component_of_wind"] = (
            ["member", "time", "lat", "lon"], to_grid(u10),
        )
        data_vars["10m_v_component_of_wind"] = (
            ["member", "time", "lat", "lon"], to_grid(v10),
        )

    if need_precip:
        precip_mm = _extract_member_data(
            all_responses, "precipitation", n_points, n_times, n_members,
        )
        data_vars["total_precipitation"] = (
            ["member", "time", "lat", "lon"],
            to_grid(precip_mm / 1000.0),
        )

    ds = xr.Dataset(
        data_vars,
        coords={
            "member": members,
            "time": times,
            "lat": grid_lats,
            "lon": grid_lons_360,
        },
    )
    return ds


# ===========================================================================
# Cache management
# ===========================================================================

def _ensemble_cache_filename(
    variable: str,
    date_start: datetime.date,
    date_end: datetime.date,
) -> str:
    """Build the cache filename for one ensemble variable."""
    var_safe = variable.replace(" ", "_").replace("-", "_")
    return (
        f"ensemble_{var_safe}"
        f"_{date_start.strftime('%Y%m%d')}"
        f"_{date_end.strftime('%Y%m%d')}.nc"
    )


def _find_ensemble_cached(
    cache_dir: str,
    fc_start: datetime.date,
    fc_end: datetime.date,
    era5_vars: list[str],
) -> bool:
    """Check whether all per-variable ensemble cache files exist and are fresh."""
    for var in era5_vars:
        path = os.path.join(
            cache_dir, _ensemble_cache_filename(var, fc_start, fc_end),
        )
        if not os.path.exists(path):
            return False

        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(path))
        age_hours = (datetime.datetime.now() - mtime).total_seconds() / 3600
        if age_hours > IFS_CACHE_STALENESS_HOURS:
            print(f"[ensemble_extract] Cache {os.path.basename(path)} is "
                  f"{age_hours:.1f}h old — will re-download.")
            return False

    return True


def _load_ensemble_cached(
    cache_dir: str,
    fc_start: datetime.date,
    fc_end: datetime.date,
    era5_vars: list[str],
) -> xr.Dataset:
    """Load and merge per-variable ensemble cache files."""
    datasets = []
    for var in era5_vars:
        path = os.path.join(
            cache_dir, _ensemble_cache_filename(var, fc_start, fc_end),
        )
        print(f"[ensemble_extract] Cache hit: {os.path.basename(path)}")
        datasets.append(xr.open_dataset(path))
    return xr.merge(datasets)


def _save_ensemble_cache(
    ds: xr.Dataset,
    cache_dir: str,
    fc_start: datetime.date,
    fc_end: datetime.date,
) -> None:
    """Save each variable as a separate NetCDF cache file."""
    os.makedirs(cache_dir, exist_ok=True)
    for var in ds.data_vars:
        path = os.path.join(
            cache_dir, _ensemble_cache_filename(var, fc_start, fc_end),
        )
        ds[[var]].to_netcdf(path)
        print(f"[ensemble_extract] Saved cache: {os.path.basename(path)}")


# ===========================================================================
# Main extraction function
# ===========================================================================

def extract_ensemble(
    era5_end_date: datetime.date,
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    forecast_days: int = 14,
    cache_dir: str = "data/cache",
    variables: list[str] | None = None,
) -> xr.Dataset | None:
    """
    Fetch ECMWF IFS ensemble forecast data from Open-Meteo.

    Downloads 51-member ensemble forecasts for the same grid as ERA5,
    converts to ERA5-compatible units, and returns an xarray Dataset
    with dimensions (member, time, lat, lon).

    Parameters
    ----------
    era5_end_date : datetime.date
        Last date of ERA5 data (used only for logging context).
    grid_lats : np.ndarray
        Latitude coordinates from the ERA5 dataset.
    grid_lons : np.ndarray
        Longitude coordinates in 0-360 convention.
    forecast_days : int
        Number of forecast days (default 14).  Set to 0 to skip.
    cache_dir : str
        Directory for cache files.
    variables : list[str], optional
        ERA5 variable names to fetch.  Defaults to all.

    Returns
    -------
    xr.Dataset or None
        Dataset with dimensions (member, time, lat, lon).
        Returns None if forecast_days == 0.
    """
    if forecast_days <= 0:
        print("[ensemble_extract] forecast_days=0 — skipping ensemble fetch.")
        return None

    today = datetime.date.today()
    fc_start = today
    fc_end = today + datetime.timedelta(days=forecast_days - 1)

    # Determine variables
    if variables is None:
        era5_vars = sorted(ERA5_TO_OPENMETEO.keys())
    else:
        era5_vars = _expand_era5_vars(variables)
    openmeteo_vars = _era5_vars_to_openmeteo(era5_vars)

    print(f"[ensemble_extract] Ensemble forecast: {fc_start} -> {fc_end} "
          f"({forecast_days} days, {N_MEMBERS} members)")

    # ── Check cache ───────────────────────────────────────────────────────
    if _find_ensemble_cached(cache_dir, fc_start, fc_end, era5_vars):
        return _load_ensemble_cached(cache_dir, fc_start, fc_end, era5_vars)

    # ── Convert lon to -180:180 for the API ───────────────────────────────
    lons_180 = _lon_360_to_180(grid_lons.astype(float))

    # ── Flatten grid to (lat, lon) pairs ──────────────────────────────────
    flat_lats = []
    flat_lons = []
    for lat in grid_lats:
        for lon in lons_180:
            flat_lats.append(float(lat))
            flat_lons.append(float(lon))

    n_points = len(flat_lats)
    print(f"[ensemble_extract] Fetching ensemble for {n_points} grid points "
          f"in batches of {ENSEMBLE_BATCH_SIZE} ...")

    # ── Fetch in batches ──────────────────────────────────────────────────
    all_responses = []
    n_batches = (n_points + ENSEMBLE_BATCH_SIZE - 1) // ENSEMBLE_BATCH_SIZE

    for batch_idx in range(n_batches):
        start = batch_idx * ENSEMBLE_BATCH_SIZE
        end = min(start + ENSEMBLE_BATCH_SIZE, n_points)
        batch_lats = flat_lats[start:end]
        batch_lons = flat_lons[start:end]

        print(f"[ensemble_extract]   Batch {batch_idx + 1}/{n_batches} "
              f"({len(batch_lats)} points) ...")
        responses = _fetch_ensemble_batch(
            batch_lats, batch_lons,
            forecast_days=forecast_days,
            hourly_vars=openmeteo_vars,
        )
        all_responses.extend(responses)

        if batch_idx < n_batches - 1:
            _time.sleep(3.0)

    print(f"[ensemble_extract] All {n_batches} batches fetched. "
          f"Assembling dataset ...")

    # ── Assemble into xr.Dataset ──────────────────────────────────────────
    ds = _ensemble_responses_to_dataset(
        all_responses, grid_lats, grid_lons, era5_vars,
    )

    actual_start = pd.Timestamp(ds.time.values[0]).date()
    actual_end = pd.Timestamp(ds.time.values[-1]).date()

    # ── Save to cache ─────────────────────────────────────────────────────
    _save_ensemble_cache(ds, cache_dir, fc_start, fc_end)

    print(f"[ensemble_extract] Done. Ensemble spans "
          f"{actual_start} -> {actual_end} "
          f"({len(ds.time)} timesteps, {len(ds.member)} members)")
    return ds
