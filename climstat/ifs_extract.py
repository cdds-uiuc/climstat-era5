"""
Step 1b: Extract IFS forecast/analysis data from Open-Meteo to fill the ERA5 gap.

What this module does
---------------------
ERA5 reanalysis data lags ~5-7 days behind the present.  This module downloads
ECMWF IFS (Integrated Forecasting System) data from the Open-Meteo API to fill
that gap, providing near-real-time climate data on the same grid as ERA5.

The data is fetched for the exact same grid points as ERA5, converted to the
same units and variable names, and returned as an xarray Dataset with identical
structure.  This means all downstream pipeline functions (compute_metrics,
daily_summary, aggregate_to_counties, etc.) work on IFS data without
modification.

Data source
-----------
- Open-Meteo ECMWF API: https://open-meteo.com/en/docs/ecmwf-api
- ECMWF IFS model at 0.25° resolution
- Updated ~4 times per day, but we only download up to the previous day.
- Free tier: 10,000 API calls/day (we use ~9 calls per fetch)

Caching strategy
----------------
IFS data is cached as one NetCDF file per variable, matching the ERA5 pattern:

    ifs_{variable}_{start_YYYYMMDD}_{end_YYYYMMDD}.nc

The cache is refreshed if the file is older than IFS_CACHE_STALENESS_HOURS
hours, or if the requested date range doesn't match any existing cache file.
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


# ── Open-Meteo ECMWF API endpoint ────────────────────────────────────────
OPENMETEO_ECMWF_URL = "https://api.open-meteo.com/v1/ecmwf"

# ── Mapping from ERA5 variable names to Open-Meteo API parameters ─────────
# Wind is special: Open-Meteo provides speed + direction, which we split
# into u/v components.  If either wind component is requested, both
# speed and direction must be fetched.
ERA5_TO_OPENMETEO = {
    "2m_temperature":           ["temperature_2m"],
    "2m_dewpoint_temperature":  ["dew_point_2m"],
    "10m_u_component_of_wind":  ["wind_speed_10m", "wind_direction_10m"],
    "10m_v_component_of_wind":  ["wind_speed_10m", "wind_direction_10m"],
    "total_precipitation":      ["precipitation"],
}

# All possible Open-Meteo hourly variables (used as fallback when no
# specific variables are requested)
ALL_OPENMETEO_HOURLY_VARS = [
    "temperature_2m",
    "dew_point_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
]

# ── Batching ──────────────────────────────────────────────────────────────
# Maximum number of (lat, lon) pairs per API call.  The Open-Meteo API
# accepts comma-separated coordinate lists, but very long URLs can fail.
# 50 locations keeps the URL well under typical limits.
BATCH_SIZE = 50

# ── Cache freshness ──────────────────────────────────────────────────────
# IFS forecasts update ~4 times per day, so cached data goes stale quickly.
IFS_CACHE_STALENESS_HOURS = 6

# ── Retry settings for API calls ─────────────────────────────────────────
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2


# ===========================================================================
# Grid helpers
# ===========================================================================

def _lon_360_to_180(lon: np.ndarray) -> np.ndarray:
    """Convert longitudes from 0-360 to -180:180 convention."""
    return np.where(lon > 180, lon - 360, lon)


def _lon_180_to_360(lon: np.ndarray) -> np.ndarray:
    """Convert longitudes from -180:180 to 0-360 convention."""
    return lon % 360


# ===========================================================================
# API fetching
# ===========================================================================

def _fetch_batch(
    lats: list[float],
    lons: list[float],
    past_days: int,
    forecast_days: int = 0,
    hourly_vars: list[str] | None = None,
) -> list[dict]:
    """
    Fetch IFS data from Open-Meteo for a batch of locations.

    Parameters
    ----------
    lats, lons : list[float]
        Coordinate pairs in -180:180 convention.  Must be same length.
    past_days : int
        Number of past days to include.
    forecast_days : int
        Number of forecast days to include (0 = analysis only).
    hourly_vars : list[str], optional
        Open-Meteo hourly variable names to request.  Defaults to all.

    Returns
    -------
    list[dict]
        One dict per location, each containing 'hourly' with time series.
    """
    if hourly_vars is None:
        hourly_vars = ALL_OPENMETEO_HOURLY_VARS
    params = {
        "latitude": ",".join(f"{lat:.2f}" for lat in lats),
        "longitude": ",".join(f"{lon:.2f}" for lon in lons),
        "hourly": ",".join(hourly_vars),
        "wind_speed_unit": "ms",
        "past_days": str(past_days),
        "forecast_days": str(forecast_days),
    }
    url = OPENMETEO_ECMWF_URL + "?" + urllib.parse.urlencode(params)

    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            # Single location returns a dict; multiple returns a list
            if isinstance(data, dict):
                # Check for API error
                if "error" in data and "hourly" not in data:
                    raise RuntimeError(f"Open-Meteo API error: {data.get('reason', data)}")
                return [data]
            return data

        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_BACKOFF_SECONDS * (2 ** attempt)
                print(f"[ifs_extract]   Retry {attempt + 1}/{MAX_RETRIES} "
                      f"after {wait}s ({e})")
                _time.sleep(wait)
            else:
                raise RuntimeError(
                    f"Open-Meteo API request failed after {MAX_RETRIES} attempts: {e}"
                ) from e
    return []  # unreachable, but satisfies type checkers


def _speed_direction_to_uv(
    speed: np.ndarray,
    direction_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert wind speed and meteorological direction to u/v components.

    Meteorological convention: direction is where wind comes FROM,
    measured clockwise from north.

        u = -speed * sin(direction)   (eastward component)
        v = -speed * cos(direction)   (northward component)

    Parameters
    ----------
    speed : array
        Wind speed in m/s.
    direction_deg : array
        Wind direction in degrees (meteorological convention).

    Returns
    -------
    u, v : arrays
        Eastward and northward wind components in m/s.
    """
    direction_rad = np.deg2rad(direction_deg)
    u = -speed * np.sin(direction_rad)
    v = -speed * np.cos(direction_rad)
    return u, v


# ===========================================================================
# Response parsing
# ===========================================================================

def _responses_to_dataset(
    all_responses: list[dict],
    grid_lats: np.ndarray,
    grid_lons_360: np.ndarray,
    era5_vars: list[str],
) -> xr.Dataset:
    """
    Convert flat per-point Open-Meteo responses into a gridded xr.Dataset
    matching ERA5 format.

    Parameters
    ----------
    all_responses : list[dict]
        Flat list of API responses, ordered lat-major (for each lat, iterate
        over all lons).
    grid_lats : array
        Latitude coordinates (descending order), matching ERA5.
    grid_lons_360 : array
        Longitude coordinates in 0-360 convention, matching ERA5.
    era5_vars : list[str]
        ERA5 variable names to include in the output Dataset.

    Returns
    -------
    xr.Dataset
        Dimensions: (time, lat, lon).
        Only contains the requested ERA5 variables.
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

    def to_grid(flat_arr):
        return flat_arr.reshape(n_lat, n_lon, n_times).transpose(2, 0, 1)

    era5_var_set = set(era5_vars)
    need_temp = "2m_temperature" in era5_var_set
    need_dewp = "2m_dewpoint_temperature" in era5_var_set
    need_wind = ("10m_u_component_of_wind" in era5_var_set or
                 "10m_v_component_of_wind" in era5_var_set)
    need_precip = "total_precipitation" in era5_var_set

    # ── Collect raw data into flat arrays (n_points, n_times) ────────────
    data_vars = {}

    if need_temp:
        temp_c = np.full((n_points, n_times), np.nan)
        for i, resp in enumerate(all_responses):
            temp_c[i] = _safe_array(resp["hourly"]["temperature_2m"], n_times)
        data_vars["2m_temperature"] = (["time", "lat", "lon"], to_grid(temp_c + 273.15))

    if need_dewp:
        dewp_c = np.full((n_points, n_times), np.nan)
        for i, resp in enumerate(all_responses):
            dewp_c[i] = _safe_array(resp["hourly"]["dew_point_2m"], n_times)
        data_vars["2m_dewpoint_temperature"] = (["time", "lat", "lon"], to_grid(dewp_c + 273.15))

    if need_wind:
        wspd = np.full((n_points, n_times), np.nan)
        wdir = np.full((n_points, n_times), np.nan)
        for i, resp in enumerate(all_responses):
            wspd[i] = _safe_array(resp["hourly"]["wind_speed_10m"], n_times)
            wdir[i] = _safe_array(resp["hourly"]["wind_direction_10m"], n_times)
        u10, v10 = _speed_direction_to_uv(wspd, wdir)
        data_vars["10m_u_component_of_wind"] = (["time", "lat", "lon"], to_grid(u10))
        data_vars["10m_v_component_of_wind"] = (["time", "lat", "lon"], to_grid(v10))

    if need_precip:
        precip_mm = np.full((n_points, n_times), np.nan)
        for i, resp in enumerate(all_responses):
            precip_mm[i] = _safe_array(resp["hourly"]["precipitation"], n_times)
        data_vars["total_precipitation"] = (["time", "lat", "lon"], to_grid(precip_mm / 1000.0))

    ds = xr.Dataset(
        data_vars,
        coords={"time": times, "lat": grid_lats, "lon": grid_lons_360},
    )
    return ds


def _safe_array(values: list, expected_len: int) -> np.ndarray:
    """Convert a list (possibly with None values) to a float array with NaN."""
    arr = np.array(values, dtype=float)
    if len(arr) != expected_len:
        raise ValueError(f"Expected {expected_len} values, got {len(arr)}")
    return arr


# ===========================================================================
# Cache management
# ===========================================================================

def _ifs_var_cache_filename(
    variable: str,
    date_start: datetime.date,
    date_end: datetime.date,
) -> str:
    """Build the cache filename for one IFS variable.

    Format mirrors ERA5: ifs_{variable}_{YYYYMMDD}_{YYYYMMDD}.nc
    """
    var_safe = variable.replace(" ", "_").replace("-", "_")
    return (
        f"ifs_{var_safe}"
        f"_{date_start.strftime('%Y%m%d')}"
        f"_{date_end.strftime('%Y%m%d')}.nc"
    )


def _expand_era5_vars(era5_vars: list[str]) -> list[str]:
    """Expand ERA5 variable list to include both wind components if either is present.

    Wind u/v are derived from the same Open-Meteo fields (speed + direction),
    so both must be fetched and cached together.
    """
    var_set = set(era5_vars)
    if "10m_u_component_of_wind" in var_set or "10m_v_component_of_wind" in var_set:
        var_set.add("10m_u_component_of_wind")
        var_set.add("10m_v_component_of_wind")
    return sorted(var_set)


def _era5_vars_to_openmeteo(era5_vars: list[str]) -> list[str]:
    """Convert ERA5 variable names to the Open-Meteo API parameter names."""
    om_vars = set()
    for v in era5_vars:
        if v in ERA5_TO_OPENMETEO:
            om_vars.update(ERA5_TO_OPENMETEO[v])
    return sorted(om_vars)


def _find_ifs_cached(
    cache_dir: str,
    ifs_start: datetime.date,
    ifs_end: datetime.date,
    era5_vars: list[str],
) -> bool:
    """
    Check whether all per-variable IFS cache files exist and are fresh.

    Returns True if every variable file exists and is younger than
    IFS_CACHE_STALENESS_HOURS, False otherwise.
    """
    for var in era5_vars:
        path = os.path.join(
            cache_dir, _ifs_var_cache_filename(var, ifs_start, ifs_end),
        )
        if not os.path.exists(path):
            return False

        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(path))
        age_hours = (datetime.datetime.now() - mtime).total_seconds() / 3600
        if age_hours > IFS_CACHE_STALENESS_HOURS:
            print(f"[ifs_extract] Cache {os.path.basename(path)} is "
                  f"{age_hours:.1f}h old — will re-download.")
            return False

    return True


def _load_ifs_cached(
    cache_dir: str,
    ifs_start: datetime.date,
    ifs_end: datetime.date,
    era5_vars: list[str],
) -> xr.Dataset:
    """Load and merge per-variable IFS cache files into a single Dataset."""
    datasets = []
    for var in era5_vars:
        path = os.path.join(
            cache_dir, _ifs_var_cache_filename(var, ifs_start, ifs_end),
        )
        print(f"[ifs_extract] Cache hit: {os.path.basename(path)}")
        datasets.append(xr.open_dataset(path))
    return xr.merge(datasets)


def _save_ifs_cache(
    ds: xr.Dataset,
    cache_dir: str,
    ifs_start: datetime.date,
    ifs_end: datetime.date,
) -> None:
    """Save each variable in the Dataset as a separate NetCDF cache file."""
    os.makedirs(cache_dir, exist_ok=True)
    for var in ds.data_vars:
        path = os.path.join(
            cache_dir, _ifs_var_cache_filename(var, ifs_start, ifs_end),
        )
        ds[[var]].to_netcdf(path)
        print(f"[ifs_extract] Saved cache: {os.path.basename(path)}")


def get_era5_end_date(
    cache_dir: str = "data/cache",
    variables: list[str] | None = None,
) -> datetime.date:
    """
    Determine the last date of ERA5 data by inspecting cache filenames.

    Scans for ERA5 cache files and returns the latest end date encoded in
    the filenames.  This avoids loading any NetCDF data.

    Parameters
    ----------
    cache_dir : str
        Directory containing ERA5 cache files.
    variables : list[str], optional
        ERA5 variable names to check.  If None, checks all era5_*.nc files.

    Returns
    -------
    datetime.date
        The latest end date found across all cache files.

    Raises
    ------
    FileNotFoundError
        If no ERA5 cache files exist.
    """
    pattern = os.path.join(cache_dir, "era5_*.nc")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No ERA5 cache files found in {cache_dir}")

    latest_end = datetime.date.min
    for path in files:
        stem = os.path.splitext(os.path.basename(path))[0]
        parts = stem.split("_")
        end_str = parts[-1]
        try:
            end_date = datetime.date(
                int(end_str[:4]), int(end_str[4:6]), int(end_str[6:])
            )
            if end_date > latest_end:
                latest_end = end_date
        except (ValueError, IndexError):
            continue

    if latest_end == datetime.date.min:
        raise FileNotFoundError(f"Could not parse dates from ERA5 cache in {cache_dir}")

    return latest_end


# ===========================================================================
# Main extraction function
# ===========================================================================

def extract_ifs(
    era5_end_date: datetime.date,
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    forecast_days: int = 0,
    cache_dir: str = "data/cache",
    variables: list[str] | None = None,
) -> xr.Dataset | None:
    """
    Fetch ECMWF IFS data from Open-Meteo to fill the gap after ERA5.

    Downloads IFS analysis/forecast data for the same grid points as ERA5,
    converts to ERA5-compatible units and variable names, and returns an
    xarray Dataset with identical structure.

    Parameters
    ----------
    era5_end_date : datetime.date
        The last date present in the ERA5 data.  IFS data will start the
        day after this date.
    grid_lats : np.ndarray
        Latitude coordinates from the ERA5 dataset (ds.lat.values).
    grid_lons : np.ndarray
        Longitude coordinates from the ERA5 dataset (ds.lon.values).
        Expected in 0-360 convention.
    forecast_days : int
        Number of days of IFS forecast to include beyond yesterday (default 0).
    cache_dir : str
        Directory for the IFS cache files (same as ERA5 cache dir).
    variables : list[str], optional
        ERA5 variable names to fetch (e.g. ["2m_temperature", "2m_dewpoint_temperature"]).
        Defaults to all variables in ERA5_TO_OPENMETEO.

    Returns
    -------
    xr.Dataset or None
        Dataset with dimensions (time, lat, lon) and ERA5-compatible
        variable names and units.  Returns None if there is no gap to fill
        (ERA5 is already up to date).
    """
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)
    ifs_start = era5_end_date + datetime.timedelta(days=1)

    # If ERA5 is already up to date, there's no gap to fill
    if ifs_start > yesterday and forecast_days == 0:
        print("[ifs_extract] ERA5 is up to date — no IFS data needed.")
        return None

    # Determine which variables to fetch
    if variables is None:
        era5_vars = sorted(ERA5_TO_OPENMETEO.keys())
    else:
        era5_vars = _expand_era5_vars(variables)
    openmeteo_vars = _era5_vars_to_openmeteo(era5_vars)

    # Only request through yesterday to avoid incomplete data for today.
    # Today's data is still being produced and would cause cache misses
    # on every run.
    ifs_end = yesterday + datetime.timedelta(days=forecast_days)

    # past_days tells Open-Meteo how many days before today to include.
    # We need at least 2 to ensure the API returns data covering ifs_start,
    # even when ifs_start == today (past_days=0 can return empty results).
    past_days = max(2, (today - ifs_start).days + 1)

    print(f"[ifs_extract] Filling gap: {ifs_start} -> {ifs_end} "
          f"(past_days={past_days}, forecast_days={forecast_days})")

    # ── Check cache ───────────────────────────────────────────────────────
    if _find_ifs_cached(cache_dir, ifs_start, ifs_end, era5_vars):
        return _load_ifs_cached(cache_dir, ifs_start, ifs_end, era5_vars)

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
    print(f"[ifs_extract] Fetching IFS data for {n_points} grid points "
          f"in batches of {BATCH_SIZE} ...")

    # ── Fetch in batches ──────────────────────────────────────────────────
    all_responses = []
    n_batches = (n_points + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(n_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, n_points)
        batch_lats = flat_lats[start:end]
        batch_lons = flat_lons[start:end]

        print(f"[ifs_extract]   Batch {batch_idx + 1}/{n_batches} "
              f"({len(batch_lats)} points) ...")
        responses = _fetch_batch(batch_lats, batch_lons,
                                 past_days=past_days,
                                 forecast_days=forecast_days,
                                 hourly_vars=openmeteo_vars)
        all_responses.extend(responses)

        # Small delay between batches to be a good API citizen
        if batch_idx < n_batches - 1:
            _time.sleep(0.5)

    print(f"[ifs_extract] All {n_batches} batches fetched. Assembling dataset ...")

    # ── Assemble into xr.Dataset ──────────────────────────────────────────
    ds = _responses_to_dataset(all_responses, grid_lats, grid_lons, era5_vars)

    # ── Trim to the IFS period (remove any data before ifs_start) ─────────
    ifs_start_ts = pd.Timestamp(ifs_start)
    ds = ds.sel(time=slice(ifs_start_ts, None))

    if len(ds.time) == 0:
        print("[ifs_extract] Warning: no IFS data available for the requested period.")
        return None

    actual_start = pd.Timestamp(ds.time.values[0]).date()
    actual_end = pd.Timestamp(ds.time.values[-1]).date()

    # ── Save to cache (one file per variable) ─────────────────────────────
    # Use the requested date range (ifs_start, ifs_end) for the filename,
    # not the actual data range.  This ensures cache lookups match on the
    # next run (the requested range is deterministic; the actual range may
    # be shorter if Open-Meteo doesn't yet have yesterday's data).
    _save_ifs_cache(ds, cache_dir, ifs_start, ifs_end)

    print(f"[ifs_extract] Done. IFS dataset spans "
          f"{actual_start} -> {actual_end} "
          f"({len(ds.time)} timesteps)")
    return ds
