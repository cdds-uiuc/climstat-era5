"""
Step 1: Extract ERA5 reanalysis data from Google Cloud ARCO-ERA5 with local caching.

What this module does
---------------------
ERA5 is a global climate reanalysis dataset produced by ECMWF, available
publicly on Google Cloud as a Zarr store (a chunked, cloud-optimised format).
This module downloads the subset of ERA5 we need (a few variables, cropped to
Illinois) and saves it as local NetCDF files so we don't have to re-download
every time we run the pipeline.

Caching strategy
----------------
One NetCDF file is stored **per variable per year**, named:

    era5_{variable}_{start_YYYYMMDD}_{end_YYYYMMDD}.nc

Examples:
    era5_2m_temperature_20160101_20161231.nc   <- complete past year
    era5_2m_temperature_20260101_20260325.nc   <- current (incomplete) year

For complete past years the cache is used indefinitely (the data will never
change).  For the current year the data is incomplete and new hours become
available every few days, so the cache is considered stale after
CURRENT_YEAR_STALENESS_DAYS days and is re-downloaded automatically.

Key concepts for Python newcomers
----------------------------------
- **xarray.Dataset / DataArray**: Think of these as labelled N-dimensional
  arrays (like numpy arrays, but with named dimensions such as "time", "lat",
  "lon").  A Dataset holds multiple DataArrays (one per variable).
- **Zarr store**: A cloud-native file format that lets xarray read only the
  chunks it needs, without downloading the whole file.
- **Lazy vs eager loading**: When we open a Zarr store with xarray, no data
  is actually downloaded yet — xarray just reads metadata.  Data is fetched
  only when we call ``.compute()`` or ``.to_netcdf()``.
- **NetCDF (.nc)**: A standard scientific data format for multidimensional
  arrays.  We use it for the local cache files.
"""

import os
import glob
import datetime

import fsspec      # Filesystem abstraction — lets xarray open Google Cloud URLs
import xarray as xr
import numpy as np
import pandas as pd


# ── Default ERA5 variables needed by the heat-metric pipeline ──────────────
# These are the "long names" that ERA5 uses.  The pipeline currently needs:
#   - 2m_temperature          : air temperature 2 m above the surface (K)
#   - 10m_u_component_of_wind : east-west wind component at 10 m (m/s)
#   - 10m_v_component_of_wind : north-south wind component at 10 m (m/s)
#   - 2m_dewpoint_temperature : dewpoint temperature 2 m above surface (K)
#   - total_precipitation     : accumulated precip per hour (m)
DEFAULT_VARIABLES = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_dewpoint_temperature",
    "total_precipitation",
]

# ── Illinois bounding box ──────────────────────────────────────────────────
# Longitude is given in degrees east (standard -180 to 180 convention).
# Internally this is converted to 0-360 when querying ERA5, which uses that
# convention (e.g. -90°W becomes 270°E).
DEFAULT_LON_BOUNDS = (-91.75, -86.75)   # western, eastern edge
DEFAULT_LAT_BOUNDS = (37.0, 42.75)      # southern, northern edge

# ── Google Cloud ARCO-ERA5 Zarr store ──────────────────────────────────────
# This is the public URL for the Analysis-Ready, Cloud-Optimised (ARCO) ERA5
# dataset hosted by Google.  It contains hourly global data at 0.25° resolution
# from 1959 to near-present.
ZARR_URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

# ── Current-year cache freshness ──────────────────────────────────────────
# If the cached file for the current year is more than this many days old
# (based on the end date in its filename), it is deleted and re-downloaded
# to pick up newly available hours.
CURRENT_YEAR_STALENESS_DAYS = 7


# ===========================================================================
# Filename helpers
# ===========================================================================

def _sanitize_var(var: str) -> str:
    """
    Convert an ERA5 variable long name to a filesystem-safe string.

    Example: "2m_dewpoint_temperature" stays the same (already safe),
    but any spaces or hyphens would be replaced with underscores.
    """
    return var.replace(" ", "_").replace("-", "_")


def _cache_filename(variable: str,
                    date_start: datetime.date,
                    date_end: datetime.date) -> str:
    """
    Build the canonical cache filename for one (variable, date range) pair.

    The filename encodes everything we need to know about the file's
    contents: which variable, and the first/last day of data it holds.

    Format: era5_{variable}_{YYYYMMDD}_{YYYYMMDD}.nc
    Example: era5_2m_temperature_20200101_20201231.nc
    """
    return (
        f"era5_{_sanitize_var(variable)}"
        f"_{date_start.strftime('%Y%m%d')}"
        f"_{date_end.strftime('%Y%m%d')}.nc"
    )


def _parse_cache_dates(path: str) -> tuple[datetime.date, datetime.date]:
    """
    Extract the start and end dates encoded in a cache filename.

    We split the filename on underscores and take the last two tokens
    (which are always the 8-digit start and end dates), regardless of
    how many underscores the variable name itself contains.

    Example:
        "era5_2m_dewpoint_temperature_20200101_20201231.nc"
        -> (datetime.date(2020, 1, 1), datetime.date(2020, 12, 31))
    """
    stem = os.path.splitext(os.path.basename(path))[0]  # strip directory + .nc
    parts = stem.split("_")
    start_str, end_str = parts[-2], parts[-1]
    start = datetime.date(int(start_str[:4]), int(start_str[4:6]), int(start_str[6:]))
    end   = datetime.date(int(end_str[:4]),   int(end_str[4:6]),   int(end_str[6:]))
    return start, end


def _find_cached(cache_dir: str, variable: str, year: int) -> str | None:
    """
    Look for a usable cached file for a given (variable, year) pair.

    Returns the file path if a valid cache exists, or None if we need
    to download.

    Rules:
      - For a past complete year (e.g. 2023 when today is 2026), any file
        whose name starts with "era5_{var}_{year}0101_" is accepted.  The
        data for a completed year will never change.
      - For the current year, the file must also be "fresh enough": its
        encoded end date must be within CURRENT_YEAR_STALENESS_DAYS of
        today.  If not, the stale file is deleted so a new one can be
        downloaded with more recent data.
    """
    today = datetime.date.today()
    var_safe = _sanitize_var(variable)

    # Use a glob pattern to find files like era5_2m_temperature_20200101_*.nc
    pattern = os.path.join(cache_dir, f"era5_{var_safe}_{year}0101_*.nc")
    matches = glob.glob(pattern)

    if not matches:
        return None

    cached_path = matches[0]
    _, cached_end = _parse_cache_dates(cached_path)

    if year < today.year:
        # Complete past year — the data is final, cache is always valid
        return cached_path

    # Current (incomplete) year — check whether the cache is fresh enough
    days_stale = (today - cached_end).days
    if days_stale <= CURRENT_YEAR_STALENESS_DAYS:
        return cached_path

    # Cache is too old — delete it so _download_one will create a fresh one
    print(
        f"[era5_extract] {os.path.basename(cached_path)} is {days_stale} days stale "
        f"(threshold: {CURRENT_YEAR_STALENESS_DAYS}). Re-downloading ..."
    )
    os.remove(cached_path)
    return None


# ===========================================================================
# Single-variable, single-year download
# ===========================================================================

def _download_one(
    reanalysis: xr.Dataset,
    variable: str,
    year: int,
    lon_bounds: tuple[float, float],
    lat_bounds: tuple[float, float],
    cache_dir: str,
) -> xr.Dataset:
    """
    Download one ERA5 variable for one year, crop it to the Illinois domain,
    save it to a local NetCDF cache file, and return the Dataset.

    How it works, step by step:
      1. Determine the date range: Jan 1 - Dec 31 for past years, or
         Jan 1 - today for the current year.
      2. Select just this variable and time range from the remote Zarr
         store.  (This is still lazy — no data has been downloaded yet.)
      3. Crop to the Illinois longitude/latitude bounding box.
      4. Call .compute() to actually download the data from Google Cloud.
      5. Save the result to a local .nc file whose name encodes the
         variable and the actual first/last date of data in the file.

    Parameters
    ----------
    reanalysis : xr.Dataset
        The full remote ERA5 Zarr store, already opened (lazy).
    variable : str
        ERA5 variable long name, e.g. "2m_temperature".
    year : int
        Calendar year to download.
    lon_bounds, lat_bounds : tuple
        Geographic bounding box (see module-level constants).
    cache_dir : str
        Local directory to save the NetCDF file into.
    """
    today = datetime.date.today()
    date_start = datetime.date(year, 1, 1)
    # For past years, download the full year.  For the current year,
    # download up to today (ERA5 may actually lag ~5 days behind, so the
    # actual last timestamp may be a few days before today).
    date_end   = datetime.date(year, 12, 31) if year < today.year else today

    print(f"[era5_extract] Downloading '{variable}' {year} "
          f"({date_start} -> {date_end}) ...")

    # Crop to the Illinois bounding box.
    # ERA5 uses 0-360 longitudes, so we convert our -180..180 bounds with % 360.
    # Example: -91.75 % 360 = 268.25
    lon_min = lon_bounds[0] % 360
    lon_max = lon_bounds[1] % 360
    lat_min, lat_max = lat_bounds

    # ── Download in monthly batches ───────────────────────────────────
    # The ARCO-ERA5 Zarr store is chunked as {time: 1, latitude: 721,
    # longitude: 1440} — each chunk is one timestep covering the entire
    # globe.  Since Illinois is only ~480 of ~1M grid points per chunk,
    # we unavoidably download ~2000x more data than we keep.
    #
    # Downloading month-by-month (rather than a whole year at once) keeps
    # memory bounded and gives progress feedback.
    monthly_pieces = []
    # Determine month range: all 12 for past years, up to current month otherwise
    last_month = 12 if year < today.year else today.month
    for month in range(1, last_month + 1):
        m_start = f"{year}-{month:02d}-01"
        # For the last month of the current year, cap at today
        if year == today.year and month == today.month:
            m_end = str(today)
        else:
            # Last day of the month: go to 1st of next month minus 1 day
            if month == 12:
                m_end = f"{year}-12-31"
            else:
                m_end = str(datetime.date(year, month + 1, 1) - datetime.timedelta(days=1))

        piece = reanalysis[[variable]].sel(
            time=slice(m_start, m_end),
            longitude=slice(lon_min, lon_max),
            latitude=slice(lat_max, lat_min),  # ERA5 latitudes are descending (90 -> -90)
        ).compute()
        monthly_pieces.append(piece)
        print(f"[era5_extract]   month {month:02d} downloaded "
              f"({len(piece.time)} timesteps)")

    # Concatenate all months into one yearly dataset
    illinois = xr.concat(monthly_pieces, dim="time")
    illinois = illinois.rename({"longitude": "lon", "latitude": "lat"})

    # Read the actual last timestamp in the data.  For the current year,
    # this tells us exactly how far ERA5 data extends (which we encode in
    # the cache filename).
    actual_end = pd.Timestamp(illinois.time.values[-1]).date()

    cache_path = os.path.join(
        cache_dir, _cache_filename(variable, date_start, actual_end)
    )
    print(f"[era5_extract] Saving to {os.path.basename(cache_path)} ...")
    illinois.to_netcdf(cache_path)
    print(f"[era5_extract] Saved  ({date_start} -> {actual_end}).")

    return illinois


# ===========================================================================
# Main extraction function  (this is what the notebook calls)
# ===========================================================================

def extract_era5(
    variables: list[str] | None = None,
    year_start: int = 2016,
    year_end: int = 2025,
    lon_bounds: tuple[float, float] = DEFAULT_LON_BOUNDS,
    lat_bounds: tuple[float, float] = DEFAULT_LAT_BOUNDS,
    cache_dir: str = "data/cache",
) -> xr.Dataset:
    """
    Extract ERA5 variables over Illinois from Google Cloud, one file per
    variable per year, and return a single merged xarray Dataset.

    This is the main function that the wrapper notebook calls.  It:
      1. Checks the local cache for each (variable, year) pair.
      2. Opens the remote Zarr store only if at least one pair is missing.
      3. Downloads any missing pairs via _download_one().
      4. Concatenates each variable's yearly files along the time axis.
      5. Merges all variables into one Dataset and returns it.

    On the first run this downloads everything (can take ~5 min per variable
    per year).  On subsequent runs, cached past years load instantly from
    disk and only the current year may need refreshing.

    Parameters
    ----------
    variables : list[str], optional
        ERA5 variable long names.  Defaults to the five variables needed
        for heat-metric computation (see DEFAULT_VARIABLES).
    year_start, year_end : int
        Inclusive year range to extract.
    lon_bounds : (float, float)
        Western and eastern longitude bounds (degrees, -180 to 180).
    lat_bounds : (float, float)
        Southern and northern latitude bounds (degrees north).
    cache_dir : str
        Directory where per-variable per-year NetCDF files are stored.

    Returns
    -------
    xr.Dataset
        Hourly gridded ERA5 data for the full requested period, with all
        requested variables, cropped to the Illinois domain.
        Dimensions: (time, lat, lon).
    """
    if variables is None:
        variables = DEFAULT_VARIABLES

    # Create the cache directory if it doesn't exist yet
    os.makedirs(cache_dir, exist_ok=True)
    years = list(range(year_start, year_end + 1))

    # ── Figure out which (variable, year) pairs are missing from cache ────
    # This builds a list of pairs we still need to download.
    to_download = [
        (var, yr)
        for var in variables
        for yr in years
        if _find_cached(cache_dir, var, yr) is None
    ]

    # ── Open the remote Zarr store only if we actually need to download ───
    # Opening the store is fast (~15 s) but requires internet, so we skip
    # it entirely when everything is already cached.
    reanalysis = None
    if to_download:
        print("[era5_extract] Opening Google Cloud ARCO-ERA5 Zarr store ...")
        # chunks=None avoids dask overhead.  The native Zarr chunks are
        # {time: 1, lat: 721, lon: 1440} (one timestep, full globe), so
        # there is no benefit to dask parallelism here — each .compute()
        # call in _download_one fetches one month at a time.
        reanalysis = xr.open_zarr(
            ZARR_URL,
            chunks=None,
            storage_options=dict(token="anon"),
        )
        print(f"[era5_extract] {len(to_download)} file(s) to download.")

    # ── Load or download each (variable, year) ────────────────────────────
    # We loop over variables first, then years within each variable.
    # For each variable, we collect a list of yearly Datasets, then
    # concatenate them along the time dimension to get one continuous
    # time series for that variable.
    var_datasets = []
    for variable in variables:
        year_slices = []
        for yr in years:
            cached = _find_cached(cache_dir, variable, yr)
            if cached:
                print(f"[era5_extract] Cache hit: {os.path.basename(cached)}")
                year_slices.append(xr.open_dataset(cached))
            else:
                ds = _download_one(reanalysis, variable, yr,
                                   lon_bounds, lat_bounds, cache_dir)
                year_slices.append(ds)

        # xr.concat joins yearly slices into one continuous time series
        # for this variable
        var_datasets.append(xr.concat(year_slices, dim="time"))

    # xr.merge combines the per-variable Datasets into a single Dataset
    # containing all variables side by side (they share the same
    # time/lat/lon coordinates)
    merged = xr.merge(var_datasets)
    print(f"[era5_extract] Done. Dataset spans "
          f"{pd.Timestamp(merged.time.values[0]).date()} -> "
          f"{pd.Timestamp(merged.time.values[-1]).date()}")
    return merged
