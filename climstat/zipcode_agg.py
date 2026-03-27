"""
Step 4b: Aggregate gridded ERA5-derived data to Illinois ZIP-code (ZCTA) level.

What this module does
---------------------
This is the ZIP-code counterpart to ``county_agg.py``.  It maps every
Illinois ZCTA (ZIP Code Tabulation Area) to its nearest ERA5 grid point
and averages the gridded summary statistics within each ZCTA.

Why nearest-centroid instead of point-in-polygon?
-------------------------------------------------
ERA5 grid spacing is 0.25° (~28 km).  Most Illinois ZCTAs are smaller than
a single grid cell, so the majority would contain zero grid points under a
traditional spatial join.  Instead we compute each ZCTA's centroid and find
the closest ERA5 grid point using ``sjoin_nearest``.  This guarantees every
ZCTA gets a value.

How Illinois ZCTAs are selected
-------------------------------
ZCTA shapefiles have no state FIPS column.  We filter in two steps:

1. Load only ZCTAs whose bounding box overlaps Illinois (fast, done at
   file-read time via the ``bbox`` parameter).
2. Keep only those that spatially intersect the union of all Illinois
   county polygons (precise, removes neighboring-state ZCTAs that only
   touched the bounding box).

Key terms
---------
- **ZCTA**: ZIP Code Tabulation Area — a Census-defined polygon that
  approximates a US Postal Service ZIP code delivery area.
- **ZCTA5CE20**: The 5-digit ZIP code string stored in the shapefile.
"""

import os

import pandas as pd
import xarray as xr
import geopandas as gpd

from .county_agg import _build_grid_points, _ensure_lon_180, IL_STATEFP


# ── Default shapefile paths ──────────────────────────────────────────────
DEFAULT_ZCTA_SHAPEFILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data", "shapefiles", "zipcodes", "tl_2025_us_zcta520.shp",
)

DEFAULT_COUNTY_SHAPEFILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data", "shapefiles", "county", "tl_2025_us_county.shp",
)


def _load_illinois_zctas(
    zcta_shapefile: str,
    county_shapefile: str,
) -> gpd.GeoDataFrame:
    """
    Load ZCTA boundaries that intersect Illinois.

    1. Build an Illinois outline from the county shapefile (using
       ``where`` to load only IL counties).
    2. Load ZCTAs that intersect the IL outline using the ``mask``
       parameter, which pushes the spatial filter into the reader
       (GDAL handles the spatial index internally, replacing the old
       bbox + Python-side ``.intersects()`` approach).
    3. Only the ``ZCTA5CE20`` column (plus geometry) is read via the
       ``columns`` parameter.
    """
    # Build Illinois outline from counties
    il_counties = gpd.read_file(
        county_shapefile,
        where=f"STATEFP = '{IL_STATEFP}'",
        columns=["STATEFP"],
    )
    if il_counties.crs is None or il_counties.crs.to_epsg() != 4326:
        il_counties = il_counties.to_crs("EPSG:4326")
    il_outline = il_counties.union_all()

    # Load only ZCTAs that intersect the IL outline, reading only the
    # ZIP code identifier column (geometry is always included)
    zctas = gpd.read_file(
        zcta_shapefile,
        mask=il_outline,
        columns=["ZCTA5CE20"],
    )
    if zctas.crs is None or zctas.crs.to_epsg() != 4326:
        zctas = zctas.to_crs("EPSG:4326")

    return zctas.reset_index(drop=True)


def build_zcta_mapping(
    ds: xr.Dataset,
    zcta_shapefile: str | None = None,
    county_shapefile: str | None = None,
) -> pd.DataFrame:
    """
    Precompute the ZCTA-to-grid-point mapping.

    This loads the shapefiles and performs the spatial join once.  The
    returned DataFrame can be passed to ``aggregate_to_zipcodes()`` via
    the ``zcta_mapping`` parameter to avoid repeated I/O.

    Parameters
    ----------
    ds : xr.Dataset
        Any gridded dataset with the same lat/lon grid as the pipeline
        data (used to build grid points).
    zcta_shapefile : str, optional
    county_shapefile : str, optional

    Returns
    -------
    pd.DataFrame
        Columns: ZCTA5CE20, lat, lon — one row per ZCTA.
    """
    if zcta_shapefile is None:
        zcta_shapefile = DEFAULT_ZCTA_SHAPEFILE
    if county_shapefile is None:
        county_shapefile = DEFAULT_COUNTY_SHAPEFILE

    ds = _ensure_lon_180(ds)

    print("[zipcode_agg] Loading Illinois ZCTAs ...")
    zctas = _load_illinois_zctas(zcta_shapefile, county_shapefile)
    print(f"[zipcode_agg] {len(zctas)} ZCTAs intersecting Illinois")

    points_gdf = _build_grid_points(ds)

    print("[zipcode_agg] Assigning ZCTAs to nearest grid points ...")
    centroids = zctas[["ZCTA5CE20", "geometry"]].copy()
    centroids["geometry"] = zctas.geometry.centroid

    nearest = gpd.sjoin_nearest(
        centroids, points_gdf, how="left",
    )[["ZCTA5CE20", "lat", "lon"]]

    # Guard against duplicate rows from equidistant grid points
    nearest = nearest.drop_duplicates(subset="ZCTA5CE20")

    n_grid = nearest[["lat", "lon"]].drop_duplicates().shape[0]
    print(f"[zipcode_agg] {len(nearest)} ZCTAs mapped to {n_grid} unique grid points")

    return nearest


def aggregate_to_zipcodes(
    ds: xr.Dataset,
    zcta_shapefile: str | None = None,
    county_shapefile: str | None = None,
    zcta_mapping: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Aggregate a gridded xr.Dataset to Illinois ZCTA (ZIP-code) level means.

    Each ZCTA is assigned to its nearest ERA5 grid point (by centroid
    distance).  If multiple ZCTAs share the same nearest grid point, they
    all receive the same values.

    Parameters
    ----------
    ds : xr.Dataset
        Gridded data with dimensions (time, lat, lon) or (lat, lon).
    zcta_shapefile : str, optional
        Path to US ZCTA shapefile.  Defaults to bundled file.
    county_shapefile : str, optional
        Path to US county shapefile (used to derive the Illinois outline
        for filtering ZCTAs).  Defaults to bundled file.
    zcta_mapping : pd.DataFrame, optional
        Precomputed ZCTA-to-grid-point mapping (from ``build_zcta_mapping``).
        If provided, shapefile arguments are ignored and no I/O is performed.

    Returns
    -------
    pd.DataFrame
        Columns: ZCTA5CE20, [time if present], plus all data variables.
    """
    # Step 1: fix longitude convention
    ds = _ensure_lon_180(ds)

    # Step 2: get the ZCTA-to-grid-point mapping
    if zcta_mapping is not None:
        nearest = zcta_mapping
    else:
        nearest = build_zcta_mapping(ds, zcta_shapefile, county_shapefile)

    # Step 3: convert xarray to DataFrame and merge ZCTA labels
    print("[zipcode_agg] Merging with full time-series data ...")
    df = ds.to_dataframe().reset_index()
    result = df.merge(nearest, on=["lat", "lon"], how="inner")

    # Step 4: group by ZCTA (and time if present) and average
    has_time = "time" in result.columns
    group_cols = ["ZCTA5CE20"]
    if has_time:
        group_cols.append("time")

    exclude = set(group_cols + ["lat", "lon"])
    data_cols = [c for c in result.columns if c not in exclude and result[c].dtype.kind in "fi"]

    zipcode_mean = result.groupby(group_cols)[data_cols].mean().reset_index()
    zipcode_mean = zipcode_mean.sort_values(group_cols).reset_index(drop=True)

    print(f"[zipcode_agg] Done. Output shape: {zipcode_mean.shape}")
    return zipcode_mean
