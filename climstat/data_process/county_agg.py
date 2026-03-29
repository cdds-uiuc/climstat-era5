"""
Step 4: Aggregate gridded ERA5-derived data to Illinois county level.

What this module does
---------------------
Steps 2 and 3 produce data on the ERA5 grid — a regular 0.25° lat/lon mesh
covering Illinois (roughly 22 x 19 grid cells).  But for the IEMA climate
database, we need **one value per county**, not per grid cell.

This module performs a "spatial join" to assign each ERA5 grid cell to the
Illinois county it falls within, then averages all grid cells belonging to
the same county.  The result is a pandas DataFrame with one row per county
(or per county per day, if the input has a time dimension).

How the spatial join works
--------------------------
1. Load the county shapefile — a file containing polygon geometries for
   every US county.  We filter to Illinois only (FIPS state code "17").
2. Create a Point geometry for each ERA5 grid cell (lat/lon pair).
3. Use GeoPandas ``sjoin`` (spatial join) to find which county polygon
   each point falls inside.
4. Merge the county labels back onto the full time-series data.
5. Group by county and average.

Key terms
---------
- **Shapefile (.shp)**: A geospatial file format that stores polygon
  boundaries (county outlines, state borders, etc.).  It comes as a set
  of files (.shp, .shx, .dbf, .prj) that must stay together.
- **EPSG:4326**: The standard lat/lon coordinate system (WGS84).  We make
  sure both the grid points and the county polygons use this CRS.
- **GEOID**: A unique numeric identifier for each county (state FIPS +
  county FIPS, e.g. "17031" = Cook County, IL).
- **NAMELSAD**: The full county name, e.g. "Cook County".
- **sjoin (spatial join)**: A GeoPandas operation that pairs rows from two
  GeoDataFrames based on their spatial relationship (e.g. "point within
  polygon").
"""

import pandas as pd
import xarray as xr
import geopandas as gpd

from .shapefiles import (
    DEFAULT_COUNTY_SHAPEFILE as DEFAULT_SHAPEFILE,
    IL_STATEFP,
    load_illinois_counties,
    build_grid_points,
    ensure_lon_180,
)


def aggregate_to_counties(
    ds: xr.Dataset,
    shapefile_path: str | None = None,
) -> pd.DataFrame:
    """
    Aggregate a gridded xr.Dataset to Illinois county-level means.

    This is the main function called by the pipeline notebook.  It takes
    gridded summary statistics (from Step 3) and returns a flat table
    with one row per county (or per county per day).

    Parameters
    ----------
    ds : xr.Dataset
        Gridded data with dimensions (time, lat, lon) — or just (lat, lon)
        for averages summaries.  All numeric data variables are averaged
        per county.
    shapefile_path : str, optional
        Path to US county shapefile.  Defaults to the bundled file at
        ``data/shapefiles/county/tl_2025_us_county.shp``.

    Returns
    -------
    pd.DataFrame
        Columns: GEOID, NAMELSAD, [time if present], plus all data variables
        averaged across grid cells within each county.
    """
    if shapefile_path is None:
        shapefile_path = DEFAULT_SHAPEFILE

    # Step 1: fix longitude convention so grid and shapefile match
    ds = ensure_lon_180(ds)

    # Step 2: load Illinois county polygons from the shapefile
    print("[county_agg] Loading Illinois county shapefile ...")
    gdf_counties = load_illinois_counties(shapefile_path)

    # Step 3: create a Point for each ERA5 grid cell
    points_gdf = build_grid_points(ds)

    # Step 4: spatial join — find which county polygon each grid point
    # falls inside.  "inner" means we only keep points that fall within
    # a county (grid points over Lake Michigan or outside IL are dropped).
    print("[county_agg] Performing spatial join ...")
    joined = gpd.sjoin(points_gdf, gdf_counties, how="inner", predicate="intersects")

    n_counties = joined["GEOID"].nunique()
    n_points = len(joined)
    print(f"[county_agg] {n_points} grid points mapped to {n_counties} counties")

    # ── Nearest-neighbour fallback for counties with no grid points ────────
    # Some small or narrow counties (e.g. Calhoun County, IL) may not contain
    # or touch any ERA5 grid point.  For these we find the closest grid point
    # by straight-line distance from the county centroid and assign it.
    matched_geoids = set(joined["GEOID"].unique())
    unmatched = gdf_counties[~gdf_counties["GEOID"].isin(matched_geoids)]
    if len(unmatched) > 0:
        print(f"[county_agg] {len(unmatched)} county/counties have no grid point — "
              f"applying nearest-neighbour fallback: "
              f"{list(unmatched['NAMELSAD'])}")
        # Compute each unmatched county's centroid in EPSG:4326
        centroids = unmatched.copy()
        centroids["geometry"] = unmatched.geometry.centroid
        # For each centroid, find the nearest ERA5 grid point using sjoin_nearest
        nearest = gpd.sjoin_nearest(
            centroids[["GEOID", "NAMELSAD", "geometry"]],
            points_gdf[["lat", "lon", "geometry"]],
            how="left",
        )[["GEOID", "NAMELSAD", "lat", "lon"]]
        # Append these fallback assignments to the main joined table
        joined = pd.concat(
            [joined[["lat", "lon", "NAMELSAD", "GEOID"]], nearest],
            ignore_index=True,
        )

    # Step 5: convert the xarray Dataset to a pandas DataFrame and merge
    # the county labels (GEOID, NAMELSAD) onto it via the lat/lon columns.
    # After this merge, each row has: time, lat, lon, data values, county name.
    print("[county_agg] Merging with full time-series data ...")
    df = ds.to_dataframe().reset_index()
    result = df.merge(
        joined[["lat", "lon", "NAMELSAD", "GEOID"]],
        on=["lat", "lon"],
        how="inner",  # drop grid points that fell outside IL counties
    )

    # Step 6: group by county (and time, if present) and average all
    # numeric data columns.  This collapses the multiple grid cells per
    # county into a single mean value.
    has_time = "time" in result.columns
    group_cols = ["GEOID", "NAMELSAD"]
    if has_time:
        group_cols.append("time")

    # Find numeric columns to average (exclude lat, lon, and grouping cols)
    exclude = set(group_cols + ["lat", "lon"])
    # dtype.kind "f" = float, "i" = integer — we only want numeric columns
    data_cols = [c for c in result.columns if c not in exclude and result[c].dtype.kind in "fi"]

    county_mean = result.groupby(group_cols)[data_cols].mean().reset_index()
    county_mean = county_mean.sort_values(group_cols).reset_index(drop=True)

    print(f"[county_agg] Done. Output shape: {county_mean.shape}")
    return county_mean
