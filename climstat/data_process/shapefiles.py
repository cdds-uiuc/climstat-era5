"""
Shared spatial utilities for loading Illinois shapefiles and building grid points.

This module centralizes shapefile I/O, CRS handling, and ERA5 grid-point
construction that was previously duplicated across county_agg, zipcode_agg,
and visualization.
"""

import os

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point


# ── Illinois FIPS state code (single source of truth) ────────────────────
IL_STATEFP = "17"

# ── Default shapefile paths ──────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

DEFAULT_COUNTY_SHAPEFILE = os.path.join(
    _PROJECT_ROOT, "data", "shapefiles", "county", "tl_2025_us_county.shp",
)

DEFAULT_ZCTA_SHAPEFILE = os.path.join(
    _PROJECT_ROOT, "data", "shapefiles", "zipcodes", "tl_2025_us_zcta520.shp",
)


def ensure_epsg4326(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Reproject to EPSG:4326 (WGS84 lat/lon) if not already."""
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    return gdf


def load_illinois_counties(
    shapefile_path: str | None = None,
    columns: list[str] | None = None,
) -> gpd.GeoDataFrame:
    """
    Load Illinois county boundaries from a TIGER/Line shapefile.

    Uses ``where`` and ``columns`` so only IL rows and the requested
    columns are read from disk.

    Parameters
    ----------
    shapefile_path : str, optional
        Path to the US county shapefile.  Defaults to the bundled file.
    columns : list[str], optional
        Columns to read (in addition to geometry).
        Defaults to ["STATEFP", "GEOID", "NAMELSAD"].
    """
    if shapefile_path is None:
        shapefile_path = DEFAULT_COUNTY_SHAPEFILE
    if columns is None:
        columns = ["STATEFP", "GEOID", "NAMELSAD"]

    gdf_il = gpd.read_file(
        shapefile_path,
        where=f"STATEFP = '{IL_STATEFP}'",
        columns=columns,
    )
    return ensure_epsg4326(gdf_il)


def load_illinois_zctas(
    zcta_shapefile: str | None = None,
    county_shapefile: str | None = None,
) -> gpd.GeoDataFrame:
    """
    Load ZCTA boundaries that intersect Illinois.

    Builds an Illinois outline from the county shapefile, then reads
    only ZCTAs whose geometry intersects that outline (via the ``mask``
    parameter, which pushes spatial filtering to GDAL).

    Parameters
    ----------
    zcta_shapefile : str, optional
    county_shapefile : str, optional
    """
    if zcta_shapefile is None:
        zcta_shapefile = DEFAULT_ZCTA_SHAPEFILE
    if county_shapefile is None:
        county_shapefile = DEFAULT_COUNTY_SHAPEFILE

    # Build Illinois outline from counties
    il_counties = gpd.read_file(
        county_shapefile,
        where=f"STATEFP = '{IL_STATEFP}'",
        columns=["STATEFP"],
    )
    il_counties = ensure_epsg4326(il_counties)
    il_outline = il_counties.union_all()

    # Load ZCTAs that intersect the IL outline (fast spatial filter via GDAL)
    zctas = gpd.read_file(
        zcta_shapefile,
        mask=il_outline,
        columns=["ZCTA5CE20"],
    )
    zctas = ensure_epsg4326(zctas)

    # Keep only ZCTAs whose centroid falls inside Illinois.
    # The mask= filter above uses "intersects", which includes
    # neighboring-state ZCTAs that merely touch the border.
    centroids = zctas.geometry.centroid
    zctas = zctas[centroids.within(il_outline)].reset_index(drop=True)
    return zctas


def build_grid_points(ds: xr.Dataset) -> gpd.GeoDataFrame:
    """
    Create a GeoDataFrame of Point geometries from the ERA5 lat/lon grid.

    Takes the lat and lon coordinates from the xarray Dataset, creates
    every (lon, lat) combination, and wraps each pair in a Shapely Point.
    """
    lons, lats = np.meshgrid(ds.lon.values, ds.lat.values)
    points_df = pd.DataFrame({"lat": lats.flatten(), "lon": lons.flatten()})
    points_df = points_df.drop_duplicates()
    geometry = [Point(lon, lat) for lon, lat in zip(points_df.lon, points_df.lat)]
    return gpd.GeoDataFrame(points_df, geometry=geometry, crs="EPSG:4326")


def ensure_lon_180(ds: xr.Dataset) -> xr.Dataset:
    """
    Convert longitude from 0-360 to -180:180 if needed.

    ERA5 sometimes uses 0-360 longitudes (e.g. 270° instead of -90°).
    Shapefiles use -180 to 180.  This ensures they match.
    """
    if ds["lon"].values.max() > 180:
        ds = ds.assign_coords(lon=(ds["lon"] + 180) % 360 - 180)
        ds = ds.sortby("lon")
    return ds
