# climstat/shapefile_extract.py
"""
Step 1b: Build derived shapefiles (one-time preprocessing).

Currently produces:
  - tl_2025_il_counties_land_only.shp: IL counties with Lake Michigan erased,
    so over-water ERA5 grid points don't contaminate Cook/Lake county averages.

Cached output is regenerated only if missing or if any input file is newer.
"""
import os
from pathlib import Path
import geopandas as gpd
import pandas as pd


def build_land_only_counties(
    county_shapefile: str,
    water_shapefiles: list[str],
    output_path: str,
    force: bool = False,
) -> str:
    """
    Erase water polygons from IL county boundaries.

    Skips work if `output_path` exists and is newer than every input.
    Returns the output path.
    """
    inputs = [county_shapefile, *water_shapefiles]
    for p in inputs:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing input shapefile: {p}")

    # Cache check: skip if output exists and is fresher than all inputs
    if not force and os.path.exists(output_path):
        out_mtime = os.path.getmtime(output_path)
        if all(os.path.getmtime(p) <= out_mtime for p in inputs):
            print(f"[shapefile_extract] Cached: {output_path}")
            return output_path

    print(f"[shapefile_extract] Building {output_path} ...")
    counties = gpd.read_file(
        county_shapefile, where="STATEFP = '17'",
    ).to_crs("EPSG:4326")

    water = gpd.GeoDataFrame(
        pd.concat(
            [gpd.read_file(p).to_crs("EPSG:4326") for p in water_shapefiles],
            ignore_index=True,
        ),
        crs="EPSG:4326",
    )

    PROJ = 5070
    counties_p = counties.to_crs(PROJ)
    water_union = water.to_crs(PROJ).union_all()
    counties_p["geometry"] = counties_p.geometry.difference(water_union)
    counties_p = counties_p[~counties_p.is_empty].reset_index(drop=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    counties_p.to_crs("EPSG:4326").to_file(output_path)
    print(f"[shapefile_extract] Wrote {output_path} "
          f"({len(counties_p)} land-only counties)")
    return output_path