"""
Module 3: Visualize climate statistics from CSV outputs.

Loads output CSVs and produces time series plots and choropleth maps
for each requested metric.  Runs independently of acquire/process —
only needs the date-range strings to locate the right files.
"""

import os

import pandas as pd

from ..data_process.process import _output_csv_path
from .visualization import (
    plot_county_timeseries,
    plot_county_map,
    plot_zipcode_map,
)
from ..data_process.shapefiles import load_illinois_counties, load_illinois_zctas


def load_metric(
    metric: str,
    output_dir: str,
    start_str: str,
    end_str: str,
    source_label: str = "",
) -> dict[str, pd.DataFrame | None]:
    """
    Load the output CSVs for a single metric on demand.

    Parameters
    ----------
    metric : str
        Metric name (e.g. "heat_index").
    output_dir : str
        Directory containing the output CSVs.
    start_str, end_str : str
        Date range in YYYYMMDD format.
    source_label : str
        Source label (e.g. "ifs").  Empty string for ERA5.

    Returns
    -------
    dict with keys: "daily_counties", "averages_counties",
                    "daily_zipcodes", "averages_zipcodes"
        Each value is a pd.DataFrame, or None if the file doesn't exist.
    """
    source_tag = source_label.upper() if source_label else "ERA5"
    result = {}
    for prefix in ("daily", "averages"):
        for region_tag in ("counties", "zipcodes"):
            path = _output_csv_path(
                output_dir, prefix, metric, start_str, end_str, source_tag, region_tag,
            )
            key = f"{prefix}_{region_tag}"
            if not os.path.exists(path):
                result[key] = None
            elif region_tag == "zipcodes":
                # Keep ZCTA5CE20 as string so "61820" matches the parameter
                result[key] = pd.read_csv(path, dtype={"ZCTA5CE20": str})
            else:
                result[key] = pd.read_csv(path)
    return result


def visualize_data(
    metrics: list[str],
    thresholds: dict[str, dict[str, list[float]]],
    output_dir: str,
    era5_start: str,
    era5_end: str,
    ifs_start: str | None,
    ifs_end: str | None,
    shapefile_path: str,
    zcta_shapefile_path: str,
    example_county: str = "Cook County",
    example_zipcode: str = "60601",
    timeseries_summary: str = "daily_max",
    map_summary: str = "avg_daily_max",
) -> None:
    """
    Load output CSVs and produce plots for all metrics (Module 3).

    Reads from CSV files on disk — does not require the pipeline to have
    run in the same session.  Only needs the date-range strings to locate
    the right files.

    For each metric produces:
    - County time series (full range + last 2 months), ERA5 + IFS overlay
    - ZIP code time series (full range + last 2 months), ERA5 + IFS overlay
    - County averages choropleth map (ERA5 only)
    - ZIP code averages choropleth map (ERA5 only)

    Parameters
    ----------
    metrics : list[str]
        Metric names to visualize.
    thresholds : dict
        Threshold configuration (used for labelling).
    output_dir : str
        Directory containing the output CSVs.
    era5_start, era5_end : str
        YYYYMMDD date range for ERA5 outputs.
    ifs_start, ifs_end : str or None
        YYYYMMDD date range for IFS outputs (None if no IFS data).
    shapefile_path : str
        Path to the US county shapefile.
    zcta_shapefile_path : str
        Path to the US ZCTA shapefile.
    example_county : str
        County name for time series plots.
    example_zipcode : str
        ZIP code for time series plots.
    timeseries_summary : str
        Daily column to plot in time series (e.g. "daily_max", "daily_mean",
        "hours_above_90").
    map_summary : str
        Averages column to plot on maps (e.g. "avg_daily_max",
        "avg_hours_per_year_above_90").
    """
    # Pre-load shapefiles once for reuse across all metrics
    counties_gdf = load_illinois_counties(shapefile_path)
    zctas_gdf = load_illinois_zctas(zcta_shapefile_path, shapefile_path)

    # Recent date range for zoomed time series
    recent_start = (pd.Timestamp.today() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    if ifs_end is not None:
        recent_end = (pd.Timestamp(ifs_end) + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    else:
        recent_end = (pd.Timestamp.today() + pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    recent_range = (recent_start, recent_end)

    for metric in metrics:
        print(f"\n{'='*60}")
        print(f"Visualizing: {metric}")
        print(f"{'='*60}")

        # Load ERA5 CSVs
        era5 = load_metric(metric, output_dir, era5_start, era5_end)
        daily_county = era5["daily_counties"]
        averages_county = era5["averages_counties"]
        daily_zip = era5["daily_zipcodes"]
        averages_zip = era5["averages_zipcodes"]

        if daily_county is None or daily_zip is None:
            print(f"  Skipping {metric}: output CSVs not found. Run process_data() first.")
            continue

        # Load IFS CSVs (if available)
        ifs_daily_county = None
        ifs_daily_zip = None
        if ifs_start is not None:
            ifs = load_metric(metric, output_dir, ifs_start, ifs_end, source_label="ifs")
            ifs_daily_county = ifs["daily_counties"]
            ifs_daily_zip = ifs["daily_zipcodes"]

        # ── Time series plots ─────────────────────────────────────────────
        if timeseries_summary not in daily_county.columns:
            print(f"  Skipping time series: '{timeseries_summary}' not in daily columns for {metric}. "
                  f"Available: {[c for c in daily_county.columns if c not in ('GEOID','NAMELSAD','time')]}")
        else:
            # County time series
            plot_county_timeseries(
                daily_county,
                county=example_county,
                daily_summary=timeseries_summary,
                title=f"{metric} {timeseries_summary} — {example_county} (full range)",
                df_forecast=ifs_daily_county,
            )
            plot_county_timeseries(
                daily_county,
                county=example_county,
                daily_summary=timeseries_summary,
                date_range=recent_range,
                title=f"{metric} {timeseries_summary} — {example_county} (recent)",
                df_forecast=ifs_daily_county,
                marker="o",
            )

            # ZIP code time series
            plot_county_timeseries(
                daily_zip,
                county=example_zipcode,
                daily_summary=timeseries_summary,
                region_col="ZCTA5CE20",
                title=f"{metric} {timeseries_summary} — ZIP {example_zipcode} (full range)",
                df_forecast=ifs_daily_zip,
            )
            plot_county_timeseries(
                daily_zip,
                county=example_zipcode,
                daily_summary=timeseries_summary,
                region_col="ZCTA5CE20",
                date_range=recent_range,
                title=f"{metric} {timeseries_summary} — ZIP {example_zipcode} (recent)",
                df_forecast=ifs_daily_zip,
                marker="o",
            )

        # ── County averages map (ERA5 only) ───────────────────────────────
        if averages_county is not None and map_summary in averages_county.columns:
            plot_county_map(
                averages_county,
                average_summary=map_summary,
                shapefile_path=shapefile_path,
                title=f"{metric}: {map_summary} — Counties",
                gdf=counties_gdf,
            )
        elif averages_county is not None:
            print(f"  Skipping county map: '{map_summary}' not in averages columns "
                  f"({[c for c in averages_county.columns if c.startswith('avg_')]})")

        # ── ZIP code averages map (ERA5 only) ─────────────────────────────
        if averages_zip is not None and map_summary in averages_zip.columns:
            plot_zipcode_map(
                averages_zip,
                average_summary=map_summary,
                zcta_shapefile_path=zcta_shapefile_path,
                county_shapefile_path=shapefile_path,
                title=f"{metric}: {map_summary} — ZIP Codes",
                gdf=zctas_gdf,
            )
        elif averages_zip is not None:
            print(f"  Skipping ZIP map: '{map_summary}' not in averages columns")

    print(f"\nVisualization complete for {len(metrics)} metrics.")
