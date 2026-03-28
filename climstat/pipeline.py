"""
Pipeline orchestration for the climstat ERA5/IFS climate statistics pipeline.

Three top-level functions corresponding to the three pipeline modules:

    acquire_data()    — download ERA5 + IFS raw data (Module 1)
    process_data()    — compute metrics, statistics, aggregation → CSV (Module 2)
    visualize_data()  — load CSVs and produce plots (Module 3)

Lower-level helpers:

    run_pipeline()    — shared workhorse for metrics → stats → agg → CSV
    load_metric()     — load output CSVs for a single metric on demand
    get_thresholds()  — extract threshold lists from the config dict
"""

import os

import pandas as pd
import xarray as xr

from .era5_extract import extract_era5
from .ifs_extract import extract_ifs
from .metrics import METRIC_REGISTRY, compute_metrics
from .statistics import daily_summary, averages_summary
from .county_agg import aggregate_to_counties
from .zipcode_agg import aggregate_to_zipcodes, build_zcta_mapping
from .visualization import (
    plot_county_timeseries,
    plot_county_map,
    plot_zipcode_map,
)
from .shapefiles import load_illinois_counties, load_illinois_zctas


# ===========================================================================
# Helpers
# ===========================================================================

def _output_csv_path(output_dir, prefix, metric, start_str, end_str, source_tag, region_tag):
    """Build the canonical CSV path for a pipeline output."""
    return os.path.join(
        output_dir,
        f"{prefix}_{metric}_{start_str}_{end_str}_{source_tag}_{region_tag}.csv",
    )


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


def get_thresholds(
    thresholds: dict[str, dict[str, list[float]]],
    name: str,
) -> tuple[list[float], list[float]]:
    """
    Extract (above, below) threshold lists for a metric.

    Parameters
    ----------
    thresholds : dict
        Mapping of metric name -> {"above": [...], "below": [...]}.
    name : str
        Metric name to look up.

    Returns
    -------
    tuple of (thresholds_above, thresholds_below)
    """
    entry = thresholds.get(name, {})
    return entry.get("above", []), entry.get("below", [])


# ===========================================================================
# run_pipeline — shared workhorse
# ===========================================================================

def run_pipeline(
    ds: xr.Dataset,
    metrics: list[str],
    thresholds: dict[str, dict[str, list[float]]],
    shapefile_path: str,
    zcta_shapefile_path: str,
    output_dir: str,
    convert_to_f: bool = True,
    source_label: str = "",
    zcta_mapping: pd.DataFrame | None = None,
    daily_only: bool = False,
) -> dict:
    """
    Run the full metrics -> statistics -> aggregation -> CSV export pipeline.

    Parameters
    ----------
    ds : xr.Dataset
        Gridded data with ERA5-compatible variable names and dimensions
        (time, lat, lon).
    metrics : list[str]
        Metric names to compute (e.g. ["heat_index", "wbgt"]).
    thresholds : dict
        Threshold configuration.
    shapefile_path : str
        Path to the US county shapefile.
    zcta_shapefile_path : str
        Path to the US ZCTA shapefile.
    output_dir : str
        Directory for output CSVs.
    convert_to_f : bool
        Convert Kelvin-native metrics to Fahrenheit (default True).
    source_label : str
        Label inserted into CSV filenames (e.g. "ifs").
    zcta_mapping : pd.DataFrame, optional
        Precomputed ZCTA-to-grid-point mapping.
    daily_only : bool
        If True, skip averages computation and only produce daily CSVs
        (e.g. for IFS data where period averages aren't meaningful).

    Returns
    -------
    dict with keys: "start_str", "end_str", "zcta_mapping"
    """
    label = f" {source_label.upper()}" if source_label else ""
    source_tag = source_label.upper() if source_label else "ERA5"

    # ── Derive date range & check CSV cache ───────────────────────────────
    start_str = pd.Timestamp(ds.time.values[0]).strftime("%Y%m%d")
    end_str = pd.Timestamp(ds.time.values[-1]).strftime("%Y%m%d")
    print(f"  {source_tag} data range: {start_str} -> {end_str}")

    def _all_csvs_exist(name):
        """Check whether expected output CSVs exist for a metric."""
        prefixes = ("daily",) if daily_only else ("daily", "averages")
        return all(
            os.path.exists(
                _output_csv_path(output_dir, prefix, name, start_str, end_str, source_tag, region_tag)
            )
            for prefix in prefixes
            for region_tag in ("counties", "zipcodes")
        )

    cached_metrics = []
    uncached_metrics = []
    for name in metrics:
        if _all_csvs_exist(name):
            cached_metrics.append(name)
        else:
            uncached_metrics.append(name)

    if cached_metrics:
        print(f"  {source_tag} cached (skipping): {', '.join(cached_metrics)}")

    if not uncached_metrics:
        print(f"\nAll{label} metrics cached — skipping computation.")
        if zcta_mapping is None:
            zcta_mapping = build_zcta_mapping(
                ds,
                zcta_shapefile=zcta_shapefile_path,
                county_shapefile=shapefile_path,
            )
        del ds
        return {
            "start_str": start_str,
            "end_str": end_str,
            "zcta_mapping": zcta_mapping,
        }

    # ── Compute derived metrics (uncached only) ──────────────────────────
    daily_results = {}
    averages_results = {}
    daily_county = {}
    averages_county = {}
    daily_zipcode = {}
    averages_zipcode = {}

    print(f"Computing{label} derived metrics ({', '.join(uncached_metrics)}) ...")
    metric_arrays = compute_metrics(
        ds, metric_names=uncached_metrics, convert_to_fahrenheit=convert_to_f,
    )

    # Build zcta_mapping from ds before freeing it
    if zcta_mapping is None:
        zcta_mapping = build_zcta_mapping(
            ds,
            zcta_shapefile=zcta_shapefile_path,
            county_shapefile=shapefile_path,
        )

    del ds  # free raw data

    for name, arr in metric_arrays.items():
        print(f"  {name}: shape={arr.shape}, "
              f"min={float(arr.min()):.1f}, max={float(arr.max()):.1f}")

    # ── Compute statistics ────────────────────────────────────────────────
    for name, arr in metric_arrays.items():
        above, below = get_thresholds(thresholds, name)
        print(f"\n---{label} {name} (above: {above}, below: {below}) ---")

        daily_ds = daily_summary(arr, thresholds=above, thresholds_below=below)
        daily_results[name] = daily_ds

        if not daily_only:
            averages_results[name] = averages_summary(
                arr,
                thresholds=above,
                thresholds_below=below,
                daily_max=daily_ds["daily_max"],
                daily_mean=daily_ds["daily_mean"],
                daily_min=daily_ds["daily_min"],
            )

    # ── Aggregate to counties ─────────────────────────────────────────────
    for name in uncached_metrics:
        print(f"\n==={label} {name} ===")
        print("  Aggregating daily summary to counties ...")
        daily_county[name] = aggregate_to_counties(
            daily_results[name], shapefile_path=shapefile_path,
        )
        if not daily_only:
            print("  Aggregating averages to counties ...")
            averages_county[name] = aggregate_to_counties(
                averages_results[name], shapefile_path=shapefile_path,
            )

    # ── Aggregate to ZIP codes ────────────────────────────────────────────
    for name in uncached_metrics:
        print(f"  Aggregating daily summary to ZIP codes ...")
        daily_zipcode[name] = aggregate_to_zipcodes(
            daily_results[name], zcta_mapping=zcta_mapping,
        )
        if not daily_only:
            print("  Aggregating averages to ZIP codes ...")
            averages_zipcode[name] = aggregate_to_zipcodes(
                averages_results[name], zcta_mapping=zcta_mapping,
            )

    # ── Save CSVs (uncached only) ─────────────────────────────────────────
    for name in uncached_metrics:
        for region_tag in ("counties", "zipcodes"):
            daily_data = daily_county if region_tag == "counties" else daily_zipcode
            daily_out = _output_csv_path(
                output_dir, "daily", name, start_str, end_str, source_tag, region_tag,
            )
            daily_data[name].to_csv(daily_out, index=False)
            print(f"  Saved: {daily_out}")

            if not daily_only:
                avg_data = averages_county if region_tag == "counties" else averages_zipcode
                avg_out = _output_csv_path(
                    output_dir, "averages", name, start_str, end_str, source_tag, region_tag,
                )
                avg_data[name].to_csv(avg_out, index=False)
                print(f"  Saved: {avg_out}")

    print(f"\n{label.strip() or 'ERA5'} pipeline complete.")

    return {
        "start_str": start_str,
        "end_str": end_str,
        "zcta_mapping": zcta_mapping,
    }


# ===========================================================================
# Module 1: acquire_data
# ===========================================================================

def acquire_data(
    metrics: list[str],
    year_start: int,
    year_end: int,
    cache_dir: str = "data/cache",
) -> dict:
    """
    Download ERA5 and IFS raw data (Module 1).

    Determines which ERA5 variables are needed for the requested metrics,
    downloads (or loads from cache) the ERA5 data, then fills the gap
    between the last valid ERA5 day and today with IFS data.

    Parameters
    ----------
    metrics : list[str]
        Metric names to compute (must be keys in METRIC_REGISTRY).
    year_start, year_end : int
        Year range for ERA5 extraction (inclusive).
    cache_dir : str
        Directory for raw NetCDF cache files.

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

    # Extract IFS to fill the gap (only fetch the variables we need)
    ifs_ds = extract_ifs(
        era5_end_date=era5_end_date,
        grid_lats=era5_lats,
        grid_lons=era5_lons,
        forecast_days=0,
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


# ===========================================================================
# Module 2: process_data
# ===========================================================================

def process_data(
    era5_ds: xr.Dataset,
    ifs_ds: xr.Dataset | None,
    metrics: list[str],
    thresholds: dict[str, dict[str, list[float]]],
    shapefile_path: str,
    zcta_shapefile_path: str,
    output_dir: str,
    convert_to_f: bool = True,
) -> dict:
    """
    Compute metrics, statistics, and aggregation, then save CSVs (Module 2).

    Runs the full pipeline for ERA5 (daily + averages) and, if available,
    IFS (daily only).  Cached CSVs are skipped automatically.

    Parameters
    ----------
    era5_ds : xr.Dataset
        ERA5 hourly gridded data.
    ifs_ds : xr.Dataset or None
        IFS hourly gridded data (None if ERA5 is up to date).
    metrics : list[str]
        Metric names to compute.
    thresholds : dict
        Threshold configuration for exceedance statistics.
    shapefile_path : str
        Path to the US county shapefile.
    zcta_shapefile_path : str
        Path to the US ZCTA shapefile.
    output_dir : str
        Directory for output CSVs.
    convert_to_f : bool
        Convert Kelvin-native metrics to Fahrenheit (default True).

    Returns
    -------
    dict with keys:
        "era5_start"  — YYYYMMDD start date for ERA5 outputs
        "era5_end"    — YYYYMMDD end date for ERA5 outputs
        "ifs_start"   — YYYYMMDD start date for IFS outputs, or None
        "ifs_end"     — YYYYMMDD end date for IFS outputs, or None
    """
    os.makedirs(output_dir, exist_ok=True)

    # ERA5: daily + averages
    era5 = run_pipeline(
        era5_ds,
        metrics=metrics,
        thresholds=thresholds,
        shapefile_path=shapefile_path,
        zcta_shapefile_path=zcta_shapefile_path,
        output_dir=output_dir,
        convert_to_f=convert_to_f,
        daily_only=False,
    )

    ifs_start = None
    ifs_end = None

    # IFS: daily only
    if ifs_ds is not None:
        ifs = run_pipeline(
            ifs_ds,
            metrics=metrics,
            thresholds=thresholds,
            shapefile_path=shapefile_path,
            zcta_shapefile_path=zcta_shapefile_path,
            output_dir=output_dir,
            convert_to_f=convert_to_f,
            source_label="ifs",
            zcta_mapping=era5["zcta_mapping"],
            daily_only=True,
        )
        ifs_start = ifs["start_str"]
        ifs_end = ifs["end_str"]

    return {
        "era5_start": era5["start_str"],
        "era5_end": era5["end_str"],
        "ifs_start": ifs_start,
        "ifs_end": ifs_end,
    }


# ===========================================================================
# Module 3: visualize_data
# ===========================================================================

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
    recent_start = (pd.Timestamp.today() - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
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
