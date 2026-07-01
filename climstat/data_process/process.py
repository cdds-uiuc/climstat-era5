"""
Module 2: Process climate data into statistics and CSV outputs.

Computes derived metrics, daily/period statistics, aggregates to
counties and ZIP codes, and exports CSVs.
"""

import os

import pandas as pd
import xarray as xr

from .metrics import compute_metrics
from .statistics import daily_summary, averages_summary
from .county_agg import aggregate_to_counties
from .zipcode_agg import aggregate_to_zipcodes, build_zcta_mapping


# ===========================================================================
# Helpers
# ===========================================================================

def _output_csv_path(output_dir, prefix, metric, start_str, end_str, source_tag, region_tag):
    """Build the canonical CSV path for a pipeline output."""
    return os.path.join(
        output_dir,
        f"{prefix}_{metric}_{start_str}_{end_str}_{source_tag}_{region_tag}.csv",
    )


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
    # water_shapefile: list[str] | None = None,
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
                # water_shapefile=water_shapefile,
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
            # water_shapefile=water_shapefile,
        )

    del ds  # free raw data

    for name, arr in metric_arrays.items():
        print(f"  {name}: shape={arr.shape}, "
              f"min={float(arr.min()):.1f}, max={float(arr.max()):.1f}")

    # ── Compute statistics ────────────────────────────────────��───────────
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

    # ── Aggregate to counties ──────────────────────────────────��──────────
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
# process_data — top-level entry point
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
    # water_shapefile: list[str] | None = None,
) -> dict:
    """
    Compute metrics, statistics, and aggregation, then save CSVs (Module 2).

    Runs the full pipeline for ERA5 (daily + averages) and IFS (daily only).

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
        # water_shapefile=water_shapefile,
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
            # water_shapefile=water_shapefile,
        )
        ifs_start = ifs["start_str"]
        ifs_end = ifs["end_str"]

    return {
        "era5_start": era5["start_str"],
        "era5_end": era5["end_str"],
        "ifs_start": ifs_start,
        "ifs_end": ifs_end,
    }
