"""
Dashboard data helpers: scan output CSVs, load data, extract column choices.

Keeps the dashboard notebook thin by centralizing data discovery and loading.
"""

import os
import re
from functools import lru_cache

import pandas as pd


# Regex to parse output CSV filenames:
#   {prefix}_{metric}_{start}_{end}_{source}_{region}.csv
_CSV_PATTERN = re.compile(
    r"^(daily|averages)_(.+?)_(\d{8})_(\d{8})_(ERA5|IFS)_(counties|zipcodes)\.csv$"
)


def scan_output_dir(output_dir: str) -> list[dict]:
    """
    Scan the output directory and return metadata for every CSV found.

    Returns a list of dicts with keys:
        prefix, metric, start, end, source, region, path
    """
    results = []
    for fname in sorted(os.listdir(output_dir)):
        m = _CSV_PATTERN.match(fname)
        if m:
            results.append({
                "prefix": m.group(1),
                "metric": m.group(2),
                "start": m.group(3),
                "end": m.group(4),
                "source": m.group(5),
                "region": m.group(6),
                "path": os.path.join(output_dir, fname),
            })
    return results


def get_available_metrics(catalog: list[dict]) -> list[str]:
    """Return sorted unique metric names from a scan catalog."""
    return sorted({entry["metric"] for entry in catalog})


def get_date_ranges(catalog: list[dict]) -> dict[str, tuple[str, str]]:
    """Return {source: (start, end)} for each data source in the catalog."""
    ranges = {}
    for entry in catalog:
        src = entry["source"]
        if src not in ranges:
            ranges[src] = (entry["start"], entry["end"])
    return ranges


@lru_cache(maxsize=32)
def load_csv(path: str, region: str) -> pd.DataFrame:
    """Load a single CSV with correct dtypes, cached in memory."""
    if region == "zipcodes":
        return pd.read_csv(path, dtype={"ZCTA5CE20": str})
    return pd.read_csv(path, dtype={"GEOID": str})


def load_metric_data(
    catalog: list[dict],
    metric: str,
    source: str,
    region: str,
) -> dict[str, pd.DataFrame | None]:
    """
    Load daily + averages DataFrames for a metric/source/region combo.

    Returns dict with keys "daily" and "averages", each a DataFrame or None.
    """
    result = {"daily": None, "averages": None}
    # catalog is a list of dicts, so we need to convert to tuple for caching
    for entry in catalog:
        if entry["metric"] == metric and entry["source"] == source and entry["region"] == region:
            df = load_csv(entry["path"], region)
            result[entry["prefix"]] = df
    return result


def get_daily_columns(df: pd.DataFrame | None) -> list[str]:
    """
    Extract plottable daily columns from a daily DataFrame.

    Returns columns like daily_mean, daily_max, daily_min, hours_above_90, etc.
    """
    if df is None:
        return []
    skip = {"GEOID", "NAMELSAD", "ZCTA5CE20", "time"}
    return [c for c in df.columns if c not in skip]


def get_averages_columns(df: pd.DataFrame | None) -> list[str]:
    """
    Extract plottable columns from an averages DataFrame.

    Returns columns like avg_daily_max, avg_days_per_year_above_90, etc.
    """
    if df is None:
        return []
    skip = {"GEOID", "NAMELSAD", "ZCTA5CE20"}
    return [c for c in df.columns if c not in skip]


def get_regions(df: pd.DataFrame | None, region_type: str) -> list[str]:
    """Return sorted unique region values from a daily DataFrame."""
    if df is None:
        return []
    col = "NAMELSAD" if region_type == "counties" else "ZCTA5CE20"
    return sorted(df[col].unique().tolist())
