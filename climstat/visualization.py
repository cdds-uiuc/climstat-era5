"""
Step 6: Visualization functions for climate statistics output.

What this module does
---------------------
After the pipeline produces county-level DataFrames (daily and averages
summaries), this module provides three standard plot types to quickly
inspect the results:

1. **County time-series** — a line plot of a metric over time for one
   county.  Good for seeing seasonal patterns and trends.

2. **County choropleth map** — a map of Illinois with each county
   colored by a metric value.  Good for spatial patterns (e.g. "which
   counties are hottest?").

3. **Threshold heatmap** — a grid with counties on the y-axis and
   years on the x-axis, colored by how many hours a threshold was
   exceeded.  Good for identifying hot spots across space and time.

All functions accept an optional ``ax`` parameter so you can embed them
in multi-panel figures.  If ``ax`` is None, a new figure is created.

Key matplotlib concepts
-----------------------
- **fig, ax = plt.subplots()**: Creates a Figure (the window) and an
  Axes (the actual plot area).  All drawing happens on the Axes.
- **ax.plot(x, y)**: Draws a line plot.
- **gdf.plot(column=..., ax=ax)**: GeoPandas can draw choropleth maps
  directly from a GeoDataFrame by coloring polygons based on a column.
- **ax.imshow(array)**: Draws a 2D array as a colored image (heatmap).
"""

import os

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates


# ── Default shapefile path (same as county_agg) ──────────────────────────
DEFAULT_SHAPEFILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data", "shapefiles", "county", "tl_2025_us_county.shp",
)

# Illinois FIPS state code
IL_STATEFP = "17"


def _load_il_counties(shapefile_path: str | None = None) -> gpd.GeoDataFrame:
    """Load Illinois county boundaries from shapefile, ensuring EPSG:4326."""
    if shapefile_path is None:
        shapefile_path = DEFAULT_SHAPEFILE
    gdf = gpd.read_file(shapefile_path)
    gdf_il = gdf[gdf["STATEFP"] == IL_STATEFP].reset_index(drop=True)
    if gdf_il.crs is None or gdf_il.crs.to_epsg() != 4326:
        gdf_il = gdf_il.to_crs("EPSG:4326")
    return gdf_il


# ===========================================================================
# 1. County time-series
# ===========================================================================

def plot_county_timeseries(
    df: pd.DataFrame,
    county: str,
    daily_summary: str,
    time_col: str = "time",
    date_range: tuple[str, str] | None = None,
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Line plot of a daily metric for a single county.

    Example usage:
        plot_county_timeseries(daily_df, "Cook County", "daily_max")
        plot_county_timeseries(daily_df, "Cook County", "daily_max",
                               date_range=("2023-06-01", "2023-08-31"))

    Parameters
    ----------
    df : pd.DataFrame
        County-level daily data (output of county_agg.aggregate_to_counties
        applied to daily_summary results).  Must contain columns NAMELSAD,
        the time column, and the metric column.
    county : str
        County NAMELSAD to plot, e.g. "Cook County".
    daily_summary : str
        Column to plot on the y-axis, e.g. "daily_max" or "hours_above_90".
    time_col : str
        Name of the time/date column (default "time").
    date_range : tuple of (start, end) strings, optional
        Restrict the plot to dates between start and end (inclusive).
        Accepts any format understood by pd.to_datetime, e.g.
        ("2023-06-01", "2023-08-31").  If None, all dates are shown.
    title : str, optional
        Custom title.  If None, one is generated automatically.
    ax : matplotlib Axes, optional
        Axes to draw on.  If None, a new figure is created.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes object (useful for further customisation).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))

    # Filter to the requested county and sort by time
    subset = df[df["NAMELSAD"] == county].sort_values(time_col)
    times = pd.to_datetime(subset[time_col])

    # Apply optional date range filter
    if date_range is not None:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        mask = (times >= start) & (times <= end)
        times = times[mask]
        subset = subset.loc[mask]

    ax.plot(times, subset[daily_summary], linewidth=0.8)

    # Adapt tick spacing to the plotted time span
    span_days = (times.max() - times.min()).days
    if span_days > 730:  # > ~2 years
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
    elif span_days > 60:  # 2 months – 2 years
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    else:  # < 2 months
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.xaxis.set_minor_locator(mdates.DayLocator())
    ax.tick_params(axis="x", rotation=0 if span_days > 730 else 45)

    ax.set_xlabel("Date")
    ax.set_ylabel(daily_summary)
    ax.set_title(title or f"{daily_summary} — {county}")
    ax.grid(True, which="major", alpha=0.3)
    plt.tight_layout()
    return ax


# ===========================================================================
# 2. County choropleth map
# ===========================================================================

def plot_county_map(
    df: pd.DataFrame,
    average_summary: str,
    shapefile_path: str | None = None,
    title: str | None = None,
    cmap: str = "YlOrRd",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Choropleth map of a metric value across Illinois counties.

    Each county polygon is colored according to its metric value.  This
    works best with "averages" summary data (one value per county) or a
    single-date slice of daily data.

    Example usage:
        plot_county_map(averages_df, "avg_daily_max")

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns "GEOID" and *average_summary*.  Should be one
        row per county.
    average_summary : str
        Column to color-code on the map.
    shapefile_path : str, optional
        County shapefile path.  Defaults to bundled shapefile.
    title : str, optional
        Custom title.
    cmap : str
        Matplotlib colormap name (e.g. "YlOrRd", "Blues", "coolwarm").
    ax : matplotlib Axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    # Load county polygons
    gdf_il = _load_il_counties(shapefile_path)

    # Merge our data onto the county geometries using GEOID as the key.
    # "left" join keeps all counties; those with no data show as grey.
    merged = gdf_il.merge(df[["GEOID", average_summary]].assign(GEOID=df["GEOID"].astype(str)), on="GEOID", how="left")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 10))

    # GeoPandas .plot() draws the polygons and colors them by the column
    merged.plot(
        column=average_summary,
        cmap=cmap,
        legend=True,
        edgecolor="black",
        linewidth=0.5,
        ax=ax,
        missing_kwds={"color": "lightgrey", "label": "No data"},
    )
    ax.set_title(title or average_summary)
    ax.set_axis_off()   # hide the lat/lon axes for a cleaner map
    plt.tight_layout()
    return ax


# ===========================================================================
# 3. Threshold heatmap (county x year)
# ===========================================================================

def plot_threshold_heatmap(
    df: pd.DataFrame,
    metric_col: str,
    time_col: str = "time",
    title: str | None = None,
    cmap: str = "YlOrRd",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Heatmap of an exceedance statistic by county (rows) and year (columns).

    Useful for visualizing questions like "which counties had the most
    hours above 90°F Heat Index in each year?"

    The daily values are summed within each year, so the cell values
    represent total hours (or days) per year.

    Example usage:
        plot_threshold_heatmap(daily_df, "hours_above_90")

    Parameters
    ----------
    df : pd.DataFrame
        County-level daily data with columns NAMELSAD, time column, and
        the metric column to visualize.
    metric_col : str
        Column to aggregate and display, e.g. "hours_above_90".
    time_col : str
        Name of the time/date column.
    title : str, optional
    cmap : str
        Colormap for the heatmap.
    ax : matplotlib Axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    work = df.copy()
    # Extract the year from the time column for grouping
    work["year"] = pd.to_datetime(work[time_col]).dt.year

    # Create a pivot table: rows=counties, columns=years, values=sum of metric
    # .unstack() converts the year index level into columns
    pivot = work.groupby(["NAMELSAD", "year"])[metric_col].sum().unstack(fill_value=0)

    if ax is None:
        # Scale figure height by number of counties so labels don't overlap
        fig, ax = plt.subplots(figsize=(12, max(6, len(pivot) * 0.18)))

    # imshow draws the 2D array as a colored image
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap)
    # Label the y-axis with county names
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=6)
    # Label the x-axis with years
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_xlabel("Year")
    plt.colorbar(im, ax=ax, label=metric_col)
    ax.set_title(title or f"{metric_col} by County and Year")
    plt.tight_layout()
    return ax
