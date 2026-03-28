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

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from .shapefiles import (
    DEFAULT_COUNTY_SHAPEFILE as DEFAULT_SHAPEFILE,
    load_illinois_counties,
    load_illinois_zctas,
)


# ===========================================================================
# 1. County time-series
# ===========================================================================

def plot_county_timeseries(
    df: pd.DataFrame,
    county: str,
    daily_summary: str,
    time_col: str = "time",
    region_col: str = "NAMELSAD",
    date_range: tuple[str, str] | None = None,
    title: str | None = None,
    ax: plt.Axes | None = None,
    df_forecast: pd.DataFrame | None = None,
    era5_label: str = "ERA5",
    forecast_label: str = "IFS Forecast",
    era5_color: str = "#1f77b4",
    forecast_color: str = "#ff7f0e",
) -> plt.Axes:
    """
    Line plot of a daily metric for a single region (county or ZIP code).

    Optionally overlay a second data source (e.g. IFS forecast) in a
    different color by passing *df_forecast*.

    Example usage:
        plot_county_timeseries(daily_df, "Cook County", "daily_max")
        plot_county_timeseries(daily_df, "61820", "daily_max",
                               region_col="ZCTA5CE20")
        # Dual-source (ERA5 + IFS):
        plot_county_timeseries(daily_df, "Cook County", "daily_max",
                               df_forecast=daily_ifs_df)

    Parameters
    ----------
    df : pd.DataFrame
        Region-level daily data.  Must contain the *region_col* column,
        the time column, and the metric column.
    county : str
        Value to match in *region_col*, e.g. "Cook County" or "61820".
    daily_summary : str
        Column to plot on the y-axis, e.g. "daily_max" or "hours_above_90".
    time_col : str
        Name of the time/date column (default "time").
    region_col : str
        Column that identifies regions.  Default "NAMELSAD" for counties;
        use "ZCTA5CE20" for ZIP codes.
    date_range : tuple of (start, end) strings, optional
        Restrict the plot to dates between start and end (inclusive).
        Accepts any format understood by pd.to_datetime, e.g.
        ("2023-06-01", "2023-08-31").  If None, all dates are shown.
    title : str, optional
        Custom title.  If None, one is generated automatically.
    ax : matplotlib Axes, optional
        Axes to draw on.  If None, a new figure is created.
    df_forecast : pd.DataFrame, optional
        Second data source (e.g. IFS forecast) to overlay.  Must have
        the same columns as *df*.  Plotted in *forecast_color*.
    era5_label : str
        Legend label for the primary data (default "ERA5").
    forecast_label : str
        Legend label for the forecast data (default "IFS Forecast").
    era5_color : str
        Color for the primary data line (default matplotlib blue).
    forecast_color : str
        Color for the forecast data line (default matplotlib orange).

    Returns
    -------
    matplotlib.axes.Axes
        The Axes object (useful for further customisation).
    """
    dual_source = df_forecast is not None

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))

    # Filter to the requested region and sort by time
    subset = df[df[region_col] == county].sort_values(time_col)
    times = pd.to_datetime(subset[time_col])

    # Apply optional date range filter
    if date_range is not None:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        mask = (times >= start) & (times <= end)
        times = times[mask]
        subset = subset.loc[mask]

    # Plot primary (ERA5) line
    plot_kwargs = dict(linewidth=0.8)
    if dual_source:
        plot_kwargs.update(color=era5_color, label=era5_label)
    ax.plot(times, subset[daily_summary], **plot_kwargs)

    # Plot forecast (IFS) line if provided
    # Only filter forecast by the start date — the whole point of forecast
    # data is to extend beyond the ERA5 range, so we never clip the end.
    if dual_source:
        fc_subset = df_forecast[df_forecast[region_col] == county].sort_values(time_col)
        fc_times = pd.to_datetime(fc_subset[time_col])
        if date_range is not None:
            fc_mask = fc_times >= start
            fc_times = fc_times[fc_mask]
            fc_subset = fc_subset.loc[fc_mask]
        if len(fc_subset) > 0:
            ax.plot(fc_times, fc_subset[daily_summary], linewidth=0.8,
                    color=forecast_color, label=forecast_label)
        ax.legend(loc="best")

    # Adapt tick spacing to the plotted time span
    all_times = times
    if dual_source and len(fc_subset) > 0:
        all_times = pd.to_datetime(
            list(times) + list(fc_times)
        )
    span_days = (all_times.max() - all_times.min()).days
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
    gdf: gpd.GeoDataFrame | None = None,
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
    gdf : gpd.GeoDataFrame, optional
        Pre-loaded Illinois county GeoDataFrame.  If provided,
        *shapefile_path* is ignored and no shapefile I/O is performed.

    Returns
    -------
    matplotlib.axes.Axes
    """
    # Load county polygons (skip if pre-loaded)
    gdf_il = gdf if gdf is not None else load_illinois_counties(shapefile_path)

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
    region_col: str = "NAMELSAD",
    top_n: int | None = None,
    title: str | None = None,
    cmap: str = "YlOrRd",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Heatmap of an exceedance statistic by region (rows) and year (columns).

    Useful for visualizing questions like "which counties had the most
    hours above 90°F Heat Index in each year?"

    The daily values are summed within each year, so the cell values
    represent total hours (or days) per year.

    Example usage:
        plot_threshold_heatmap(daily_df, "hours_above_90")
        plot_threshold_heatmap(zip_df, "hours_above_90",
                               region_col="ZCTA5CE20", top_n=50)

    Parameters
    ----------
    df : pd.DataFrame
        Region-level daily data with the *region_col*, time column, and
        the metric column to visualize.
    metric_col : str
        Column to aggregate and display, e.g. "hours_above_90".
    time_col : str
        Name of the time/date column.
    region_col : str
        Column that identifies regions.  Default "NAMELSAD" for counties;
        use "ZCTA5CE20" for ZIP codes.
    top_n : int, optional
        If set, only show the top N regions by total exceedance.  Useful
        for ZIP-code heatmaps where plotting all ~1 500 rows is unreadable.
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

    # Create a pivot table: rows=regions, columns=years, values=sum of metric
    pivot = work.groupby([region_col, "year"])[metric_col].sum().unstack(fill_value=0)

    # Optionally keep only the top-N regions by total exceedance
    if top_n is not None and len(pivot) > top_n:
        totals = pivot.sum(axis=1).nlargest(top_n)
        pivot = pivot.loc[totals.index]

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
    ax.set_title(title or f"{metric_col} by {region_col} and Year")
    plt.tight_layout()
    return ax


# ===========================================================================
# 4. ZIP-code choropleth map
# ===========================================================================

def plot_zipcode_map(
    df: pd.DataFrame,
    average_summary: str,
    zcta_shapefile_path: str | None = None,
    county_shapefile_path: str | None = None,
    title: str | None = None,
    cmap: str = "YlOrRd",
    ax: plt.Axes | None = None,
    gdf: gpd.GeoDataFrame | None = None,
) -> plt.Axes:
    """
    Choropleth map of a metric value across Illinois ZIP codes (ZCTAs).

    Each ZCTA polygon is colored according to its metric value.  Works
    best with "averages" summary data (one value per ZIP code).

    Example usage:
        plot_zipcode_map(averages_zip_df, "avg_daily_max")

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns "ZCTA5CE20" and *average_summary*.  Should be
        one row per ZIP code.
    average_summary : str
        Column to color-code on the map.
    zcta_shapefile_path : str, optional
        ZCTA shapefile path.  Defaults to bundled shapefile.
    county_shapefile_path : str, optional
        County shapefile path (used to derive the Illinois outline).
    title : str, optional
    cmap : str
        Matplotlib colormap name.
    ax : matplotlib Axes, optional
    gdf : gpd.GeoDataFrame, optional
        Pre-loaded Illinois ZCTA GeoDataFrame.  If provided,
        shapefile path arguments are ignored.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if gdf is not None:
        gdf_zctas = gdf
    else:
        gdf_zctas = load_illinois_zctas(zcta_shapefile_path, county_shapefile_path)

    # Merge data onto ZCTA geometries
    merged = gdf_zctas.merge(
        df[["ZCTA5CE20", average_summary]].assign(
            ZCTA5CE20=df["ZCTA5CE20"].astype(str)
        ),
        on="ZCTA5CE20",
        how="left",
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 10))

    merged.plot(
        column=average_summary,
        cmap=cmap,
        legend=True,
        edgecolor="black",
        linewidth=0.15,
        ax=ax,
        missing_kwds={"color": "lightgrey", "label": "No data"},
    )
    ax.set_title(title or average_summary)
    ax.set_axis_off()
    plt.tight_layout()
    return ax
