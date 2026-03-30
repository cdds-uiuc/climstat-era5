"""
Interactive Panel dashboard for ERA5/IFS climate statistics.

Run with:  panel serve climstat/data_visualization/dashboard_panel.py --show
Opens at:  http://localhost:5006/
"""

# ══════════════════════════════════════════════════════════════════════
# Section 0: Imports and constants
# ══════════════════════════════════════════════════════════════════════

import os
import sys

# Ensure the repo root is on sys.path so absolute imports work
# when Panel runs this file as a script.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import geopandas as gpd
import pandas as pd
import panel as pn
import holoviews as hv
import hvplot.pandas  # noqa: F401 — registers .hvplot accessor

from climstat.data_visualization.dashboard_data import (
    scan_output_dir,
    get_available_metrics,
    get_date_ranges,
    load_metric_data,
    get_daily_columns,
    get_averages_columns,
    get_regions,
)
from climstat.data_process.shapefiles import load_illinois_counties, load_illinois_zctas

pn.extension("bokeh")

# Paths (relative to repo root — panel serve runs from there)
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")
SHAPEFILE_PATH = os.path.join(DATA_DIR, "shapefiles", "county", "tl_2025_us_county.shp")
ZCTA_SHAPEFILE_PATH = os.path.join(DATA_DIR, "shapefiles", "zipcodes", "tl_2025_us_zcta520.shp")

# Colors matching the matplotlib dashboard
ERA5_COLOR = "#1f77b4"
IFS_GAP_COLOR = "#2ca02c"
IFS_FORECAST_COLOR = "#ff7f0e"

# ══════════════════════════════════════════════════════════════════════
# Section 1: Data loading (cached)
# ══════════════════════════════════════════════════════════════════════

CATALOG = scan_output_dir(OUTPUT_DIR)
METRICS = get_available_metrics(CATALOG)
DATE_RANGES = get_date_ranges(CATALOG)


SIMPLIFIED_ZCTA_PATH = os.path.join(DATA_DIR, "shapefiles", "zipcodes", "il_zcta_simplified.geojson")


@pn.cache
def cached_counties_gdf():
    return load_illinois_counties(SHAPEFILE_PATH)


@pn.cache
def cached_zctas_gdf():
    """Load ZCTAs, using a simplified cached version for faster rendering."""
    if os.path.exists(SIMPLIFIED_ZCTA_PATH):
        gdf = gpd.read_file(SIMPLIFIED_ZCTA_PATH)
        gdf["ZCTA5CE20"] = gdf["ZCTA5CE20"].astype(str)
        return gdf
    # First run: load full ZCTAs, simplify, and save
    gdf = load_illinois_zctas(ZCTA_SHAPEFILE_PATH, SHAPEFILE_PATH)
    gdf["geometry"] = gdf.geometry.simplify(tolerance=0.005, preserve_topology=True)
    gdf.to_file(SIMPLIFIED_ZCTA_PATH, driver="GeoJSON")
    return gdf


def load_data(metric, source, region_type):
    """Load daily + averages for a metric/source/region combo."""
    return load_metric_data(CATALOG, metric, source, region_type)


# ══════════════════════════════════════════════════════════════════════
# Section 2: Widget definitions
# ══════════════════════════════════════════════════════════════════════

# Global controls
w_metric = pn.widgets.Select(name="Metric", options=METRICS, value=METRICS[0])
w_region_type = pn.widgets.RadioButtonGroup(
    name="Region Type",
    options={"Counties": "counties", "ZIP Codes": "zipcodes"},
    value="counties",
    button_type="primary",
    button_style="outline",
)

# Tab 1: Time Series
w_ts_region = pn.widgets.Select(name="Region", options=[], width=250)
w_ts_stat = pn.widgets.Select(name="Statistic", options=[], width=250)
w_ts_recent = pn.widgets.Checkbox(name="Recent 60 days", value=False)

# Tab 2: Spatial Map
w_map_stat = pn.widgets.Select(name="Statistic", options=[], width=300)
w_map_cmap = pn.widgets.Select(
    name="Colormap",
    options=["YlOrRd", "Blues", "coolwarm", "RdYlGn_r", "viridis", "plasma"],
    value="YlOrRd",
    width=150,
)

# ══════════════════════════════════════════════════════════════════════
# Section 3: Dependent dropdown updaters
# ══════════════════════════════════════════════════════════════════════


def update_ts_dropdowns(metric, region_type):
    """Update time series dropdowns when metric or region type changes."""
    era5 = load_data(metric, "ERA5", region_type)
    daily = era5["daily"]

    regions = get_regions(daily, region_type)
    w_ts_region.options = regions
    if regions:
        w_ts_region.value = regions[0]

    cols = get_daily_columns(daily)
    w_ts_stat.options = cols
    if cols:
        w_ts_stat.value = cols[0]


def update_map_dropdowns(metric, region_type):
    """Update map stat dropdown when metric or region type changes."""
    era5 = load_data(metric, "ERA5", region_type)
    avgs = era5["averages"]

    cols = get_averages_columns(avgs)
    w_map_stat.options = cols
    if cols:
        w_map_stat.value = cols[0]


# Watch global controls and fire updaters
w_metric.param.watch(lambda e: update_ts_dropdowns(w_metric.value, w_region_type.value), "value")
w_metric.param.watch(lambda e: update_map_dropdowns(w_metric.value, w_region_type.value), "value")
w_region_type.param.watch(lambda e: update_ts_dropdowns(w_metric.value, w_region_type.value), "value")
w_region_type.param.watch(lambda e: update_map_dropdowns(w_metric.value, w_region_type.value), "value")


# ══════════════════════════════════════════════════════════════════════
# Section 4: Tab 1 — Time Series plot builder
# ══════════════════════════════════════════════════════════════════════


def build_timeseries(metric, region_type, region, stat, recent):
    """Build an interactive time series plot for the selected parameters."""
    if not region or not stat:
        return hv.Text(0, 0, "Select a region and statistic")

    region_col = "NAMELSAD" if region_type == "counties" else "ZCTA5CE20"

    # Load ERA5 data
    era5 = load_data(metric, "ERA5", region_type)
    daily = era5["daily"]
    if daily is None or stat not in daily.columns:
        return hv.Text(0, 0, "No data available")

    subset = daily[daily[region_col] == region].copy()
    subset["time"] = pd.to_datetime(subset["time"])
    subset = subset.sort_values("time")

    # Apply date range filter
    if recent:
        start = pd.Timestamp.today() - pd.Timedelta(days=60)
        end = pd.Timestamp.today() + pd.Timedelta(days=14)
        subset = subset[(subset["time"] >= start) & (subset["time"] <= end)]

    # ERA5 line
    plot = subset.hvplot.line(
        x="time", y=stat,
        color=ERA5_COLOR, label="ERA5",
        hover_cols=["time", stat],
        line_width=1.5,
    )

    # IFS overlay
    ifs = load_data(metric, "IFS", region_type)
    ifs_daily = ifs["daily"]
    if ifs_daily is not None and stat in ifs_daily.columns:
        fc = ifs_daily[ifs_daily[region_col] == region].copy()
        fc["time"] = pd.to_datetime(fc["time"])
        fc = fc.sort_values("time")

        if recent:
            fc = fc[fc["time"] >= start]

        if len(fc) > 0:
            today = pd.Timestamp.today().normalize()

            # Gap-fill (past)
            gap = fc[fc["time"] <= today]
            if len(gap) > 0:
                plot *= gap.hvplot.line(
                    x="time", y=stat,
                    color=IFS_GAP_COLOR, label="IFS Gap-fill",
                    hover_cols=["time", stat],
                    line_width=1.5,
                )

            # Forecast (future)
            fcast = fc[fc["time"] >= today]
            if len(fcast) > 0:
                plot *= fcast.hvplot.line(
                    x="time", y=stat,
                    color=IFS_FORECAST_COLOR, label="IFS Forecast",
                    hover_cols=["time", stat],
                    line_width=1.5, line_dash="dashed",
                )

            # Today line
            plot *= hv.VLine(today).opts(
                color="grey", line_dash="dashed", line_width=1, alpha=0.6,
            )

    title = f"{metric}: {stat} — {region}"
    plot = plot.opts(
        title=title,
        xlabel="Date", ylabel=stat,
        frame_height=400, responsive=True,
        legend_position="top_left",
        tools=["pan", "wheel_zoom", "box_zoom", "reset", "hover"],
        active_tools=["pan", "wheel_zoom"],
    )
    return plot


ts_bound = pn.bind(
    build_timeseries,
    metric=w_metric,
    region_type=w_region_type,
    region=w_ts_region,
    stat=w_ts_stat,
    recent=w_ts_recent,
)


# ══════════════════════════════════════════════════════════════════════
# Section 5: Tab 2 — Spatial Map plot builder
# ══════════════════════════════════════════════════════════════════════


def build_choropleth(metric, region_type, map_stat, cmap):
    """Build an interactive choropleth map for the selected parameters."""
    if not map_stat:
        return hv.Text(0, 0, "Select a statistic")

    era5 = load_data(metric, "ERA5", region_type)
    avgs = era5["averages"]
    if avgs is None or map_stat not in avgs.columns:
        return hv.Text(0, 0, "No averages data available")

    # Load the appropriate GeoDataFrame
    if region_type == "counties":
        gdf = cached_counties_gdf().copy()
        key_col = "GEOID"
        name_col = "NAMELSAD"
        avgs_merge = avgs[[key_col, map_stat]].copy()
        avgs_merge[key_col] = avgs_merge[key_col].astype(str)
    else:
        gdf = cached_zctas_gdf().copy()
        key_col = "ZCTA5CE20"
        name_col = "ZCTA5CE20"
        avgs_merge = avgs[[key_col, map_stat]].copy()
        avgs_merge[key_col] = avgs_merge[key_col].astype(str)

    # Left-join averages onto geometry
    merged = gdf.merge(avgs_merge, on=key_col, how="left")

    # Background layer for missing data
    missing = merged[merged[map_stat].isna()]
    has_data = merged[merged[map_stat].notna()]

    layers = []
    if len(missing) > 0:
        layers.append(
            missing.hvplot(
                geo=True, color="lightgrey",
                line_color="black", line_width=0.3,
                hover=False,
            )
        )

    if len(has_data) > 0:
        hover_cols = [name_col, map_stat] if name_col in has_data.columns else [map_stat]
        layers.append(
            has_data.hvplot(
                geo=True, c=map_stat, cmap=cmap,
                hover_cols=hover_cols,
                line_color="black", line_width=0.3,
                colorbar=True, clabel=map_stat,
            )
        )

    if not layers:
        return hv.Text(0, 0, "No data to display")

    plot = layers[0]
    for layer in layers[1:]:
        plot *= layer

    title = f"{metric}: {map_stat} — {'Counties' if region_type == 'counties' else 'ZIP Codes'}"
    plot = plot.opts(
        title=title,
        frame_height=700, frame_width=400,
        xaxis=None, yaxis=None,
        tools=["pan", "wheel_zoom", "box_zoom", "reset", "hover"],
        active_tools=["pan", "wheel_zoom"],
    )
    return plot


map_bound = pn.bind(
    build_choropleth,
    metric=w_metric,
    region_type=w_region_type,
    map_stat=w_map_stat,
    cmap=w_map_cmap,
)


# ══════════════════════════════════════════════════════════════════════
# Section 6: Layout assembly
# ══════════════════════════════════════════════════════════════════════

header = pn.pane.Markdown(
    "# Climate Statistics Dashboard\n"
    "ERA5/IFS heat and cold metrics for Illinois counties and ZIP codes.",
    sizing_mode="stretch_width",
)

global_controls = pn.Row(w_metric, w_region_type, sizing_mode="stretch_width")

ts_controls = pn.Row(w_ts_region, w_ts_stat, w_ts_recent)
ts_tab = pn.Column(ts_controls, pn.panel(ts_bound, loading_indicator=True))

map_controls = pn.Row(w_map_stat, w_map_cmap)
map_tab = pn.Column(map_controls, pn.panel(map_bound, loading_indicator=True))

tabs = pn.Tabs(
    ("Spatial Map", map_tab),
    ("Time Series", ts_tab),
)

# ══════════════════════════════════════════════════════════════════════
# Section 7: Initialize and serve
# ══════════════════════════════════════════════════════════════════════

# Populate dropdowns with initial metric data
update_ts_dropdowns(METRICS[0], "counties")
update_map_dropdowns(METRICS[0], "counties")

app = pn.Column(header, global_controls, pn.layout.Divider(), tabs)
app.servable(title="Climate Statistics — ERA5/IFS Illinois")
