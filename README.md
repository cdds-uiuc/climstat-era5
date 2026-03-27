# climstat-era5

This repository contains a modular pipeline to extract variables from ERA5 gridded climate reanalysis data, compute derived heat stress metrics, calculate summary statistics, and output county-level `.csv` files for Illinois. The data supports the climate database for the **Illinois Department of Public Health and Illinois Emergency Management Agency (IEMA)**. This work builds on previous work by by Maile Sasaki and Rachel Tam

---

## Supported Dataset

- **ERA5** reanalysis at 0.25° resolution (~28 km), accessed from the [Google Cloud ARCO-ERA5 Zarr store](https://cloud.google.com/storage/docs/public-datasets/era5)

---

## Supported Climate Metrics

| Metric | Unit | Notes |
|---|---|---|
| Heat Index | °F | NWS multi-step Rothfusz regression |
| Wet Bulb Temperature | °F | Stull (2011) empirical formula |
| Wet Bulb Globe Temperature (WBGT) | °F | Simplified shaded formula: 0.7×Twb + 0.3×T2m |
| Apparent Temperature | °F | ECMWF formula |
| Humidex | °F | Canadian humidity-temperature index |
| Normal Effective Temperature | °F | Comfort index |
| Wind Chill | °F | NWS formula; defined only for T ≤ 50°F and wind > 3 mph |

---

## Statistics Computed

For each climate metric, the pipeline computes two levels of summary statistics:

**Daily statistics** (one value per day per county — a time series):
- Daily mean, daily max, daily min
- Hours above each threshold per day (e.g. hours above 90°F)
- Hours below each threshold per day (e.g. hours below 32°F for wind chill)

**Averages** (one value per county — climatological averages over the full period):
- Average daily max, average daily mean, average daily min
- Average days per year above/below each threshold
- Average hours per year above/below each threshold

---

## Pipeline Workflow

```
┌──────────────────────────────────────────────────────────┐
│  climstat_pipeline.ipynb  (single entry point)           │
│  Parameters: DATA_DIR, time interval, metrics,           │
│              thresholds                                  │
│                                                          │
│  ┌────────┐   ┌────────┐   ┌────────┐   ┌────────────┐   │
│  │ Step 1 │──▶│ Step 2 │──▶│ Step 3 │──▶│ Step 4a/4b │   │
│  │Extract │   │Metrics │   │Summary │   │County/ZIP  │   │
│  │ ERA5   │   │ Calc   │   │ Stats  │   │Agg + Viz   │   │
│  └────────┘   └────────┘   └────────┘   └────────────┘   │
└──────────────────────────────────────────────────────────┘
        │              │             │             │
        ▼              ▼             ▼             ▼
   era5_extract.py  metrics.py  statistics.py  county_agg.py
   (+ cache check)  (Tier 0-3)  (daily/averages) zipcode_agg.py
```

**Step 1 — Extract ERA5 data** (`climstat/era5_extract.py`)

Downloads the required variables from Google Cloud and caches them locally as NetCDF files — **one file per variable per year**. Each cached file is named with the variable and the actual date range it contains:

```
data/cache/                                                ← raw ERA5 NetCDF downloads
├── era5_2m_temperature_20160101_20161231.nc               ← complete past year
├── era5_2m_temperature_20170101_20171231.nc
├── ...
├── era5_2m_temperature_20260101_20260321.nc               ← current (incomplete) year
├── era5_2m_dewpoint_temperature_20160101_20161231.nc
└── ...
```

On subsequent runs, completed past years load instantly from cache (no network access needed). The current year is re-downloaded automatically when the cached file becomes more than 7 days stale.

**Step 2 — Compute derived metrics** (`climstat/metrics.py`)

Applies the heat-metric formulas to the raw ERA5 fields. Metrics are organized in dependency tiers (Tier 1: wind, vapor pressure, RH → Tier 2: HI, WC, WBT, etc. → Tier 3: WBGT). Any subset of metrics can be requested by name via the `METRIC_REGISTRY`.

**Step 3 — Compute summary statistics** (`climstat/statistics.py`)
- **Daily summary**: daily mean / max / min, and hours above each threshold (time series)
- **Averages**: average daily max / mean / min, average days per year above threshold, average hours per year above threshold (one value per grid cell)

**Step 4a — Aggregate to county level** (`climstat/county_agg.py`)

Spatially joins ERA5 grid points to Illinois county polygons (via the bundled Census TIGER/Line shapefile) and averages all statistics within each county. Uses `where` and `columns` parameters for efficient shapefile loading (only IL rows and needed columns are read from disk). Outputs a `pd.DataFrame` indexed by county.

**Step 4b — Aggregate to ZIP code level** (`climstat/zipcode_agg.py`)

Maps each Illinois ZCTA (ZIP Code Tabulation Area) to its nearest ERA5 grid point via centroid distance (`sjoin_nearest`), since most ZCTAs are smaller than a single grid cell. Uses `mask` for spatial filtering and `columns` for selective column loading when reading the ZCTA shapefile. Supports a precomputed mapping (`build_zcta_mapping()`) to avoid repeated shapefile I/O across metrics.

**Step 5 — Save outputs**

County-level and ZIP-code-level CSVs are written to `data/output/`.

**Step 6 — Visualize** (`climstat/visualization.py`)

Five plot types: county time-series line plot, county choropleth map, threshold exceedance heatmap (region × year), ZIP-code time-series, and ZIP-code choropleth map.

---

## Project Structure

```
climstat-era5/
├── climstat_pipeline.ipynb        # Single entry-point notebook
├── environment.yml                # Conda environment
├── climstat/                      # Python package
│   ├── __init__.py                # Makes climstat/ importable; loads all submodules
│   ├── era5_extract.py            # Step 1: ERA5 extraction + per-variable-per-year caching
│   ├── metrics.py                 # Step 2: heat metric formulas (Tier 0-3)
│   ├── statistics.py              # Step 3: daily and averages summary statistics
│   ├── county_agg.py              # Step 4a: spatial aggregation to IL counties
│   ├── zipcode_agg.py             # Step 4b: spatial aggregation to IL ZIP codes (ZCTAs)
│   └── visualization.py           # Step 6: plotting utilities (counties + ZIP codes)
│
└── data/
    ├── cache/                       # Raw ERA5 NetCDF downloads (one per variable per year)
    ├── shapefiles/
    │   ├── county/                # tl_2025_us_county.shp (Census TIGER/Line)
    │   └── zipcodes/              # tl_2025_us_zcta520.shp
    └── output/                    # Final county-level CSVs (pipeline output)
```

---

## Quick Start

**1. Download required shapefiles:**

The pipeline requires US Census TIGER/Line shapefiles, which are too large for GitHub. Download and unzip them into `data/shapefiles/` before running:

- **County shapefile** — download `tl_2025_us_county.zip` from the [Census Bureau](https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2025&layergroup=Counties+%28and+equivalent%29) and unzip into `data/shapefiles/county/`
- **ZIP Code (ZCTA) shapefile** — download `tl_2025_us_zcta520.zip` from the [Census Bureau](https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2025&layergroup=ZIP+Code+Tabulation+Areas) and unzip into `data/shapefiles/zipcodes/`

**2.(Optional) Request ERA5 pre-cached data:**
The pipeline downloads ERA5 data and caches a local version. If the cached version exists, subsequent runs do not need to download the data again. This download process is slow because of the way the data is chunked (see [Download Performance Note](#download-performance-note)). You can speed this up by requesting pre-cached data from cristi. 


**3. Create and activate the conda environment:**
```bash
conda env create -f environment.yml
conda activate iema
python -m ipykernel install --user --name iema --display-name "Python (iema)"
```

**4. Open `climstat_pipeline.ipynb` and select the `Python (iema)` kernel:**
- **VS Code**: Open the notebook, click the kernel picker in the top right, and select `Python (iema)`
- **Jupyter Lab / Notebook**: Run `jupyter lab` (or `jupyter notebook`), open `climstat_pipeline.ipynb`, then select the `Python (iema)` kernel from the **Kernel → Change Kernel** menu

**5. Edit the Parameters cell:**
```python
DATA_DIR   = "data"          # all sub-paths are derived from this
YEAR_START = 2016
YEAR_END   = 2025
METRICS    = ["heat_index", "wbgt", "wind_chill"]
THRESHOLDS = {
    "heat_index": [90, 104],
    "wbgt":       [80, 85],
    "wind_chill": [],
}
```

**6. Run all cells.**

- (optional) before first run: request the pre-cached data from cristi (e.g. via sharepoint for University of Illinois users)
- First run: ERA5 data is downloaded from Google Cloud and cached as one NetCDF file per variable per year (~5 min per file).
- Subsequent runs: cached past years load from disk instantly. Only the current year may need refreshing.

---

## Caching Details

| Scenario | Behavior |
|---|---|
| Past year, file exists in `data/cache/` | Loaded from disk — no internet needed |
| Past year, file missing | Downloaded from Google Cloud and saved |
| Current year, file exists and < 7 days old | Loaded from disk |
| Current year, file > 7 days old or missing | Re-downloaded to get latest available data |

Cache filenames encode the variable and actual date range:
```
era5_{variable}_{YYYYMMDD}_{YYYYMMDD}.nc
```
The end date reflects the last timestamp actually present in the file (which may lag a few days behind today for the current year, since ERA5 data is not available in real-time).

---

## Download Performance Note

The ARCO-ERA5 Zarr store is chunked as `{time: 1, latitude: 721, longitude: 1440}` — each chunk is **one timestep covering the entire globe**. Illinois uses only ~480 of the ~1 million grid points per chunk, so roughly **2000x more data is transferred than retained**. This is a property of the store's chunk layout and cannot be avoided client-side.

As a result, first-run downloads are slow (~5 min per variable per year on a typical home connection). The pipeline downloads month-by-month to keep memory bounded and show progress, but the network transfer is the bottleneck.

---

## Output Format

Each metric produces two CSVs in `data/output/`:

| File | Contents |
|---|---|
| `daily_{metric}_{start}_{end}_counties.csv` | Daily mean/max/min and hours above each threshold, per county |
| `averages_{metric}_{start}_{end}_counties.csv` | Climatological averages (avg daily max/mean/min, avg days/hours per year above threshold), per county |
| `daily_{metric}_{start}_{end}_zipcodes.csv` | Daily mean/max/min and hours above each threshold, per ZIP code |
| `averages_{metric}_{start}_{end}_zipcodes.csv` | Climatological averages, per ZIP code |

County files are indexed by `GEOID` and `NAMELSAD`. ZIP code files are indexed by `ZCTA5CE20` (5-digit ZIP code).

---

## Contact

Cristi Proistosescu — cristi@illinois.edu
