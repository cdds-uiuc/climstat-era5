# climstat-era5

This repository contains a modular pipeline to extract variables from ERA5 gridded climate reanalysis data, compute derived heat stress metrics, calculate summary statistics, and output county-level and ZIP-code-level `.csv` files for Illinois. The data supports the climate database for the **Illinois Department of Public Health and Illinois Emergency Management Agency (IEMA)**. This work builds on previous work by Maile Sasaki and Rachel Tam.

---

## Supported Datasets

- **ERA5** reanalysis at 0.25° resolution (~28 km), accessed from the [Google Cloud ARCO-ERA5 Zarr store](https://cloud.google.com/storage/docs/public-datasets/era5)
- **IFS** (ECMWF Integrated Forecasting System) via the [Open-Meteo ECMWF API](https://open-meteo.com/en/docs/ecmwf-api), used to fill the ~5-7 day gap between ERA5 availability and the present day

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
| 2m Temperature | °F | Raw ERA5 surface temperature (converted from K) |

---

## Statistics Computed

For each climate metric, the pipeline computes two levels of summary statistics:

**Daily statistics** (one value per day per county/ZIP — a time series):
- Daily mean, daily max, daily min
- Hours above each threshold per day (e.g. hours above 90°F)
- Hours below each threshold per day (e.g. hours below 32°F for wind chill)

**Averages** (one value per county/ZIP — climatological averages over the full period):
- Average daily max, average daily mean, average daily min
- Average days per year above/below each threshold
- Average hours per year above/below each threshold

> **Note:** IFS data produces daily statistics only (no period averages), since it covers only a short gap period.

---

## Pipeline Architecture

The pipeline is organized into three modules, called from a simplified notebook:

```
┌───────────────────────────────────────────────────────────────┐
│  climstat_pipeline.ipynb                                      │
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐     │
│  │  Module 1    │  │  Module 2    │  │  Module 3        │     │
│  │  acquire_    │─▶│  process_    │─▶│  visualize_      │     │
│  │  data()      │  │  data()      │  │  data()          │     │
│  └──────────────┘  └──────────────┘  └──────────────────┘     │
└───────────────────────────────────────────────────────────────┘
       │                    │                    │
       ▼                    ▼                    ▼
  era5_extract.py     metrics.py          visualization.py
  ifs_extract.py      statistics.py        (reads from CSV)
                      county_agg.py
                      zipcode_agg.py
```

### Module 1 — Acquire raw data (`acquire_data()`)

Downloads ERA5 data from Google Cloud and caches it locally as NetCDF files — **one file per variable per year**. Then fills the gap between the last valid ERA5 day and yesterday with IFS data from Open-Meteo, cached as **one file per variable** (matching the ERA5 pattern). Only the ERA5 variables needed by the requested metrics are downloaded for both ERA5 and IFS.

ERA5 trailing NaN timesteps are automatically trimmed to find the true last day of valid data.

### Module 2 — Process data (`process_data()`)

For each requested metric:
1. **Compute derived metrics** from raw ERA5/IFS fields (heat index, WBGT, etc.)
2. **Compute summary statistics** — daily mean/max/min and threshold exceedance hours
3. **Aggregate to counties** via spatial join with Census TIGER/Line shapefiles
4. **Aggregate to ZIP codes** via nearest-centroid mapping to ZCTAs
5. **Save output CSVs** — separate files for ERA5 and IFS data

Includes CSV-level caching: if output CSVs already exist for a metric's date range, that metric is skipped.

### Module 3 — Visualize (`visualize_data()`)

Reads from CSV files only — can run independently. Produces:
- County and ZIP code time series (full range + last 2 months zoom), with ERA5 and IFS data overlaid in different colors
- County and ZIP code choropleth maps of average statistics (ERA5 only)

---

## Project Structure

```
climstat-era5/
├── climstat_pipeline.ipynb        # Entry-point notebook (3 module calls)
├── environment.yml                # Conda environment
├── climstat/                      # Python package
│   ├── __init__.py                # Package init; exports acquire/process/visualize
│   ├── era5_extract.py            # ERA5 download + per-variable-per-year caching
│   ├── ifs_extract.py             # IFS download + per-variable caching
│   ├── metrics.py                 # Heat/cold metric formulas (METRIC_REGISTRY)
│   ├── statistics.py              # Daily and period-average summary statistics
│   ├── shapefiles.py              # Shared spatial utilities (IL county/ZCTA loading)
│   ├── county_agg.py              # Spatial aggregation to IL counties
│   ├── zipcode_agg.py             # Spatial aggregation to IL ZIP codes (ZCTAs)
│   ├── pipeline.py                # Orchestration: acquire, process, visualize
│   └── visualization.py           # Plotting utilities (time series, maps, heatmaps)
│
└── data/
    ├── cache/                     # Raw NetCDF downloads
    │   ├── era5_{var}_{start}_{end}.nc   # One file per variable per year
    │   └── ifs_{var}_{start}_{end}.nc    # One file per variable (gap period)
    ├── shapefiles/
    │   ├── county/                # tl_2025_us_county.shp (Census TIGER/Line)
    │   └── zipcodes/              # tl_2025_us_zcta520.shp
    └── output/                    # Final CSVs (counties + ZIP codes)
```

---

## Quick Start

**1. Download required shapefiles:**

The pipeline requires US Census TIGER/Line shapefiles, which are too large for GitHub. Download and unzip them into `data/shapefiles/` before running:

- **County shapefile** — download `tl_2025_us_county.zip` from the [Census Bureau](https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2025&layergroup=Counties+%28and+equivalent%29) and unzip into `data/shapefiles/county/`
- **ZIP Code (ZCTA) shapefile** — download `tl_2025_us_zcta520.zip` from the [Census Bureau](https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2025&layergroup=ZIP+Code+Tabulation+Areas) and unzip into `data/shapefiles/zipcodes/`

**2. (Optional) Request ERA5 pre-cached data:**
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
DATA_DIR   = "data"
YEAR_START = 2020
YEAR_END   = 2026
METRICS    = ["heat_index", "wbgt", "wind_chill", "2m_temperature"]
THRESHOLDS = {
    "heat_index":      {"above": [90, 104]},
    "wbgt":            {"above": [80, 85]},
    "wind_chill":      {"below": [-25, -15, 0, 32]},
    "2m_temperature":  {"above": [95, 100], "below": [0, 32]},
}
```

**6. Run all cells.**

- (optional) Before first run: request the pre-cached data from cristi (e.g. via SharePoint for University of Illinois users)
- First run: ERA5 data is downloaded from Google Cloud and cached as one NetCDF file per variable per year (~5 min per file).
- Subsequent runs: cached data loads from disk instantly. Only the current year may need refreshing (if >14 days stale).

---

## Caching Details

### Raw data caching (ERA5)

| Scenario | Behavior |
|---|---|
| Past year, file exists in `data/cache/` | Loaded from disk — no internet needed |
| Past year, file missing | Downloaded from Google Cloud and saved |
| Current year, file exists and < 14 days old | Loaded from disk |
| Current year, file > 14 days old or missing | Re-downloaded to get latest available data |

Cache filenames encode the variable and actual date range:
```
era5_{variable}_{YYYYMMDD}_{YYYYMMDD}.nc
```
The end date reflects the last timestamp with valid (non-NaN) data, which typically lags ~5-7 days behind today.

### Raw data caching (IFS)

IFS data fills the gap between the last valid ERA5 day and yesterday (today's data is excluded to avoid incomplete observations). Cache files follow the same per-variable pattern:
```
ifs_{variable}_{YYYYMMDD}_{YYYYMMDD}.nc
```
IFS cache files are refreshed if older than 6 hours.

### CSV output caching

Output CSVs act as a processing cache. If all expected CSVs exist for a metric's date range, that metric is skipped during processing. This means only new or modified metrics need recomputation.

---

## Output Format

### ERA5 outputs (4 files per metric)

| File | Contents |
|---|---|
| `daily_{metric}_{start}_{end}_ERA5_counties.csv` | Daily mean/max/min and hours above/below thresholds, per county |
| `averages_{metric}_{start}_{end}_ERA5_counties.csv` | Climatological averages, per county |
| `daily_{metric}_{start}_{end}_ERA5_zipcodes.csv` | Daily statistics, per ZIP code |
| `averages_{metric}_{start}_{end}_ERA5_zipcodes.csv` | Climatological averages, per ZIP code |

### IFS outputs (2 files per metric, daily only)

| File | Contents |
|---|---|
| `daily_{metric}_{start}_{end}_IFS_counties.csv` | Daily statistics, per county |
| `daily_{metric}_{start}_{end}_IFS_zipcodes.csv` | Daily statistics, per ZIP code |

County files are indexed by `GEOID` and `NAMELSAD`. ZIP code files are indexed by `ZCTA5CE20` (5-digit ZIP code string).

---

## Download Performance Note

The ARCO-ERA5 Zarr store is chunked as `{time: 1, latitude: 721, longitude: 1440}` — each chunk is **one timestep covering the entire globe**. Illinois uses only ~480 of the ~1 million grid points per chunk, so roughly **2000x more data is transferred than retained**. This is a property of the store's chunk layout and cannot be avoided client-side.

As a result, first-run downloads are slow (~5 min per variable per year on a typical home connection). The pipeline downloads month-by-month to keep memory bounded and show progress, but the network transfer is the bottleneck.

---

## Contact

Cristi Proistosescu — cristi@illinois.edu
