"""
climstat: Climate statistics pipeline for ERA5 reanalysis data over Illinois.

This package turns raw ERA5 gridded climate data into county-level and
ZIP-code-level summary statistics for Illinois.  Three top-level functions
in ``pipeline`` correspond to the three pipeline modules:

    acquire_data()    — download ERA5 + IFS raw data
    process_data()    — metrics → statistics → aggregation → CSV
    visualize_data()  — load CSVs and produce plots

Supporting modules
------------------
era5_extract  — Download ERA5 from Google Cloud ARCO-ERA5 Zarr store.
ifs_extract   — Fill the ~5-7 day ERA5 gap with IFS data from Open-Meteo.
metrics       — Derived heat/cold indices (Heat Index, WBGT, Wind Chill, etc.)
                plus raw 2m_temperature.
statistics    — Daily summaries and period averages.
shapefiles    — Shared spatial utilities (IL county/ZCTA loading, grid points).
county_agg    — Spatial join to IL counties.
zipcode_agg   — Nearest-centroid aggregation to IL ZCTAs.
visualization — Time series, choropleth maps, heatmaps.

The wrapper notebook ``climstat_pipeline.ipynb`` calls the three top-level
functions and is the main entry point.
"""

from . import era5_extract
from . import ifs_extract
from . import metrics
from . import statistics
from . import shapefiles
from . import county_agg
from . import zipcode_agg
from . import pipeline
from . import visualization

from .pipeline import acquire_data, process_data, visualize_data
