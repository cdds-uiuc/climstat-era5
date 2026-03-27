"""
climstat: Climate statistics pipeline for ERA5 reanalysis data over Illinois.

This package turns raw ERA5 gridded climate data into county-level summary
statistics for Illinois.  The pipeline has four stages, each in its own module:

Modules
-------
era5_extract  — Download ERA5 data from Google Cloud and cache it locally
                as NetCDF files (one file per variable per year).
metrics       — Compute derived heat/cold stress indices (Heat Index,
                Wind Chill, WBGT, etc.) from the raw ERA5 fields.
statistics    — Aggregate hourly metric data into daily and averages
                (climatological) summary statistics.
county_agg    — Map the 0.25°-resolution ERA5 grid onto Illinois county
                polygons and average values within each county.
visualization — Plotting utilities (time series, choropleth maps, heatmaps).

How the modules connect
-----------------------
    era5_extract  -->  metrics  -->  statistics  -->  county_agg
    (raw NetCDF)     (hourly      (daily &         (county-level
                      DataArrays)  averages)        DataFrames)

The wrapper notebook ``climstat_pipeline.ipynb`` calls these modules
in sequence and is the main entry point for running the pipeline.
"""

# These "from . import ..." lines make the submodules accessible as
# attributes of the package.  For example, after "import climstat" you
# can write climstat.metrics.compute_metrics(...) without a separate
# "import climstat.metrics" statement.
from . import era5_extract
from . import metrics
from . import statistics
from . import county_agg
from . import visualization
