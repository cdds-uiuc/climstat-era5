"""
climstat: Climate statistics pipeline for ERA5 reanalysis data over Illinois.

This package turns raw ERA5 gridded climate data into county-level and
ZIP-code-level summary statistics for Illinois.  Three top-level functions
correspond to the three pipeline modules:

    acquire_data()    — download ERA5 + IFS raw data
    process_data()    — metrics → statistics → aggregation → CSV
    visualize_data()  — load CSVs and produce plots

Subpackages
-----------
data_acquisition  — ERA5 and IFS deterministic forecast downloads.
data_process      — Derived metrics, statistics, spatial aggregation, CSV export.
data_visualization — Time series, choropleth maps, heatmaps.

The wrapper notebook ``climstat_pipeline.ipynb`` calls the three top-level
functions and is the main entry point.
"""

from . import data_acquisition
from . import data_process
from . import data_visualization

from .data_acquisition import acquire_data
from .data_process import process_data
from .data_visualization import visualize_data
