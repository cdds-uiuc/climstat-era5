"""
Step 2: Compute derived climate / heat-stress metrics from raw ERA5 fields.

What this module does
---------------------
ERA5 provides basic atmospheric variables (temperature, wind, humidity).
From these, we compute higher-level metrics that describe how hot, cold,
or uncomfortable the weather feels to humans — things like the Heat Index
or Wind Chill.

The metrics are organized in dependency "tiers":

    Tier 0  — raw ERA5 output (t2m, u10, v10, dewpoint, precip)
              These come straight from the downloaded data.

    Tier 1  — depend only on raw ERA5 variables
              wind_magnitude, vapor_pressure, relative_humidity
              These are intermediate quantities needed by Tier 2.

    Tier 2  — depend on raw ERA5 + Tier 1
              heat_index, wind_chill, wet_bulb_temperature,
              apparent_temperature, humidex, normal_effective_temperature
              These are the main user-facing metrics.

    Tier 3  — depend on raw ERA5 + Tier 1 + Tier 2
              wbgt (Wet Bulb Globe Temperature)
              This metric needs wet_bulb_temperature (Tier 2) as an input.

The user-facing functions (Tier 2 and 3) take raw ERA5 DataArrays as input
and internally call whatever Tier 1 helpers they need, so you don't have to
manually compute intermediate quantities.

Units
-----
- ERA5 temperatures are in Kelvin (K).  Most metric functions accept Kelvin
  and either return Kelvin or Fahrenheit — check each function's docstring.
- The compute_metrics() convenience function can auto-convert everything
  to Fahrenheit if desired (the default).

Key Python concepts used here
-----------------------------
- **xr.where(condition, x, y)**: element-wise "if condition then x, else y".
  Works on entire arrays at once (vectorized), no for-loops needed.
- **np.nan**: "Not a Number" — used as a fill value for grid cells where a
  metric is not physically defined (e.g. Wind Chill when T > 50°F).
- **lambda**: A one-line anonymous function.  Used in METRIC_REGISTRY to
  define how to call each metric function given a raw ERA5 Dataset.
"""

import numpy as np
import xarray as xr


# ===========================================================================
# Unit conversion helpers
# ===========================================================================

def k_to_f(temp_k: xr.DataArray) -> xr.DataArray:
    """Kelvin -> Fahrenheit.  Formula: °F = (K - 273.15) * 1.8 + 32"""
    return (temp_k - 273.15) * 1.8 + 32


def f_to_k(temp_f: xr.DataArray) -> xr.DataArray:
    """Fahrenheit -> Kelvin.  Formula: K = (°F - 32) / 1.8 + 273.15"""
    return (temp_f - 32) / 1.8 + 273.15


def k_to_c(temp_k: xr.DataArray) -> xr.DataArray:
    """Kelvin -> Celsius.  Formula: °C = K - 273.15"""
    return temp_k - 273.15


# ===========================================================================
# Tier 1 — depend only on raw ERA5 variables
#
# These are intermediate quantities that the Tier 2 metric functions call
# internally.  You generally don't need to call these yourself.
# ===========================================================================

def wind_magnitude(u10: xr.DataArray, v10: xr.DataArray) -> xr.DataArray:
    """
    Compute wind speed magnitude from the east-west (u) and north-south (v)
    wind components.

    u10 and v10 are the two perpendicular components of wind velocity at
    10 m above the surface.  The magnitude (total wind speed) is just the
    Pythagorean combination: sqrt(u^2 + v^2).

    Input units:  m/s
    Output units: m/s
    """
    return np.sqrt(u10 ** 2 + v10 ** 2)


def wind_direction(u10: xr.DataArray, v10: xr.DataArray) -> xr.DataArray:
    """
    Compute meteorological wind direction from u and v components.

    Uses arctan2 to get the angle in degrees.  Note: this gives the
    direction the wind is blowing *toward*, not *from*.

    Input units:  m/s
    Output units: degrees
    """
    mag = wind_magnitude(u10, v10)
    # arctan2(y, x) returns radians; multiply by 180/pi to get degrees
    direction = np.arctan2(v10 / mag, u10 / mag) * 180 / np.pi
    return direction


def vapor_pressure(dewpoint_k: xr.DataArray) -> xr.DataArray:
    """
    Compute vapor pressure from dewpoint temperature.

    Vapor pressure tells you how much water vapor is in the air.  It is
    calculated from the dewpoint temperature using the Magnus formula.

    Reference: https://www.weather.gov/epz/wxcalc_vaporpressure

    Formula: e = 6.11 * 10^(7.5 * Td_C / (237.3 + Td_C))

    Input:  dewpoint in Kelvin
    Output: vapor pressure in hPa (hectopascals, same as millibars)
    """
    td_c = k_to_c(dewpoint_k)  # convert dewpoint from Kelvin to Celsius
    e = 6.11 * 10 ** ((7.5 * td_c) / (237.3 + td_c))
    return e


def relative_humidity(dewpoint_k: xr.DataArray, t2m_k: xr.DataArray) -> xr.DataArray:
    """
    Compute relative humidity from dewpoint and air temperature.

    Relative humidity is the ratio of the actual water vapor in the air
    to the maximum amount the air could hold at the current temperature.
    RH = 1.0 (100%) means the air is fully saturated.

    Formula: RH = vapor_pressure(dewpoint) / vapor_pressure(air_temp)
    The denominator is the "saturation" vapor pressure — the vapor
    pressure you'd get if the air were fully saturated at temperature T.

    Input:  both temperatures in Kelvin
    Output: dimensionless ratio (0 to 1, where 1 = 100% humidity)
    """
    vp = vapor_pressure(dewpoint_k)       # actual vapor pressure
    vp_sat = vapor_pressure(t2m_k)        # saturation vapor pressure
    return vp / vp_sat


# ===========================================================================
# Tier 2 — depend on raw ERA5 + Tier 1
#
# These are the main public metric functions.  Each one takes raw ERA5
# variables as input (in Kelvin / m/s) and internally calls the Tier 1
# helpers above as needed.
# ===========================================================================

def heat_index(t2m_k: xr.DataArray, dewpoint_k: xr.DataArray) -> xr.DataArray:
    """
    NWS Heat Index — what the temperature "feels like" in hot, humid weather.

    This is the official US National Weather Service formula.  It uses a
    multi-step approach:
      1. Start with a simple linear formula.
      2. If the result is > 80°F, switch to the more accurate Rothfusz
         regression (a polynomial in T and RH).
      3. Apply corrections for very low humidity (RH < 13%) or very high
         humidity (RH > 85%) in certain temperature ranges.

    Reference: https://www.wpc.ncep.noaa.gov/html/heatindex_equation.shtml

    Input:  temperatures in Kelvin
    Output: Heat Index in °F  (this metric is conventionally reported in °F)
    """
    # Compute intermediate quantities from Tier 1
    RH = relative_humidity(dewpoint_k, t2m_k)
    T_F = k_to_f(t2m_k)        # air temperature in Fahrenheit
    RH_p = RH * 100             # relative humidity as a percentage (0-100)

    # Step 1: simple formula (adequate for mild conditions)
    hi = 0.5 * (T_F + 61.0 + ((T_F - 68.0) * 1.2) + (RH_p * 0.094))

    # Step 2: Rothfusz regression — a more accurate polynomial fit for
    # higher heat index values.  We compute it everywhere and then use
    # xr.where to only apply it where the simple formula gave HI > 80.
    hi_rothfusz = (
        -42.379
        + 2.04901523 * T_F
        + 10.14333127 * RH_p
        - 0.22475541 * T_F * RH_p
        - 6.83783e-3 * T_F ** 2
        - 5.481717e-2 * RH_p ** 2
        + 1.22874e-3 * T_F ** 2 * RH_p
        + 8.5282e-4 * T_F * RH_p ** 2
        - 1.99e-6 * T_F ** 2 * RH_p ** 2
    )
    # xr.where(condition, value_if_true, value_if_false) — works element-wise
    # on the entire array, like a vectorized if/else
    hi = xr.where(hi > 80, hi_rothfusz, hi)

    # Step 3: correction for very dry air (RH < 13%, T between 80-112°F)
    adj_low_rh = hi_rothfusz - ((13 - RH_p) / 4) * np.sqrt((17 - abs(T_F - 95)) / 17)
    hi = xr.where((RH_p < 13) & (T_F > 80) & (T_F < 112), adj_low_rh, hi)

    # Step 4: correction for very humid air (RH > 85%, T between 80-87°F)
    adj_high_rh = hi_rothfusz + ((RH_p - 85) / 10) * ((87 - T_F) / 5)
    hi = xr.where((RH_p > 85) & (T_F > 80) & (T_F < 87), adj_high_rh, hi)

    return hi


def wind_chill(t2m_k: xr.DataArray, u10: xr.DataArray, v10: xr.DataArray) -> xr.DataArray:
    """
    NWS Wind Chill — what the temperature "feels like" in cold, windy weather.

    The wind chill index is only physically meaningful when it is cold and
    windy.  Outside its valid range (T > 50°F or wind <= 3 mph), this
    function returns NaN (Not a Number) to indicate "not applicable".

    Reference: https://www.weather.gov/safety/cold-wind-chill-chart

    Formula: WC = 35.74 + 0.6215*T - 35.75*V^0.16 + 0.4275*T*V^0.16
    where T is in °F and V is in mph.

    Input:  temperature in Kelvin, wind components in m/s
    Output: Wind Chill in °F (NaN where not defined)
    """
    T_F = k_to_f(t2m_k)
    wind = wind_magnitude(u10, v10)
    wind_mph = wind / 0.44704  # convert m/s to miles per hour

    wc = 35.74 + 0.6215 * T_F - 35.75 * (wind_mph ** 0.16) + 0.4275 * T_F * (wind_mph ** 0.16)

    # NWS definition: Wind Chill is only defined for T <= 50°F and wind > 3 mph.
    # For all other conditions, fill with NaN.
    wc = xr.where((T_F <= 50) & (wind_mph > 3), wc, np.nan)

    return wc


def wet_bulb_temperature(t2m_k: xr.DataArray, dewpoint_k: xr.DataArray) -> xr.DataArray:
    """
    Wet Bulb Temperature — the lowest temperature achievable by evaporative
    cooling (like a wet cloth in front of a fan).

    This uses the Stull (2011) empirical formula, which approximates the
    wet bulb temperature using a series of arctan functions.  It is accurate
    to within about 0.3°C for typical atmospheric conditions.

    Reference: Stull (2011), JAMC
        https://journals.ametsoc.org/view/journals/apme/50/11/jamc-d-11-0143.1.xml

    Input:  temperatures in Kelvin
    Output: Wet Bulb Temperature in Kelvin
    """
    RH = relative_humidity(dewpoint_k, t2m_k)
    RH_p = RH * 100   # percentage
    t_C = k_to_c(t2m_k)

    # Stull's empirical approximation — a sum of arctan terms that was
    # curve-fitted to match the exact wet bulb temperature
    T_w = (
        t_C * np.arctan2(0.151977 * ((RH_p + 8.313659) ** 0.5), 1)
        + np.arctan2((t_C + RH_p), 1)
        - np.arctan2((RH_p - 1.676331), 1)
        + 0.00391838 * (RH_p ** (3 / 2)) * np.arctan2(0.023101 * RH_p, 1)
        - 4.686035
    )

    return T_w + 273.15  # convert back to Kelvin


def apparent_temperature(t2m_k: xr.DataArray, dewpoint_k: xr.DataArray,
                         u10: xr.DataArray, v10: xr.DataArray) -> xr.DataArray:
    """
    Apparent Temperature — a general "feels like" index that accounts for
    humidity and wind.

    Unlike Heat Index (hot weather only) or Wind Chill (cold weather only),
    this metric is defined for all conditions.

    Reference: ECMWF
        https://confluence.ecmwf.int/display/FCST/New+parameters

    Formula: AT = T_C + 0.33 * VP - 0.7 * wind - 4.0
    where VP is vapor pressure (hPa) and wind is in m/s.

    Input:  temperature and dewpoint in Kelvin, wind in m/s
    Output: Apparent Temperature in Kelvin
    """
    t_C = k_to_c(t2m_k)
    vp = vapor_pressure(dewpoint_k)    # Tier 1 helper
    wind = wind_magnitude(u10, v10)    # Tier 1 helper

    at_C = t_C + 0.33 * vp - 0.7 * wind - 4.0
    return at_C + 273.15  # back to Kelvin


def humidex(t2m_k: xr.DataArray, dewpoint_k: xr.DataArray) -> xr.DataArray:
    """
    Canadian Humidex — a heat comfort index used by Environment Canada.

    Similar in spirit to the US Heat Index, but computed with a simpler
    formula.  It combines air temperature and vapor pressure into a single
    number that reflects perceived temperature in humid conditions.

    Formula: H = T_C + 0.5555 * (VP - 10)

    Input:  temperatures in Kelvin
    Output: Humidex in Kelvin
    """
    t_C = k_to_c(t2m_k)
    vp = vapor_pressure(dewpoint_k)

    h_C = t_C + 0.5555 * (vp - 10)
    return h_C + 273.15


def normal_effective_temperature(t2m_k: xr.DataArray, dewpoint_k: xr.DataArray,
                                  u10: xr.DataArray, v10: xr.DataArray) -> xr.DataArray:
    """
    Normal Effective Temperature (NET) — a thermal comfort index that
    accounts for temperature, humidity, and wind.

    This index estimates the temperature of a hypothetical environment
    with no wind and 100% humidity that would produce the same thermal
    sensation as the actual conditions.

    Formula:
        NET = 37 - (37-T_C) / (0.68 - 0.0014*RH% + 1/(1.76 + 1.4*wind^0.75))
              - 0.29 * T_C * (1 - 0.01*RH%)

    Input:  temperature and dewpoint in Kelvin, wind in m/s
    Output: NET in Kelvin
    """
    t_C = k_to_c(t2m_k)
    RH = relative_humidity(dewpoint_k, t2m_k)
    RH_p = RH * 100
    wind = wind_magnitude(u10, v10)

    net_C = (
        37
        - (37 - t_C) / (0.68 - 0.0014 * RH_p + 1 / (1.76 + 1.4 * wind ** 0.75))
        - 0.29 * t_C * (1 - 0.01 * RH_p)
    )
    return net_C + 273.15


# ===========================================================================
# Tier 3 — depend on raw + Tier 1 + Tier 2
# ===========================================================================

def wbgt(t2m_k: xr.DataArray, dewpoint_k: xr.DataArray) -> xr.DataArray:
    """
    Simplified Wet Bulb Globe Temperature (WBGT) for shaded areas.

    WBGT is the gold-standard index for outdoor heat stress assessment,
    used by OSHA and the military.  The full WBGT formula requires globe
    temperature (measured with a special black-globe thermometer), which
    ERA5 does not provide.  This simplified version is appropriate for
    shaded conditions where solar radiation is not a factor.

    This is a Tier 3 metric because it depends on wet_bulb_temperature(),
    which is itself a Tier 2 function.

    Reference: https://iopscience.iop.org/article/10.1088/1748-9326/ab7d04

    Formula: WBGT = 0.7 * T_wetbulb + 0.3 * T_air

    Input:  temperatures in Kelvin
    Output: WBGT in Kelvin
    """
    T_w = wet_bulb_temperature(t2m_k, dewpoint_k)  # calls Tier 2
    return 0.7 * T_w + 0.3 * t2m_k


# ===========================================================================
# Metric registry
#
# This dictionary maps human-readable metric names (the strings you put in
# the METRICS list in the notebook) to all the information needed to compute
# them.  Each entry has:
#   "func"     : the Python function itself
#   "era5_vars": which ERA5 variables the function needs as input
#   "call"     : a lambda that, given a raw ERA5 Dataset, extracts the right
#                variables and calls the function
#   "unit"     : the native output unit (before optional °F conversion)
#
# This pattern is called a "registry" — it lets compute_metrics() look up
# any metric by name without a big if/elif chain.
# ===========================================================================

METRIC_REGISTRY = {
    "heat_index": {
        "func": heat_index,
        "era5_vars": ["2m_temperature", "2m_dewpoint_temperature"],
        "call": lambda ds: heat_index(ds["2m_temperature"], ds["2m_dewpoint_temperature"]),
        "unit": "°F",   # heat_index already returns °F
    },
    "wet_bulb_temperature": {
        "func": wet_bulb_temperature,
        "era5_vars": ["2m_temperature", "2m_dewpoint_temperature"],
        "call": lambda ds: wet_bulb_temperature(ds["2m_temperature"], ds["2m_dewpoint_temperature"]),
        "unit": "K",    # returns Kelvin; can be auto-converted to °F
    },
    "apparent_temperature": {
        "func": apparent_temperature,
        "era5_vars": ["2m_temperature", "2m_dewpoint_temperature",
                      "10m_u_component_of_wind", "10m_v_component_of_wind"],
        "call": lambda ds: apparent_temperature(
            ds["2m_temperature"], ds["2m_dewpoint_temperature"],
            ds["10m_u_component_of_wind"], ds["10m_v_component_of_wind"]),
        "unit": "K",
    },
    "humidex": {
        "func": humidex,
        "era5_vars": ["2m_temperature", "2m_dewpoint_temperature"],
        "call": lambda ds: humidex(ds["2m_temperature"], ds["2m_dewpoint_temperature"]),
        "unit": "K",
    },
    "normal_effective_temperature": {
        "func": normal_effective_temperature,
        "era5_vars": ["2m_temperature", "2m_dewpoint_temperature",
                      "10m_u_component_of_wind", "10m_v_component_of_wind"],
        "call": lambda ds: normal_effective_temperature(
            ds["2m_temperature"], ds["2m_dewpoint_temperature"],
            ds["10m_u_component_of_wind"], ds["10m_v_component_of_wind"]),
        "unit": "K",
    },
    "wind_chill": {
        "func": wind_chill,
        "era5_vars": ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind"],
        "call": lambda ds: wind_chill(
            ds["2m_temperature"],
            ds["10m_u_component_of_wind"], ds["10m_v_component_of_wind"]),
        "unit": "°F",   # wind_chill already returns °F
    },
    "wbgt": {
        "func": wbgt,
        "era5_vars": ["2m_temperature", "2m_dewpoint_temperature"],
        "call": lambda ds: wbgt(ds["2m_temperature"], ds["2m_dewpoint_temperature"]),
        "unit": "K",
    },
}


def compute_metrics(
    ds: xr.Dataset,
    metric_names: list[str] | None = None,
    convert_to_fahrenheit: bool = True,
) -> dict[str, xr.DataArray]:
    """
    Compute requested derived metrics from a raw ERA5 Dataset.

    This is the main entry point for Step 2 of the pipeline.  It looks up
    each requested metric in METRIC_REGISTRY, calls its function, and
    optionally converts the result to Fahrenheit.

    Parameters
    ----------
    ds : xr.Dataset
        Raw ERA5 data (must contain all variables needed by selected metrics).
    metric_names : list[str], optional
        Which metrics to compute (e.g. ["heat_index", "wbgt"]).
        Defaults to all metrics in the registry.
    convert_to_fahrenheit : bool
        If True (default), metrics whose native unit is Kelvin are
        automatically converted to °F.  Metrics that already return °F
        (heat_index, wind_chill) are left as-is.

    Returns
    -------
    dict[str, xr.DataArray]
        A dictionary mapping metric name -> computed hourly DataArray.
        Each DataArray has dimensions (time, lat, lon).
    """
    if metric_names is None:
        metric_names = list(METRIC_REGISTRY.keys())

    results = {}
    for name in metric_names:
        if name not in METRIC_REGISTRY:
            raise ValueError(
                f"Unknown metric '{name}'. Available: {list(METRIC_REGISTRY.keys())}"
            )
        entry = METRIC_REGISTRY[name]
        print(f"[metrics] Computing {name} ...")

        # entry["call"] is a lambda that pulls the right variables from ds
        # and passes them to the metric function
        result = entry["call"](ds)

        # Optionally convert Kelvin -> Fahrenheit (skip metrics already in °F)
        if convert_to_fahrenheit and entry["unit"] == "K":
            result = k_to_f(result)

        results[name] = result

    return results
