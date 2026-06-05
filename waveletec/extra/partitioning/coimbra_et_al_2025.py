# built-in modules
import logging

# 3rd party modules
import numpy as np
import pandas as pd
import xarray as xr
import itertools

# project modules
from waveletec.core.addons import read_file

logger = logging.getLogger(__name__)

__all__ = [
    "conditional_sampling",
    "partition_DWCS",
    "partition_DWCS_H2O",
    "partition_DWCS_CH4",
    "partition_DWCS_CO",
]


def _as_series(data, x):
    """Resolve a partition input into a numeric array/series.

    Args:
        data: Mapping (DataFrame/Dataset) used to look up column names.
        x: ``None`` (treated as additive identity ``0``), a column name,
            a list/tuple of column names (summed), or an already-resolved array.

    Returns:
        The resolved value: ``0`` for ``None``, ``data[x]`` for a string, the
        element-wise sum for a list/tuple, otherwise ``x`` unchanged.
    """
    if x is None:
        return 0
    if isinstance(x, str):
        return data[x]
    if isinstance(x, (list, tuple)):
        return sum(data[c] for c in x)
    return x


def conditional_sampling(Y12, *args, names=None, label=None, false=0):
    """
    Perform conditional sampling on xarray.DataArray objects.

    Parameters:
    - Y12: xarray.DataArray (main variable)
    - *args: xarray.DataArray objects for conditional sampling
    - names: list of names for each variable (defaults to ['xy', 'a'])
    - label: dictionary mapping condition values to labels, e.g. {1: "+", -1: "-", 0: "·"}
      (defaults to {1: "+", -1: "-"})
    - false: value to use for false conditions

    Returns:
    - xarray.Dataset with conditionally sampled variables
    """
    if names is None:
        names = ['xy', 'a']
    if label is None:
        label = {1: "+", -1: "-"}
    nargs = len(args)
    if nargs < len(names):
        names = names[:nargs]
    if nargs > len(names):
        names = names + ['b'] * (nargs-len(names))
        
    YS = list(args)
    Ys = xr.Dataset()  # Initialize as an empty xarray.Dataset
    
    # run for all unique combinations of + and - for groups of size n
    # (e.g., n=2: ++, +-, -+, --, n=3 : +++, ++-, ...)
    for co in set(itertools.combinations(list(label.keys())*nargs, nargs)):
        if not co:
            continue
        sign = ''.join([label[c] for c in co])
        name = ''.join([c for cs in zip(names, sign) for c in cs])
        Ys[name] = Y12
        logger.debug(f"name: {name}, co: {co}, sign: {sign}")

        # Apply conditions by sign
        for i, c in enumerate(co):
            if c:
                mask = xr.where(c * YS[i] > 0, 1, false)
            else:
                mask = xr.where(YS[i] == 0, 1, false)
            
            mask = np.where(mask == 0, false, mask)
            Ys[name] = Ys[name] * mask
    
    return Ys


def partition_DWCS(data, labelpositive='GPP', labelnegative='Reco', all='wco2',
                  positive='wco2-wh2o+', negative='wco2-wh2o-', NIGHT=None):
    if isinstance(data, str):
        data = read_file(data)
    else:
        data = data.copy()

    all_ = _as_series(data, all)
    positive_ = _as_series(data, positive)
    negative_ = _as_series(data, negative)

    if NIGHT is not None:
        night = data[NIGHT]
        islight = xr.where((~night.isnull()) & night.astype(bool), 0, 1)
    else:
        islight = xr.ones_like(positive_)
    data[labelpositive] = islight * (positive_ + 0.5 * negative_)
    data[labelnegative] = (all_ - data[labelpositive])
    return data

def partition_DWCS_H2O(data=None, NEE='NEE', GPP='GPP', Reco='Reco', CO2='wco2',
                  CO2neg_H2Opos='wco2-wh2o+', CO2neg_H2Oneg='wco2-wh2o-', NIGHT=None):
    if isinstance(data, str):
        data = read_file(data)
    elif data is None:
        data = pd.DataFrame()
    else:
        data = data.copy()

    CO2 = _as_series(data, CO2)
    CO2neg_H2Opos = _as_series(data, CO2neg_H2Opos)
    CO2neg_H2Oneg = _as_series(data, CO2neg_H2Oneg)

    if NIGHT is not None:
        night = data[NIGHT]
        islight = xr.where((~night.isnull()) & night.astype(bool), 0, 1)
    else:
        islight = xr.ones_like(CO2)

    data[GPP] = islight * (CO2neg_H2Opos + 0.5 * CO2neg_H2Oneg)
    data[Reco] = (CO2 - data[GPP])
    data[NEE] = CO2
    return data

def partition_DWCS_CH4(data=None, NEE='NEE', GPP='GPP', Reco='Reco', CO2='wco2', 
                  CO2pos_CH4pos='wco2+wch4+', CO2pos_CH4neg='wco2+wch4-', 
                  CO2neg_CH4pos='wco2-wch4+', CO2neg_CH4neg='wco2-wch4-', NIGHT=None):
    if isinstance(data, str):
        data = read_file(data)
    elif data is None:
        data = pd.DataFrame()
    else:
        data = data.copy()

    CO2 = _as_series(data, CO2)
    CO2pos_CH4pos = _as_series(data, CO2pos_CH4pos)
    CO2pos_CH4neg = _as_series(data, CO2pos_CH4neg)
    CO2neg_CH4pos = _as_series(data, CO2neg_CH4pos)
    CO2neg_CH4neg = _as_series(data, CO2neg_CH4neg)

    if NIGHT is not None:
        night = data[NIGHT]
        islight = xr.where((~night.isnull()) & night.astype(bool), 0, 1)
    else:
        islight = xr.ones_like(CO2)

    data[Reco] = CO2pos_CH4pos + 0.5 * CO2pos_CH4neg
    # GPP (photosynthetic uptake) only occurs in daylight; islight gates it.
    data[GPP] = islight * (CO2 - data[Reco])
    data[NEE] = CO2
    return data

def partition_DWCS_CO(data=None, NEE='NEE', GPP='GPP', Reco='Reco', ffCO2='ffCO2',
                     CO2='wco2', 
                     CO2neg_H2Opos='wco2-wh2o+', 
                     CO2neg_H2Oneg='wco2-wh2o-', 
                     CO2pos_COpos='wco2+wco+',
                     CO2pos_COneg='wco2+wco-',
                     NIGHT=None):
    if isinstance(data, str):
        data = read_file(data)
    elif data is None:
        data = pd.DataFrame()
    else:
        data = data.copy()

    CO2 = _as_series(data, CO2)
    CO2neg_H2Opos = _as_series(data, CO2neg_H2Opos)
    CO2neg_H2Oneg = _as_series(data, CO2neg_H2Oneg)
    CO2pos_COpos = _as_series(data, CO2pos_COpos)
    CO2pos_COneg = _as_series(data, CO2pos_COneg)

    if NIGHT is not None:
        night = data[NIGHT]
        islight = xr.where((~night.isnull()) & night.astype(bool), 0, 1)
    else:
        islight = xr.ones_like(CO2)

    data[NEE] = CO2
    data[GPP] = islight * (CO2neg_H2Opos + CO2neg_H2Oneg / 3)

    data[ffCO2] = CO2pos_COpos
    data[Reco]  = CO2pos_COneg

    remaining   = CO2 - data[GPP] - data[Reco] - data[ffCO2]
    data[Reco]  = data[Reco]  + remaining / 2
    data[ffCO2] = data[ffCO2] + remaining / 2

    return data