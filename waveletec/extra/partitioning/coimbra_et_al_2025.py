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


def conditional_sampling(Y12, *args, names=['xy', 'a'], label={1: "+", -1: "-"}, false=0):
    """
    Perform conditional sampling on xarray.DataArray objects.

    Parameters:
    - Y12: xarray.DataArray (main variable)
    - *args: xarray.DataArray objects for conditional sampling
    - names: list of names for each variable
    - label: dictionary mapping condition values to labels, e.g. {1: "+", -1: "-", 0: "·"}
    - false: value to use for false conditions

    Returns:
    - xarray.Dataset with conditionally sampled variables
    """
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
    if isinstance(data, str): data = read_file(data)
    #lightresponse = lambda p: np.where(np.isnan(p), 1, (p-np.nanmin(p))/(np.nanmax(p)-np.nanmin(p)))
    #data["DW_GPP_withPARratio"] = np.where((np.isnan(data.PPFD)==False) * (data.PPFD<10), 0, 1) * (
    #    data["dwt_wco2-+h2o_uStar_f"] + lightresponse(data["PPFD"]) * (data["dwt_wco2--h2o_uStar_f"]))
    #data["DW_Reco_withPARratio"] = (data["DWT_NEE_uStar_f"] - data["DW_GPP_withPARratio"])
    
    if NIGHT is not None:
        islight = np.where((np.isnan(data[NIGHT]) == False) * (data[NIGHT]), 0, 1)
    else:
        islight = xr.ones_like(data[positive])
    data[labelpositive] = islight * (data[positive] + 0.5*data[negative])
    data[labelnegative] = (data[all] - data[labelpositive])
    return data

def partition_DWCS_H2O(data=None, NEE='NEE', GPP='GPP', Reco='Reco', CO2='wco2', 
                  CO2neg_H2Opos='wco2-wh2o+', CO2neg_H2Oneg='wco2-wh2o-', NIGHT=None):
    if isinstance(data, str): data = read_file(data)
    else: data = data.copy()
    
    logger.debug('CO2 = __input_to_series__(data, CO2)')
    CO2 = data[CO2] if isinstance(CO2, str) else CO2
    CO2neg_H2Opos = data[CO2neg_H2Opos] if isinstance(
        CO2neg_H2Opos, str) else CO2neg_H2Opos
    CO2neg_H2Oneg = data[CO2neg_H2Oneg] if isinstance(
        CO2neg_H2Oneg, str) else CO2neg_H2Oneg
    if data is None: data = pd.DataFrame()

    if NIGHT is not None:
        islight = xr.where((np.isnan(data[NIGHT]) == False) * (data[NIGHT]), 0, 1)
    else:
        islight = xr.ones_like(CO2)

    logger.debug('data[GPP] = islight * (CO2neg_H2Opos + 0.5*CO2neg_H2Oneg)')
    data[GPP] = islight * (CO2neg_H2Opos + 0.5*CO2neg_H2Oneg)
    logger.debug('data[Reco] = (CO2 - data[GPP])')
    data[Reco] = (CO2 - data[GPP])

    data[NEE] = CO2
    #data_pt = data_pt[[NEE, GPP, Reco]]
    return data

def partition_DWCS_CH4(data=None, NEE='NEE', GPP='GPP', Reco='Reco', CO2='wco2', 
                  CO2pos_CH4pos='wco2+wch4+', CO2pos_CH4neg='wco2+wch4-', 
                  CO2neg_CH4pos='wco2-wch4+', CO2neg_CH4neg='wco2-wch4-', NIGHT=None):
    if isinstance(data, str): data = read_file(data)

    CO2 = data[CO2] if isinstance(CO2, str) else CO2
    CO2pos_CH4pos = data[CO2pos_CH4pos] if isinstance(
        CO2pos_CH4pos, str) else CO2pos_CH4pos
    CO2pos_CH4neg = data[CO2pos_CH4neg] if isinstance(
        CO2pos_CH4neg, str) else CO2pos_CH4neg
    CO2neg_CH4pos = data[CO2neg_CH4pos] if isinstance(
        CO2neg_CH4pos, str) else CO2neg_CH4pos
    CO2neg_CH4neg = data[CO2neg_CH4neg] if isinstance(
        CO2neg_CH4neg, str) else CO2neg_CH4neg
    if data is None: data = pd.DataFrame()

    if NIGHT is not None:
        islight = np.where((np.isnan(data[NIGHT]) == False) * (data[NIGHT]), 0, 1)
    else:
        islight = np.array([1] * len(data))
    
    data[Reco] = CO2pos_CH4pos + 0.5 * CO2pos_CH4neg
    data[GPP] = (CO2 - data[Reco])

    data[NEE] = CO2
    #data_pt = data_pt[[NEE, GPP, Reco]]
    return data

def partition_DWCS_CO(data=None, NEE='NEE', GPP='GPP', Reco='Reco', ffCO2='ffCO2',
                     CO2='wco2', 
                     CO2neg_H2Opos='wco2-wh2o+', 
                     CO2neg_H2Oneg='wco2-wh2o-', 
                     CO2pos_COpos='wco2+wco+',
                     CO2pos_COneg='wco2+wco-',
                     NIGHT=None):
    if isinstance(data, str): data = read_file(data)
    #prefix = 'DWnf_' #NEE.split('_', 1)[0] +'_'
    # suffix = ''
    CO2 = data[CO2] if isinstance(CO2, str) else CO2
    CO2neg_H2Opos = data[CO2neg_H2Opos] if isinstance(
        CO2neg_H2Opos, str) else CO2neg_H2Opos
    CO2neg_H2Oneg = data[CO2neg_H2Oneg] if isinstance(
        CO2neg_H2Oneg, str) else CO2neg_H2Oneg
    CO2pos_COpos = data[CO2pos_COpos] if isinstance(
        CO2pos_COpos, str) else CO2pos_COpos
    CO2pos_COneg = data[CO2pos_COneg] if isinstance(
        CO2pos_COneg, str) else CO2pos_COneg
    if data is None: data = pd.DataFrame()

    if NIGHT: NIGHT = data[NIGHT]
    else: NIGHT = 0

    data[NEE] = CO2
    islight = np.where((np.isnan(NIGHT == False) * NIGHT), 0, 1)
    
    data[GPP] = islight * (CO2neg_H2Opos + CO2neg_H2Oneg / 3)

    data[ffCO2] = CO2pos_COpos
    data[Reco]  = CO2pos_COneg

    remaining   = CO2 - data[GPP] - data[Reco] - data[ffCO2]
    data[Reco]  = data[Reco]  + remaining / 2
    data[ffCO2] = data[ffCO2] + remaining / 2
    
    #data = data[[NEE, GPP, Reco, ffCO2]]
    return data