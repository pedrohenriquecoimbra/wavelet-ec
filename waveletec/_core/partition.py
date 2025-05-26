# built-in modules
import logging

# 3rd party modules
import numpy as np
import pandas as pd
import itertools

# project modules
from .commons import __input_to_series__


def conditional_sampling(Y12, *args, names=['xy', 'a'], label={1: "+", -1: "-"}, false=0):
    logger = logging.getLogger('wvlt.partition.conditional_sampling')
    logger.debug(f"star conditional_sampling")

    # label can also be {1: "+", -1: "-", 0: "·"}
    # guarantee names are enough to name all arguments
    nargs = len(args)
    if nargs < len(names):
        names = names[:nargs]
    if nargs > len(names):
        names = names + ['b'] * (nargs-len(names))
    # [Y12] + list(args)
    YS = list(args)
    Ys = {}

    logger.debug(f"YS: {YS}")

    # run for all unique combinations of + and - for groups of size n
    # (e.g., n=2: ++, +-, -+, --, n=3 : +++, ++-, ...)
    for co in set(itertools.combinations(list(label.keys())*nargs, nargs)):
        sign = ''.join([label[c] for c in co])
        name = ''.join([c for cs in zip(names, sign) for c in cs])
        Ys[name] = Y12
        logger.debug(f"name: {name}")
        # condition by sign
        for i, c in enumerate(co):
            if c:
                mask = 1 * (c*YS[i] > 0)
            else:
                mask = 1 * (YS[i] == 0)
            # xy[xy==0] = false
            mask = np.where(mask == 0, false, mask)
            Ys[name] = Ys[name] * mask
    return Ys
    # Ys['info_vars'] = list(Ys.keys())
    # Ys['info_names'] = names
    # return type('var_', (object,), Ys)


def partition_DWCS(data, labelpositive='GPP', labelnegative='Reco', all='wco2', 
                  positive='wco2-wh2o+', negative='wco2-wh2o-', NIGHT=None):
    if isinstance(data, str): data = pd.read_file(data)
    #lightresponse = lambda p: np.where(np.isnan(p), 1, (p-np.nanmin(p))/(np.nanmax(p)-np.nanmin(p)))
    #data["DW_GPP_withPARratio"] = np.where((np.isnan(data.PPFD)==False) * (data.PPFD<10), 0, 1) * (
    #    data["dwt_wco2-+h2o_uStar_f"] + lightresponse(data["PPFD"]) * (data["dwt_wco2--h2o_uStar_f"]))
    #data["DW_Reco_withPARratio"] = (data["DWT_NEE_uStar_f"] - data["DW_GPP_withPARratio"])
    
    if NIGHT is not None:
        islight = np.where((np.isnan(data[NIGHT]) == False) * (data[NIGHT]), 0, 1)
    else:
        islight = np.array([1] * len(data))
    data[labelpositive] = islight * (data[positive] + 0.5*data[negative])
    data[labelnegative] = (data[all] - data[labelpositive])
    return data

def partition_DWCS_H2O(data=None, NEE='NEE', GPP='GPP', Reco='Reco', CO2='wco2', 
                  CO2neg_H2Opos='wco2-wh2o+', CO2neg_H2Oneg='wco2-wh2o-', NIGHT=None):
    if isinstance(data, str): data = pd.read_file(data)
    else: data = data.copy()
    
    CO2 = __input_to_series__(data, CO2)
    CO2neg_H2Opos = __input_to_series__(data, CO2neg_H2Opos)
    CO2neg_H2Oneg = __input_to_series__(data, CO2neg_H2Oneg)
    if data is None: data = pd.DataFrame()

    if NIGHT is not None:
        islight = np.where((np.isnan(data[NIGHT]) == False) * (data[NIGHT]), 0, 1)
    else:
        islight = np.array([1] * len(data))
    
    data[GPP] = islight * (CO2neg_H2Opos + 0.5*CO2neg_H2Oneg)
    data[Reco] = (CO2 - data[GPP])

    data[NEE] = CO2
    #data_pt = data_pt[[NEE, GPP, Reco]]
    return data

def partition_DWCS_CH4(data=None, NEE='NEE', GPP='GPP', Reco='Reco', CO2='wco2', 
                  CO2pos_CH4pos='wco2+wch4+', CO2pos_CH4neg='wco2+wch4-', 
                  CO2neg_CH4pos='wco2-wch4+', CO2neg_CH4neg='wco2-wch4-', NIGHT=None):
    if isinstance(data, str): data = pd.read_file(data)
    
    CO2 = __input_to_series__(data, CO2)
    CO2pos_CH4pos = __input_to_series__(data, CO2pos_CH4pos)
    CO2pos_CH4neg = __input_to_series__(data, CO2pos_CH4neg)
    CO2neg_CH4pos = __input_to_series__(data, CO2neg_CH4pos)
    CO2neg_CH4neg = __input_to_series__(data, CO2neg_CH4neg)
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
    if isinstance(data, str): data = pd.read_file(data)
    #prefix = 'DWnf_' #NEE.split('_', 1)[0] +'_'
    #suffix = ''
    CO2 = __input_to_series__(data, CO2)
    CO2neg_H2Opos = __input_to_series__(data, CO2neg_H2Opos)
    CO2neg_H2Oneg = __input_to_series__(data, CO2neg_H2Oneg)
    CO2pos_COpos =  __input_to_series__(data, CO2pos_COpos)
    CO2pos_COneg =  __input_to_series__(data, CO2pos_COneg)
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