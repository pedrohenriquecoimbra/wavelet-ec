"""
This script is a key part of the following publications:
    - Herig Coimbra, Pedro Henrique and Loubet, Benjamin and Laurent, Olivier and Mauder, Matthias and Heinesch, Bernard and 
    Bitton, Jonathan and Delpierre, Nicolas and Depuydt, Jérémie and Buysse, Pauline, Improvement of Co2 Flux Quality Through 
    Wavelet-Based Eddy Covariance: A New Method for Partitioning Respiration and Photosynthesis. 
    Available at SSRN: https://ssrn.com/abstract=4642939 or http://dx.doi.org/10.2139/ssrn.4642939
"""

##########################################
###     IMPORTS                           
##########################################

# standard modules
import copy
import os
import re
import warnings
import logging
import time
import datetime
import sys
from contextlib import contextmanager
from functools import reduce

# 3rd party modules
import yaml
import numpy as np
from itertools import chain
from sklearn.linear_model import LinearRegression, RANSACRegressor, HuberRegressor, QuantileRegressor
import zipfile
from io import StringIO

# Add-ons
def _warning(
    message,
    category = UserWarning,
    filename = '',
    lineno = -1,
    file = None, 
    line = None):
    print("%s: %s" % (category.__name__, message))
warnings.showwarning = _warning

import matplotlib.pyplot as plt
# Reads styles in /styles
stylesheets = plt.style.core.read_style_directory(os.path.join(os.getcwd(), 'style'))
# Update dictionary of styles
plt.style.core.update_nested_dict(plt.style.library, stylesheets)
plt.style.core.available[:] = sorted(plt.style.library.keys())

import pandas as pd
from pandas.api.types import is_numeric_dtype, is_object_dtype
pd.DataFrame.columnstartswith = lambda self, x: [c for c in self.columns if c.startswith(x)]
pd.DataFrame.columnsmatch = lambda self, x: [c for c in self.columns if re.findall(x, c)]
def columnsconditioned(self, start, *args):
    columns = self.columnsmatch(f'^{start}[^_]+$')
    if args:
        for a in args:
            for c in [c for c in columns]:
                if not re.findall(a, c):
                    columns.pop(columns.index(c))
    
    return columns
pd.DataFrame.columnsconditioned = columnsconditioned
def df_to_file(self, file_name, *a, **k): 
    to_functions = {'csv': pd.DataFrame.to_csv,
                    'xlsx': pd.DataFrame.to_excel,
                    'txt': pd.DataFrame.to_csv,
                    'parquet': pd.DataFrame.to_parquet,
                    'temporary': pd.DataFrame.to_parquet,
                    'json': pd.DataFrame.to_json}
    for file_ext, to in to_functions.items():
        if (isinstance(file_name, str) and file_name.replace('.part', '').endswith(file_ext)) | (not isinstance(file_name, str) and file_name.name.replace('.part', '').endswith(file_ext)):
            to(self, file_name, *a, **k)   
    return None
pd.DataFrame.to_file = df_to_file
def pd_read_file(file_name, *a, **k):
    read_functions = {'csv': pd.read_csv,
                    'xlsx': pd.read_excel,
                    'txt': pd.read_csv,
                    'parquet': pd.read_parquet,
                    'temporary': pd.read_parquet,
                    'json': pd.read_json}
    for file_ext, read in read_functions.items():
        if file_name.endswith(file_ext):
            return read(file_name, *a, **k)         
    return None
pd.read_file = pd_read_file

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

##########################################
###     PROJECT CHOICES                           
##########################################

SITES_TO_STUDY = ['SAC']

##########################################
###     FUNCTIONS                           
##########################################

class structuredData:
    def __init__(self, **kwargs):
        for k, v in kwargs.items(): self.__dict__[k]=v
        pass

def read_eddypro_metadata_file(filename):
    metadata = {}
    with open(filename, 'r') as file:
        section = None
        for line in file:
            line = line.strip()
            if line.startswith(';') or not line:
                continue  # Skip comments and empty lines
            if line.startswith('[') and line.endswith(']'):
                section = line[1:-1]  # Extract section name
                metadata[section] = {}
            else:
                key, value = line.split('=', 1)
                metadata[section][key.strip()] = value.strip()
    return metadata

def calculate_mean_wind_direction(wind_speeds, wind_directions):
    """
    Calculates the mean wind direction given wind speeds and wind directions.

    Args:
        wind_speeds (np.ndarray): Array of wind speeds.
        wind_directions (np.ndarray): Array of wind directions in degrees (0° to 360°).

    Returns:
        float: Mean wind direction in degrees (0° to 360°).
    """
    # Convert wind directions to radians
    wind_dir_rad = np.radians(wind_directions)

    # Compute eastward and northward components
    V_east = np.nanmean(wind_speeds * np.sin(wind_dir_rad))
    V_north = np.nanmean(wind_speeds * np.cos(wind_dir_rad))

    # Calculate the mean wind direction
    mean_WD = np.arctan2(V_east, V_north) * (180 / np.pi)

    # Ensure the result is in the range [0°, 360°]
    mean_WD = (360 + mean_WD) % 360

    return mean_WD

month2season = lambda month: {1:1, 2:1, 3:2, 4:2, 5:2, 6:3, 7:3, 8:3, 9:4, 10:4, 11:4, 12:1}[month]
    
def custom_round(x, base=5):
    return base * np.round(x/base)

def __input_to_series__(data, request):
    if data is None: return request

    columns = data.columns
    if request is None: return 0 
    elif isinstance(request, str) and request in columns: 
        request = data[request]
    elif isinstance(request, (list, tuple)) and all(isinstance(i, str) and (i in columns) for i in request): 
        request = np.sum(data[request], axis=1)
    return request

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

def summarisestats(X, y, method='Linear', fit_intercept=True, **kw):
    statisticsToReturn = structuredData()
    X = np.array(X)
    y = np.array(y)
    NaN = np.isnan(X) + np.isnan(y)
    X = X[NaN==0]
    y = y[NaN==0]

    regression = {'Linear': LinearRegression,
                  'RANSAC': RANSACRegressor,
                  'Huber':  HuberRegressor}[method](fit_intercept=fit_intercept, **kw)
    regression.fit(X.reshape(-1, 1), y.reshape(-1, 1))
    statisticsToReturn.m = regression.coef_[0]#[0]
    b = regression.intercept_
    statisticsToReturn.b = b
    b_ = np.round(b, 2) if b else 0
    b_ = "+" + str(b_) if b_ >= 0 else str(b_)
    if method == 'Huber':
        statisticsToReturn.outliers = regression.outliers_
        print(len(regression.outliers_), sum(regression.outliers_))
        X = X[regression.outliers_==False]
        y = y[regression.outliers_==False]
        
    statisticsToReturn.me = np.nanmean((X-y)).round(2)
    statisticsToReturn.mae = np.nanmean(abs(X-y)).round(2)
    statisticsToReturn.r2 = regression.score(X.reshape(-1, 1), y.reshape(-1, 1))
    return statisticsToReturn

def summarisestatslabel(meta, xn, yn):    
    stat_label = f"R²: {np.round(meta.r2, 2)}"
    stat_label = stat_label + f"\nME: {np.round(meta.me, 2)}"
    stat_label = stat_label + f"\nMAE: {np.round(meta.mae, 2)}"
    stat_label = stat_label + f"\n{yn}={np.round(meta.m, 2)} {xn}"
    return stat_label
    
def summarisestatstext(meta, xn='x', yn='y'):    
    stat_label = f"R²= {np.round(meta.r2, 2)}"
    stat_label = stat_label + f", ME= {np.round(meta.me, 2)} µmol m-2 s-1"
    stat_label = stat_label + f", MAE= {np.round(meta.mae, 2)} µmol m-2 s-1"
    stat_label = stat_label + f", {yn}={np.round(meta.m, 2)}"+r"$\times$"+"{xn} linear fit" #×
    return stat_label

def get_r2(X, y):
    if len(X)==0:
        return 0
    X = np.array(X).ravel()
    y = np.array(y).ravel()
    finite = np.isfinite(X*y)
    X = X[finite].reshape(-1, 1)
    y = y[finite].reshape(-1, 1)
    regression = LinearRegression(fit_intercept=True)
    regression.fit(X, y)
    r2 = regression.score(X, y)
    return r2

def abline(intercept=0, slope=1, origin=(0, 0), **kwargs):
    kwargs.pop('data', None)
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, **kwargs)

def abline2(origin=(0, 0), slope=1, length=1, scale='', **kwargs):
    kwargs.pop('data', None)
    if scale=='log': origin = np.log10(origin)
    x_vals = np.array([origin[0], origin[0]+length])#np.linspace(origin[0], origin[0]+length)
    y_vals = origin[1] - slope * x_vals[0] + slope * x_vals
    if scale=='log': 
        x_vals = 10**x_vals
        y_vals = 10**y_vals
    plt.plot(x_vals, y_vals, **kwargs)

def add_subplot_axes(ax,rect):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)    
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    #subax = fig.add_axes([x,y,width,height],facecolor=facecolor)  # matplotlib 2.0+
    subax = fig.add_axes([x,y,width,height])
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax

def sum_nan_arrays(a, b):
    ma = np.isnan(a)
    mb = np.isnan(b)
    return np.where(ma & mb, np.nan, np.where(ma, 0, a) + np.where(mb, 0, b))

def force_array_dimension(shape, out):
    """
    Return array with same shape as base one.
    """
    out_ = np.zeros(shape) * np.nan

    shape_dif = (np.array(shape) - np.array(out.shape)) / 2
    signal = shape_dif / abs(shape_dif)

    bas_cut = [None] * 4
    out_cut = [None] * 4

    for i, s_ in enumerate(signal):

        dif = [int(np.ceil(abs(shape_dif[i]))), -
               int(np.floor(abs(shape_dif[i])))]
        dif = [el if el != 0 else None for el in dif]

        if i == 0:
            if s_ == 1:
                bas_cut[:2] = dif

            elif s_ == -1:
                out_cut[:2] = dif

        elif i == 1:
            if s_ == 1:
                bas_cut[2:] = dif

            elif s_ == -1:
                out_cut[2:] = dif

    out_[bas_cut[0]:bas_cut[1],
         bas_cut[2]:bas_cut[3]] = \
        sum_nan_arrays(out_[bas_cut[0]:bas_cut[1],
                               bas_cut[2]:bas_cut[3]],
                          out[out_cut[0]:out_cut[1],
                              out_cut[2]:out_cut[3]])

    return out_

##########################################
###     GET DATASETS                           
##########################################

def get_all_sites(fc, *args, **kwargs):
    datasetToReturn = structuredData()
    datasetToReturn.data = {}
    for sitename in SITES_TO_STUDY:
        data = fc(*args, sitename=sitename, **kwargs)
        if data is None: data = pd.DataFrame({"TIMESTAMP": []})
        datasetToReturn.data[sitename] = data
    datasetToReturn.alldata = []
    for k, v in datasetToReturn.data.items():
        v.insert(0, 'CO_SITE', k)
        datasetToReturn.alldata += [copy.deepcopy(v).reset_index(drop=True)]
    datasetToReturn.alldata = pd.concat(
        datasetToReturn.alldata).reset_index(drop=True)
    return datasetToReturn.alldata

def get_cospectra(sitename=None, **kwargs):
    if sitename is None:
        return get_all_sites(get_cospectra, **kwargs)
    mergeorconcat = kwargs.get('mergeorconcat', 'merge')
    folder        = kwargs.get('folder', 'data')
    duplicates    = kwargs.get('duplicates', False)

    wv_path = os.path.join(folder, sitename, 'output', 'DWCS')
    data = []
    for name in os.listdir(wv_path):
        if any([name.endswith(ext) for ext in ['csv', 'xlsx', 'txt', 'parquet', 'json']]):
            if re.findall('_full_cospectra', name) and re.findall('.30mn', name):
                data.append(pd.read_file(os.path.join(wv_path, name)))
    
    if len(data) == 0:
        return None
    elif len(data) == 1:
        data = data[0]
    elif mergeorconcat == 'concat':
        data = pd.concat(data)
    else:
        if duplicates:
            data = reduce(lambda left, right: pd.merge(left, right, on=['TIMESTAMP'], how="outer", suffixes=('', '_DUP')),
                            data)
        else:
            print(data)
            data = reduce(lambda left, right: pd.merge(left, right[['TIMESTAMP'] + list(set(right) - set(left))], 
                                                       on=['TIMESTAMP'], how="outer", suffixes=('', '_DUP')), data)
    
    for tc in data.columnstartswith('TIMESTAMP'):
        data[tc] = pd.to_datetime(
            data[tc])
    return data
    
def get_metadata(sitename=None, folder='data'):
    if sitename is None:
        return get_all_sites(get_metadata)
    
    mt_path = os.path.join(folder, sitename, f'{sitename}_metadata.yaml')
    if os.path.exists(mt_path):
        meta = yaml.safe_load(open(mt_path, 'r'))
        return pd.DataFrame(meta, index=[0])
    return None


def get_eddypro_cospectra(sitename=None, x='natural_frequency', y='f_nat*cospec(w_co2)/cov(w_co2)', folder='data', subfolder='', help=False):
    assert (x is not None and y is not None) or (help)
    if sitename is None:
        return get_all_sites(get_eddypro_cospectra, x=x, y=y, folder=folder, subfolder=subfolder, help=help)
    
    ep_path = os.path.join(folder, sitename, 'output/eddypro_output', subfolder, 'eddypro_binned_cospectra')
    
    if not os.path.exists(ep_path): return None
    files = []
    for name in os.listdir(ep_path):
        if name.endswith('.csv'):
            if re.findall('_binned_cospectra_', name):
                binned_cosp = pd.read_csv(os.path.join(ep_path, name), skiprows=11, na_values=[-9999, 'NAN'])
                if help: 
                    print(binned_cosp.columns)
                    return
                if x not in binned_cosp.columns or y not in binned_cosp.columns: continue
                binned_cosp.dropna(subset=[x], inplace=True)
                binned_cosp = binned_cosp[[x, y]]
                binned_cosp['TIMESTAMP'] = name.split('_')[0]
                binned_cosp = binned_cosp.pivot(index='TIMESTAMP', columns=x, values=y).reset_index(drop=False)
                binned_cosp.columns = [c for c in binned_cosp.columns]
                files += [binned_cosp]
    
    if len(files) == 1:
        data = files[0]
    else:
        data = pd.concat(files)
    
    data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP'], format='%Y%m%d-%H%M')
    return data

def get_eddypro_output(sitename=None, **kwargs):
    if sitename is None:
        return get_all_sites(get_eddypro_output, **kwargs)
    mergeorconcat = kwargs.get('mergeorconcat', 'merge')
    folder        = kwargs.get('folder', 'data')
    
    ep_read_params = {'skiprows': [0,2], 'na_values': [-9999, 'NAN']}
    
    ep_path = os.path.join(folder, sitename, 'eddypro_output')

    files = {'FLUX': [], 'FLXNT': [], 'QCQA': [], 'META': []}
    for name in os.listdir(ep_path):
        if name.endswith('.csv'):
            if re.findall('_full_output_', name):
                if name.endswith('_adv.csv'):
                    files['FLUX'] += [pd.read_csv(os.path.join(ep_path, name), **ep_read_params)]
                else:
                    files['FLUX'] += [pd.read_csv(os.path.join(ep_path, name), na_values=[-9999, 'NAN'])]
            elif re.findall('_fluxnet_', name):
                flxnt_last = pd.read_csv(os.path.join(ep_path, name), na_values=[-9999, 'NAN'])
                flxnt_last["date"] = pd.to_datetime(flxnt_last["TIMESTAMP_START"], format='%Y%m%d%H%M').dt.strftime('%Y-%m-%d')
                flxnt_last["time"] = pd.to_datetime(flxnt_last["TIMESTAMP_START"], format='%Y%m%d%H%M').dt.strftime('%H:%M')
                files['FLXNT'] += [flxnt_last]
                del flxnt_last
            elif re.findall('_qc_details_', name):
                if name.endswith('_adv.csv'):
                    files['QCQA'] += [pd.read_csv(os.path.join(ep_path, name), **ep_read_params)]
                else:
                    files['QCQA'] += [pd.read_csv(os.path.join(ep_path, name), na_values=[-9999, 'NAN'])]
            elif re.findall('_metadata_', name):
                files['META'] += [pd.read_csv(os.path.join(ep_path, name), na_values=[-9999, 'NAN'])]
    
    for k in [k for k in list(files.keys()) if not files[k]]:
        del([files[k]])

    for k in files.keys():
        if len(files[k]) == 1:
            files[k] = files[k][0]
        elif mergeorconcat == 'concat':
            files[k] = pd.concat(files[k])
        else:
            files[k] = reduce(lambda left, right: pd.merge(
                left, right, on=["date", 'time'], how="outer", suffixes=('', '_DUP')), files[k])
    
    data = pd.DataFrame(files.pop('FLUX', {}))
    for name, dat in files.items():
        data = pd.merge(data, dat, on=["date", 'time'], how="outer", suffixes=('', f'_{name}'))
    
    data['TIMESTAMP'] = pd.to_datetime(data.date + ' ' + data.time)#.dt.tz_localize('UTC')
    
    # Despike because open path analysers can be noisy
    data['air_molar_volume_despiked'] = mauder2013(data.air_molar_volume, 5)
    data['Vd'] = (data.air_molar_volume_despiked * data.air_pressure /
                           (data.air_pressure - data.e))
    # make wet from dry
    data['Va'] = (data.air_molar_volume_despiked * data.air_pressure /
                           (data.air_pressure - data.e))
    return data

def j2sj(e, samp_rate=10): return 1/(samp_rate*(2**-float(e)))
def sj2j(s, samp_rate=10): return np.log(samp_rate*s) / np.log(2)

def get_dic_flux_data(data): return {k: copy.deepcopy(data.query(
    f"CO_SITE == '{k}'").reset_index(drop=True)) for k in data.CO_SITE.unique()}

##########################################
###     GENERIC FUNCTIONS                           
##########################################

def matrixtotimetable(time, mat, c0name="TIMESTAMP", **kwargs):
    assert len(time) in mat.shape, f"Time ({time.shape}) and matrix ({mat.shape}) do not match."
    mat = np.array(mat)

    if len(time) != mat.shape[0] and len(time) == mat.shape[1]:
        mat = mat.T

    __temp__ = pd.DataFrame(mat, **kwargs)
    __temp__.insert(0, c0name, time)

    return __temp__


def yaml_to_dict(path):
    with open(path, 'r+') as file:
        file = yaml.safe_load(file)
    return file


def list_time_in_period(tmin, tmax, fastfreq, slowfreq, include='both'):
    if include=="left":
        return [(pd.date_range(max(p, pd.to_datetime(tmin)), min(p + pd.Timedelta(slowfreq), pd.to_datetime(tmax)),
          freq=fastfreq)[:-1]) for p in pd.date_range(tmin, tmax, freq=slowfreq).floor(slowfreq)]
    elif include == "right":
        return [(pd.date_range(max(p, pd.to_datetime(tmin)), min(p + pd.Timedelta(slowfreq), pd.to_datetime(tmax)),
          freq=fastfreq)[1:]) for p in pd.date_range(tmin, tmax, freq=slowfreq).floor(slowfreq)]
    elif include == "both":
        return [(pd.date_range(max(p, pd.to_datetime(tmin)), min(p + pd.Timedelta(slowfreq), pd.to_datetime(tmax)),
          freq=fastfreq)) for p in pd.date_range(tmin, tmax, freq=slowfreq).floor(slowfreq)]
    return


def checkifinprogress(path, LIMIT_TIME_OUT=30*60):
    if os.path.exists(path) and (time.time()-os.path.getmtime(path)) < LIMIT_TIME_OUT:
        logging.debug(f'Fresh file found ({time.time()-os.path.getmtime(path)} s old, {os.path.getmtime(path)}), skipping.')
        return 1
    else:
        if os.path.exists(path): logging.debug(f'Old file found ({time.time()-os.path.getmtime(path)} s old, {time.time()*10**-3}, {os.path.getmtime(path)*10**-3}), new in progress file created.')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a+"):
            pass
        return 0


def nanminmax(x):
    return [np.nanmin(x), np.nanmax(x)]


def mkdirs(filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    
def nearest(items, pivot, direction=0):
    if direction == 0:
        nearest = min(items, key=lambda x: abs(x - pivot))
        difference = abs(nearest - pivot)
        
    elif direction == -1:
        nearest = min(items, key=lambda x: abs(x - pivot) if x<pivot else pd.Timedelta(999, "d"))
        difference = (nearest - pivot)
        
    elif direction == 1:
        nearest = min(items, key=lambda x: abs(x - pivot) if x>pivot else pd.Timedelta(999, "d"))
        difference = (nearest - pivot)
    return nearest, difference


def update_nested_dict(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = update_nested_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def update_nested_dicts(*ds, fstr=None):
    r = {}
    for d in ds:
        if isinstance(d, str) and fstr:
            try:
                d = fstr(d)
            except Exception as e:
                continue
        r = update_nested_dict(r, d)
    return r

def concat_into_single_file(path, pattern, output_path=None, **kwargs):
    print('\nCONSOLIDATING DATASET\n')
    if output_path is None: output_path = os.path.join(path, 'concat_into_single_file') 
    
    files_to_concat = []
    for name in os.listdir(path):
        if re.findall(pattern, name):
            files_to_concat += [os.path.join(path, name)]
    
    files_to_concat = [pd.read_csv(f, **kwargs) for f in files_to_concat]
    data = reduce(lambda left, right: pd.concat([left, right]), files_to_concat)
    
    mkdirs(output_path)
    data.to_csv(output_path, index=False)
    print(os.path.basename(output_path), ': Saved.', ' '*15, end='\n', sep='')
    
    return

##########################################
###     UNIVERSAL READER                            
##########################################

DEFAULT_FILE_RAW = {
    'file_pattern': '([0-9]{8}-[0-9]{4})_raw_dataset_.*.txt', 
    'date_format': '%Y%m%d-%H%M', 
    'dt': 0.05, 
    'tname': "TIMESTAMP", 
    'id': None,
    'datefomatfrom': '%Y%m%d%H%M%S.%f', 
    'datefomatto': '%Y-%m-%dT%H:%M:%S.%f'
}

DEFAULT_READ_CSV = {
    'skiprows': 8,
    'sep': r"\s+",
    'na_values': ['NaN', 'nan', -9999],
}

DEFAULT_READ_GHG = {
    'skiprows': 7,
    'sep': r"\t",
    'engine': 'python'
}

DEFAULT_FMT_DATA = {
}


class structuredDataFrame:
    def __init__(self, data=None, dt=None, **kwargs):
        if data is None:
            loopvar = kwargs.pop('lookup', [])
            loopvar = [l.to_list() if isinstance(l, list)==False else l for l in loopvar]
            for l in loopvar:
                result = universal_reader(lookup=l, **kwargs, verbosity=0)
                self.__dict__.update(result.__dict__)
        
        else:
            assert dt is not None, 'Missing measurement frequency (dt).'
            self.data = data
            self.dt = dt
            self.__dict__.update(**kwargs)
    
    def filter(self, items: dict):
        for k, v in items.items():
            if isinstance(v, tuple):
                self.data = self.data.loc[(self.data[k] > v[0])
                                          & (self.data[k] < v[1])].copy()
            else:
                self.data = self.data[self.data[k].isin(v)].copy()
        return self

    def rename(self, names: dict):
        self.data = self.data.rename(columns=names)
        return self

    def modify(self, items: dict):
        for k, v in items.items():
            self.data[k] = v
        return self

    def format(self, 
               cols={'t':'ts'}, 
               keepcols=['u', 'v', 'w', 'ts', 'co2', 'co2_dry', 'h2o', 'h2o_dry', 'ch4', 'n2o'],
               addkeep=[],
               colsfunc=str.lower, cut=False, **kwargs):
        
        if isinstance(self, pd.DataFrame):
            formated = self
        else:
            fmt_clas = structuredDataFrame(**self.__dict__)
            formated = fmt_clas.data

        if colsfunc is not None:
            formated.columns = map(colsfunc, formated.columns)
        #cols.update(kwargs)
        cols.update({v.lower(): k.lower() for k, v in kwargs.items() if isinstance(v, list)==False})
        cols = {v: k for k, v in {v: k for k, v in cols.items()}.items()}
        cols.update({'timestamp': 'TIMESTAMP'})
        #formated.TIMESTAMP = formated.TIMESTAMP.apply(np.datetime64)
        if cut:
            #formated = formated[[
            #    c for c in formated.columns if c in cols.keys()]]
            formated = formated.loc[:, np.isin(formated.columns, keepcols+addkeep+list(cols.keys()))]
        
        formated = formated.rename(columns=cols)

        if isinstance(self, pd.DataFrame):
            return formated
        else:
            fmt_clas.data = formated
            return fmt_clas
    
    def interpolate(self, cols=["co2", "w"], qcname="qc"):
        interpolated = structuredDataFrame(**self.__dict__)
        interpolated.data[qcname] = 0
        for c_ in list(cols):
            interpolated.data[qcname] = interpolated.data[qcname] + 0 * \
                np.array(interpolated.data[c_])
            interpolated.data.loc[np.isnan(interpolated.data[qcname]), qcname] = 1
            interpolated.data[qcname] = interpolated.data[qcname].astype(int)
            interpolated.data[c_] = interpolated.data[c_].interpolate(method='pad')
            
        return interpolated


def __universal_reader__(path, **kw_csv):
    if path.endswith('.gz'): kw_csv.update(**{'compression': 'gzip'})
    elif path.endswith('.csv'): kw_csv.pop('compression', None)
    if path.endswith('.ghg'):
        with zipfile.ZipFile(path, 'r') as zip_ref:
            datafile = [zip_ref.read(name) for name in zip_ref.namelist() if name.endswith(".data")][0]
        datafile = str(datafile, 'utf-8')
        path = StringIO(datafile)
        # DEFAULT_READ_GHG
        kw_csv.update(DEFAULT_READ_GHG)
    try:
        df_td = pd.read_csv(path, **kw_csv)
    except Exception as e:
        # (EOFError, pd.errors.ParserError, pd.errors.EmptyDataError):
        try:
            #if verbosity>1: warnings.warn(f'{e}, when opening {path}, using {kw_csv}. Re-trying using python as engine and ignoring bad lines.')
            df_td = pd.read_csv(path, on_bad_lines='warn', engine='python', **kw_csv)
        except Exception as ee:
            warnings.warn(f'{ee}, when opening {str(path)}, using {kw_csv}')
            return None
    return df_td

def universal_reader(path, lookup=[], fill=False, fmt={}, onlynumeric=True, verbosity=1, fkwargs={}, tipfile="readme.txt", **kwargs):
    if isinstance(path, dict):
        dataframes = {}
        for k, v in path.items():
            assert isinstance(v, str), f'Path ({v}) is not string.'
            path_ = universal_reader(v, lookup, fill, fmt, onlynumeric, verbosity, fkwargs, tipfile, **kwargs)
            dt = path_.dt
            dataframes[k] = path_.data
        #df_site = dataframes.pop(k, pd.DataFrame())
        dup_keys = {}
        for k in dataframes.keys():
            dup_keys[k] = [set(dataframes[k].columns).intersection(set(v_.columns)) - set(['TIMESTAMP']) for k_, v_ in dataframes.items() if k_!=k]
            dup_keys[k] = list(chain.from_iterable(dup_keys[k]))
            dataframes[k].rename(columns={c: c + k for c in dup_keys[k]}, inplace=True)
        
        try:
            df_site = reduce(lambda left, right: pd.merge(left, right, on='TIMESTAMP', how='outer', suffixes=('', '_DUP')), 
                            list(dataframes.values()))
        except Exception as e:
            logging.error(str(e))
            logging.debug('Concatenating instead of merging.')
            #df_site = reduce(lambda left, right: pd.concat([left, right], axis=1), dataframes)
            return structuredDataFrame(pd.DataFrame(), dt=dt)
        #for k, v in dataframes.items():
        #    df_site = df_site.merge(v, on='TIMESTAMP', how='outer', suffixes=('', k))
        return structuredDataFrame(df_site, dt=dt)
    
    df_site = pd.DataFrame()
    
    folders = [path + p + '/' for p in os.listdir(path) if os.path.isdir(path + p)]
    folders = folders if folders else [path]
    
    #print("Check readable files in", path if len(path)<40 else f'{path[:5]}...{path[-30:]}')#, fmt, fkwargs, kwargs)

    for path_ in folders:
        df_td = pd.DataFrame()

        # read tips file
        kw_ = update_nested_dicts({"FILE_RAW": DEFAULT_FILE_RAW, "READ_CSV": DEFAULT_READ_CSV, "FMT_DATA": DEFAULT_FMT_DATA}, 
                                  os.path.join(path, tipfile), os.path.join(path_, tipfile),
                                  {"FILE_RAW": fkwargs, "READ_CSV": kwargs, "FMT_DATA": fmt},
                                  fstr=lambda d: yaml_to_dict(d))
        
        kw = structuredData(**kw_['FILE_RAW'])
        kw_csv = kw_['READ_CSV']
        
        try:
            if ('header_file' in kw_csv.keys()) and (os.path.exists(kw_csv['header_file'])):
                kw_csv['header_file'] = "[" + open(kw_csv['header_file']).readlines()[0].replace("\n", "") + "]"
        except:
            None
        
        lookup_ = list(set([f.strftime(kw.date_format) for f in lookup]))
        files_list = {}

        for root, directories, files in os.walk(path_):
            for name in files:
                dateparts = re.findall(kw.file_pattern, name, flags=re.IGNORECASE)
                if len(dateparts) == 1:
                    files_list[dateparts[0]] = os.path.join(root, name)

        for td in set(lookup_) & files_list.keys() if lookup_ != [] else files_list.keys():
            path_to_tdfile = files_list[td]
            if os.path.exists(path_to_tdfile):
                df_td = __universal_reader__(path_to_tdfile, **kw_csv)
                """
                if path_to_tdfile.endswith('.gz'): kw_csv.update(**{'compression': 'gzip'})
                elif path_to_tdfile.endswith('.csv'): kw_csv.pop('compression', None)
                if path_to_tdfile.endswith('.ghg'):
                    with zipfile.ZipFile(path_to_tdfile, 'r') as zip_ref:
                        datafile = [zip_ref.read(name) for name in zip_ref.namelist() if name.endswith(".data")][0]
                    datafile = str(datafile, 'utf-8')
                    path_to_tdfile = StringIO(datafile)
                    # DEFAULT_READ_GHG
                    kw_csv.update(DEFAULT_READ_GHG)
                try:
                    df_td = pd.read_csv(path_to_tdfile, **kw_csv)
                except Exception as e:
                    # (EOFError, pd.errors.ParserError, pd.errors.EmptyDataError):
                    try:
                        if verbosity>1: warnings.warn(f'{e}, when opening {path_to_tdfile}, using {kw_csv}. Re-trying using python as engine and ignoring bad lines.')
                        df_td = pd.read_csv(path_to_tdfile, on_bad_lines='warn', engine='python', **kw_csv)
                    except Exception as ee:
                        warnings.warn(f'{ee}, when opening {str(path_to_tdfile)}, using {kw_csv}')
                        continue
                """
                """
                if kw.tname in df_td.columns:
                    try:
                        df_td.loc[:, kw.tname] = pd.to_datetime(df_td.loc[:, kw.tname].astype(str))
                        print(max(df_td[kw.tname].dt.year), max(df_td[kw.tname]), min(df_td[kw.tname].dt.year))
                        assert max(df_td[kw.tname].dt.year) > 1990 and min(df_td[kw.tname].dt.year) > 1990
                    except:
                        df_td.rename({kw.tname+'_orig': kw.tname})
                """
                if kw.datefomatfrom == 'drop':
                    df_td = df_td.rename({kw.tname: kw.tname+'_orig'})
                
                if kw.tname not in df_td.columns or kw.datefomatfrom == 'drop':
                    if "date" in df_td.columns and "time" in df_td.columns:
                        df_td[kw.tname] = pd.to_datetime(
                            df_td.date + " " + df_td.time, format='%Y-%m-%d %H:%M')
                    else:
                        df_td[kw.tname] = pd.to_datetime(
                            td, format=kw.date_format) - datetime.timedelta(seconds=kw.dt) * (len(df_td)-1 + -1*df_td.index)
                            #td, format=kw.date_format) + datetime.timedelta(seconds=kw.dt) * (df_td.index)
                        df_td[kw.tname] = df_td[kw.tname].dt.strftime(
                            kw.datefomatto)
                else:
                    try:
                        if is_numeric_dtype(df_td[kw.tname]):
                            df_td.loc[:, kw.tname] = df_td.loc[:, kw.tname].apply(lambda e: pd.to_datetime('%.2f' % e, format=kw.datefomatfrom).strftime(kw.datefomatto))
                        elif is_object_dtype(df_td[kw.tname]):
                            df_td.loc[:, kw.tname] = df_td.loc[:, kw.tname].apply(lambda e: pd.to_datetime(e).strftime(kw.datefomatto))
                        else:
                            df_td.loc[:, kw.tname] = pd.to_datetime(df_td[kw.tname], format=kw.datefomatfrom).strftime(kw.datefomatto)
                    except:
                        warnings.warn(f'error when converting {kw.tname} from {kw.datefomatfrom} to {kw.datefomatto}.')
                        continue
                
                df_td['file'] = td
                #df_site = df_site.append(df_td)
                df_site = pd.concat([df_site, df_td], ignore_index=True).reset_index(drop=True)
        
        if df_td.empty == False:
            break
        
    #print('df_td.empty ', df_td.empty)
    if onlynumeric:
        valcols = [i for i in df_site.columns if i.lower() not in [kw.tname.lower(), 'file']]
        _bf = df_site.dtypes
        #df_site.loc[:, valcols] = df_site.loc[:, valcols].apply(pd.to_numeric, errors='coerce')
        df_site[valcols] = df_site[valcols].apply(pd.to_numeric, errors='coerce')
        _af = df_site.dtypes
        if verbosity>1:
            _bfaf = []
            for (k, b) in _bf.items():
                if b!=_af[k]:
                    _nonnum = [s for s in np.unique(df_site[k].apply(lambda s: str(s) if re.findall('[A-z/]+', str(s)) else '')) if s]
                    _bfaf += ['{}, changed from {} to {}. ({})'.format(k, b, _af[k], ', '.join(_nonnum) if _nonnum else 'All numeric')]
            if _bfaf:
                warnings.warn(', '.join(_bfaf))
    
    #if kw_fmt:
    df_site = structuredDataFrame.format(df_site, **kw_['FMT_DATA'])

    if fill:
        if lookup:
            minmax = [min(lookup), max(lookup)]
        else:
            minmax = [np.nanmin(df_site[kw.tname]),
                      np.nanmax(df_site[kw.tname])]
        df_site = df_site.set_index(kw.tname).join(pd.DataFrame({kw.tname: pd.date_range(*minmax, freq=str(kw.dt) + ' S')}).set_index(kw.tname),
                how='outer').ffill().reset_index()
        #if 'co2' in df_site.columns and (abs(np.max(df_site.co2)) < 1000) and (abs(np.min(df_site.co2)) < 1000):
        #    df_site.loc[:, "co2"] = df_site.loc[:, "co2"] * 1000  # mmol/m3 -> μmol/m3
    
    if kw.id is not None:
        return {kw.id: structuredDataFrame(df_site, dt=kw.dt)}
    else:
        return structuredDataFrame(df_site, dt=kw.dt)


def loaddatawithbuffer(d0, d1=None, freq=None, buffer=None, 
                       tname="TIMESTAMP", f_freq=30, **kwargs):
    if isinstance(d0, (pd.DatetimeIndex, list, set, tuple)) and d1 is None:
        d0, d1 = [np.nanmin(d0), np.nanmax(d0)]
    
    if buffer == None:
        datarange = [pd.date_range(start=d0, end=d1, freq=freq)[:-1] + pd.Timedelta(freq)]
    else:
        # buffer align with file frequency (e.g. 30 min)
        freqno = int(re.match(r"\d*", f"{f_freq}min")[0])
        bufi = np.ceil(buffer / (freqno*60)) * freqno
        datarange = [
            pd.date_range(
                start=pd.to_datetime(d0) - pd.Timedelta(bufi, unit='min'),
                end=pd.to_datetime(d1) + pd.Timedelta(bufi, unit='min'),
                freq=freq)[:-1] + pd.Timedelta(freq)]
                
    if not datarange:
        return pd.DataFrame()
    
    data = structuredDataFrame(lookup=datarange, **kwargs)
    if data == None or data.data.empty:
        return data.data
    data.data[tname] = pd.to_datetime(data.data[tname])
    
    if buffer:
        d0 = pd.to_datetime(d0) - pd.Timedelta(buffer*1.1, unit='s')
        d1 = pd.to_datetime(d1) + pd.Timedelta(buffer*1.1, unit='s')
        data.filter({tname: (d0, d1)})

    # garantee all data points, if any valid time, else empty dataframe
    if np.sum(np.isnat(data.data.TIMESTAMP)==False):
        #data.data = pd.merge(pd.DataFrame({tname: pd.date_range(*nanminmax(data.data.TIMESTAMP), freq="0.05S")}),
        #                    data.data,
        #                    on=tname, how='outer').reset_index(drop=True)
        return data.data
    else:
        pd.DataFrame()

##########################################
###     METEO                            
##########################################

def vapour_deficit_pressure(T, RH):
    if np.nanquantile(T, 0.1) < 100 and np.nanquantile(T, 0.9) < 100:
        T = T + 274.15

    # Saturation Vapor Pressure (es)
    #es = 0.6108 * np.exp(17.27 * T / (T + 237.3))
    es = (T **(-8.2)) * (2.7182)**(77.345 + 0.0057*T-7235*(T**(-1)))

    # Actual Vapor Pressure (ea)
    ea = es * RH / 100

    # Vapor Pressure Deficit (Pa)
    return (es - ea)# * 10**(-3)

##########################################
###     DESPIKING                            
##########################################

def mauder2013(x, q=7):
    # it does not do the check for n consecutive spikes 
    x = np.array(x)
    x_med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - x_med))
    bounds = (x_med - (q * mad) / 0.6745, x_med + (q * mad) / 0.6745)
    #print("median", x_med, "mad", mad, "bounds", bounds)
    x[x < min(bounds)] = np.nan
    x[x > max(bounds)] = np.nan

    #if fill is not None:
    #    x = fill(pd.Series(x) if fill in (pd.Series.ffill, pd.Series.interpolate) else x)
    return x


##########################################
###     WAVELET-RELATED                            
##########################################

try:
    import pywt
    def bufferforfrequency_dwt(N=0, n_=None, fs=20, level=None, f0=None, max_iteration=10**4, wave='db6'):
        if level is None and f0 is None: f0 = 1/(2*60*60)  # 18
        lvl = level if level is not None else int(np.ceil(np.log2(fs/f0)))
        if n_ is None: n_ = fs * 60 * 30
        n0 = N
        cur_iteration = 0
        while True:
            n0 += pd.to_timedelta(n_)/pd.to_timedelta("1S") * fs if isinstance(n_, str) else n_
            if lvl <= pywt.dwt_max_level(n0, wave):
                break
            cur_iteration += 1
            if cur_iteration > max_iteration:
                warnings.warn('Limit of iterations attained before buffer found. Current buffer allows up to {} levels.'.format(
                    pywt.dwt_max_level(n0, wave)))
                break
        return (n0-N) * fs**-1
except Exception as e:
    print(e)

try:
    import pycwt
    def bufferforfrequency(f0, dt=0.05, param=6, mother="MORLET", wavelet=pycwt.Morlet(6)):
        #check if f0 in right units
        # f0 ↴
        #    /\
        #   /  \
        #  /____\
        # 2 x buffer
        
        c = wavelet.flambda() * wavelet.coi()
        n0 = 1 + (2 * (1/f0) * (c * dt)**-1)
        N = int(np.ceil(n0 * dt))
        return N
except Exception as e:
    print(e)
