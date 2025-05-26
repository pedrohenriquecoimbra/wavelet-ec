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
from functools import reduce

# 3rd party modules
import yaml
import numpy as np
from itertools import chain
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression, RANSACRegressor, HuberRegressor, QuantileRegressor
import zipfile
from io import StringIO

# project modules
from .addons import *

##########################################
###     PROJECT CHOICES                           
##########################################

SITES_TO_STUDY = ['SAC']

month2season = lambda month: {1:1, 2:1, 3:2, 4:2, 5:2, 6:3, 7:3, 8:3, 9:4, 10:4, 11:4, 12:1}[month]

DEFAULT_COVARIANCE = ['co2*co2', 'h2o*h2o', 'ts*ts', 'co*co',  'ch4*ch4', 'n2o*n2o',
                      'w*co2', 'w*h2o', 'w*ts', 'w*co', 'w*ch4',  'w*n2o',
                      'w*co2|w*h2o', 'w*co2|w*co', 'w*co2|w*ch4', 'w*co2|w*ts', 'w*co2|w*h2o|w*co',
                      'w*h2o|w*co2', 'w*h2o|w*co', 'w*h2o|w*ch4', 'w*h2o|w*ts',
                      'w*co|w*co2',  'w*co|w*ts', 'w*co|w*ch4', 'w*co|w*h2o',
                      'w*ch4|w*co2',  'w*ch4|w*co', 'w*ch4|w*ts', 'w*ch4|w*h2o',
                      'w*ts|w*co2',  'w*ts|w*co', 'w*ts|w*ch4', 'w*ts|w*h2o',
                      ]


##########################################
###     GENERIC FUNCTIONS                           
##########################################


class structuredData:
    def __init__(self, **kwargs):
        for k, v in kwargs.items(): self.__dict__[k]=v
        pass


def start_logging(outputpath, **kwargs):
    """
    Start logging to a file in the specified output path.
    """
    logname = str(os.path.join(
        outputpath, f"log/current_{datetime.datetime.now().strftime('%y%m%dT%H%M%S')}.log"))
    mkdirs(logname)

    params = dict(filemode='a',
                   format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S',
                   level=logging.DEBUG,
                   force=True)
    params.update(kwargs)

    # with open(logname, "w+"): pass
    logging.basicConfig(filename=logname, **params)

    logging.captureWarnings(True)
    logging.info("STARTING THE RUN")


def save_locals(locals_, path, **kwargs):
    locals_ = {
        k: v for k, v in locals_.items()
        if isinstance(v, (str, int, float, list, dict, bool, type(None)))
    }
    with open(path, 'w') as stp:
        yaml.safe_dump(locals_, stp)


def available_combinations(interesting_combinations, variables_available=['u', 'v', 'w', 'ts', 'co2', 'h2o']):
        # Reduce interesting to possible
        possible_combinations = [sum([v not in variables_available for v in re.split('[*|]', t)])==0 for t in interesting_combinations]
        # Limit run to the realm of possible 
        varstorun = [t for t, p in zip(interesting_combinations, possible_combinations) if p]
        return varstorun


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
    """
    Recursively updates a nested dictionary `d` with values from another dictionary `u`.
    If a key in `u` maps to a dictionary and the corresponding key in `d` also maps to a dictionary,
    the function updates the nested dictionary in `d`. Otherwise, it overwrites the value in `d`.

    Args:
        d (dict): The dictionary to update.
        u (dict): The dictionary containing updates.

    Returns:
        dict: The updated dictionary.
    """
    # Iterate over each key-value pair in the update dictionary `u`
    for k, v in u.items():
        # Check if the current value is a dictionary
        if isinstance(v, dict):
            # If the corresponding value in `d` is also a dictionary, recursively update it
            # Use `d.get(k, {})` to handle cases where the key `k` is not already in `d`
            d[k] = update_nested_dict(d.get(k, {}), v)
        else:
            # If the value is not a dictionary, directly update/overwrite the key in `d`
            d[k] = v
    # Return the updated dictionary
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


def __input_to_series__(data, request):
    if data is None: return request

    columns = data.columns
    if request is None: return 0 
    elif isinstance(request, str) and request in columns: 
        request = data[request]
    elif isinstance(request, (list, tuple)) and all(isinstance(i, str) and (i in columns) for i in request): 
        request = np.sum(data[request], axis=1)
    return request


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
###     STATISTICS (PRINT)
##########################################


def custom_round(x, base=5):
    return base * np.round(x/base)


def nanminmax(x):
    return [np.nanmin(x), np.nanmax(x)]


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


def __fit_noise__(y_axis, x_axis, x_max=5, a=1):
    logger = logging.getLogger('fitnoise')
    """
    freqs = [1/j2sj(j, 1/dt) for j in np.arange(1,len(spec)+1)]
    a: noise color (white = 1)
    """
    r = structuredData()
    r.curve_0 = lambda f, b: np.log((f**a)*b)
    specna = np.where(np.isnan(y_axis[:x_max]) | (y_axis[:x_max] <= 0), False, True)
    try:
        x_axis = np.array(x_axis)
        y_axis  = np.array(y_axis)
        r.params_0, _ = curve_fit(r.curve_0, x_axis[:x_max][specna], np.log(y_axis[:x_max][specna]), bounds=(0, np.inf))
    except Exception as e:
        logger.error(f"UserWarning: {str(e)}")
        r.params_0 = [0]
    r.fit = np.array([(f**a)*r.params_0[0] for f in x_axis])
    return r


def sum_nan_arrays(a, b):
    ma = np.isnan(a)
    mb = np.isnan(b)
    return np.where(ma & mb, np.nan, np.where(ma, 0, a) + np.where(mb, 0, b))


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


##########################################
###     PLOT                           
##########################################


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


def j2sj(e, samp_rate=10): return 1/(samp_rate*(2**-float(e)))
def sj2j(s, samp_rate=10): return np.log(samp_rate*s) / np.log(2)

def get_dic_flux_data(data): return {k: copy.deepcopy(data.query(
    f"CO_SITE == '{k}'").reset_index(drop=True)) for k in data.CO_SITE.unique()}
