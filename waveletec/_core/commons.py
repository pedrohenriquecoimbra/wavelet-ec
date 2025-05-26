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
    'handle_eddypro_raw_dataset': True,
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
    '4thgas': '4th gas',
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
    handle_eddypro_raw_dataset = kw_csv.pop('handle_eddypro_raw_dataset', False)
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
        if handle_eddypro_raw_dataset:
            with open(path, 'r') as file: header = [c.replace('\n', '').strip() for c in file.readlines()[9].split('  ') if c]
            kw_csv.update({'skiprows': 10, 'sep': '\s+', 'na_values': ['NaN', 'nan', -9999], 'names': header})
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
    """
    path: str or dict 
    e.g. (str): path = str(os.path.join(inputpath, 'eddypro_raw_datasets/level_6'))
    e.g. (dict): path = {k: str(os.path.join(v, 'eddypro_raw_datasets/level_6')) for k, v in inputpath.items()}
        where k will become the suffix in case of duplicates
    """
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
        if kw_csv != DEFAULT_READ_CSV: kw_csv['handle_eddypro_raw_dataset'] = False
        
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
        df_site = df_site.set_index(kw.tname).join(pd.DataFrame({kw.tname: pd.date_range(*minmax, freq=str(kw.dt*1000) + 'ms')}).set_index(kw.tname),
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
        datarange = [pd.date_range(start=d0, end=d1, freq=f"{f_freq}min")[:-1] + pd.Timedelta(freq)]
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

