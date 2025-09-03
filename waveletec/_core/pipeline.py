"""
This script is a key part of the following publications:
    - Herig Coimbra, Pedro Henrique and Loubet, Benjamin and Laurent, Olivier and Mauder, Matthias and Heinesch, Bernard and 
    Bitton, Jonathan and Delpierre, Nicolas and Depuydt, Jérémie and Buysse, Pauline, Improvement of Co2 Flux Quality Through 
    Wavelet-Based Eddy Covariance: A New Method for Partitioning Respiration and Photosynthesis. 
    Available at SSRN: https://ssrn.com/abstract=4642939 or http://dx.doi.org/10.2139/ssrn.4642939

The main function is:  
- run_wt
    function: (1) gets data, (2) performs wavelet transform, (3) cross calculate variables, (4) averages by 30 minutes, (5) saves 
    call: run_wt()
    Input:
        a: 
    Return:
        b: 

- conditional_sampling
    function: split an array (n dimensions) into 4 arrays based on signal (+ or -) of itself and 2nd array 
    call: conditional_sampling()
    Input:
        args: arrays to be used as filter 
    Return:
        b: 

- universal_wt
    function: call any wavelet transform
    call: universal_wt()
    Input:
        a: 
    Return:
        b: 
"""

# built-in modules
import os
import re
import warnings
import logging
import copy
import time
import datetime
import glob

# 3rd party modules
from functools import reduce
import numpy as np
import pandas as pd
import itertools
from scipy.optimize import curve_fit
import pywt
import yaml
try: 
    import pycwt
except ImportError as e:
    pycwt = None
    pass
try: 
    import fcwt
except ImportError as e:
    fcwt = None
    pass

# project modules
from . import commons as hc24
from .read_data import loaddatawithbuffer
from .wavelet_functions import universal_wt, formula_to_vars, prepare_signal, bufferforfrequency_dwt, bufferforfrequency, inside_cone_of_influence
from ..partitioning.coimbra_et_al_2025 import conditional_sampling

logger = logging.getLogger('wvlt.pipeline')


def integrate_cospectra(data, f0, dst_path=None):
    data0 = data[(np.isnan(data['natural_frequency']) == False) * (data['natural_frequency'] >= f0)
                 ].groupby(['variable', 'TIMESTAMP'])['value'].agg(np.nansum).reset_index(drop=False)
    data1 = data[np.isnan(data['natural_frequency'])].drop(
        'natural_frequency', axis=1)

    datai = pd.concat([data1[np.isin(
        data1['variable'], data0['variable'].unique()) == False], data0]).drop_duplicates()
    datai = datai.pivot_table('value', 'TIMESTAMP',
                              'variable').reset_index(drop=False)

    if dst_path:
        datai.to_file(dst_path, index=False)
    return datai

def integrate_cospectra_from_file(root, f0, pattern='', dst_path=None):
    # use glob.glob to find files matching the pattern
    logger = logging.getLogger('wvlt.pipeline.integrate_cospectra_from_file')
    if isinstance(root, str):
        saved_files = {}
        for name in os.listdir(root):
            dateparts = re.findall(pattern, name, flags=re.IGNORECASE)
            if len(dateparts) == 1:
                saved_files[dateparts[0]] = os.path.join(root, name)

        def __read__(date, path):
            r = pd.read_csv(path, skiprows=11, sep=',')
            if 'natural_frequency' not in r.columns: 
                logger.warning(f'Skipping spectral file. Natural frequency column not found ({path}).')
                return pd.DataFrame()
            if r.natural_frequency.dtype != float: print(date, r.natural_frequency.dtype)
            r['TIMESTAMP'] = pd.to_datetime(date, format='%Y%m%d%H%M')
            return r

        data = pd.concat([__read__(k, v) for k, v in saved_files.items()])
    else:
        data = root
    
    return integrate_cospectra(data, f0, dst_path=dst_path)


def decompose_variables(data, variables=['w', 'co2'],
                     nan_tolerance=.3,
                     verbosity=1, identifier='0000', 
                     **kwargs):
    """    Calculate data decomposed with wavelet transform
    """
    # dictionary of wavelet transforms
    φ = {'info_names': []}
    sj = None

    # run by couple of variables (e.g.: co2*w -> mean(co2'w'))
    try:
        logger.debug(f"Debug variables are: {variables} and data columns are: {data.columns.tolist()}")
        assert len(
            variables), 'Empty list of covariances to run. Check available variables and covariances to be performed.'
        for var in variables:
            if var not in φ.keys():
                ready_signal = prepare_signal(
                    data[var], nan_tolerance=nan_tolerance, identifier=identifier)
                logger.debug(f"signal is ready: {ready_signal.signal.shape}")
                wt_signal = universal_wt(
                    signal=ready_signal.signal, **kwargs, inv=True)
                logger.debug(
                    f"wt_signal is ready: {wt_signal.wave.shape}, {wt_signal.sj}")
                # φ[var], sj
                φ[var] = wt_signal.wave
                φ[f'{var}_qc'] = np.where(ready_signal.signan, 0, wt_signal.coi)
                φ['info_names'] += [var, f'{var}_qc']
                logger.debug(f"wt_signal is done.")
        φ.update({'sj': wt_signal.sj})
        φ.update({'coi': wt_signal.coi})
    except Exception as e:
        logger.critical(e)
        print(f"Error in decompose_variables: {e}")
    # return φ
    return type('var_', (object,), φ)


def decompose_data(data, variables=['w', 'co2'], dt=0.05, method='dwt', nan_tolerance=.3, verbosity=1, **kwargs):
    """    Calculate data decomposed with wavelet transform
    """
    logger = logging.getLogger('wvlt.pipeline.decompose_data')
    assert method in [
        'dwt', 'cwt', 'fcwt'], "Method not found. Available methods are: dwt, cwt, fcwt"
    
    # φ = {}
    # run by couple of variables (e.g.: co2*w -> mean(co2'w'))
    info_t_startvarloop = time.time()
    
    # run wavelet transform
    φ = decompose_variables(data, variables=variables,
                              nan_tolerance=nan_tolerance, **kwargs)
    # sj = φ.pop('sj', None)
    # for v in variables:
    #     if v not in φ.keys():
    #         signal = np.array(data[v])
    #         signan = np.isnan(signal)
    #         N = len(signal)
    #         Nnan = np.sum(signan)
    #         if Nnan:
    #             if (nan_tolerance > 1 and Nnan > nan_tolerance) or (Nnan > nan_tolerance * N):
    #                 logger.warning(
    #                     f"UserWarning: Too much nans ({np.sum(signan)}, {np.round(100*np.sum(signan)/len(signal), 1)}%) in {data.TIMESTAMP.head(-1)[0]}.")
    #         if Nnan and Nnan < N:
    #             signal = np.interp(np.linspace(0, 1, N), 
    #                     np.linspace(0, 1, N)[signan == False],
    #                     signal[signan==False])
    #         φ[v], sj = universal_wt(signal, method, **wt_kwargs, inv=True)
    #         N = len(signal)
        

    logger.debug(f'\t\tDecompose all variables took {round(time.time() - info_t_startvarloop)} s (run_wt).')
    
    φs_names = []
    # φs_names = [f'valid_{l}' if l else 'valid' for l in φ.sj]
    logger.debug(f'\t\tφ.info_names ({φ.info_names}).')
    for n in φ.info_names:
        if vars(φ)[n].shape[0] > 1:
            for l in φ.sj:  # use '+ ['']' if __integrate_decomposedarrays_in_dictionary__
                if l: φs_names += [f'{n}_{l}'] 
                else: φs_names += [n] 
        else: φs_names += [n]

    logger.debug(f"\t\tvars(φ).keys(): {vars(φ).keys()}")
    logger.debug(f'\t\tφs_names: {φs_names} ({len(φs_names)}).')

    # transform 2D arrays to DataFrame
    values = [vars(φ)[n] for n in φ.info_names]
    logger.debug(f'\t\tTransform 2D arrays to DataFrame with columns `{"`; `".join(φs_names)}`.')
    logger.debug(f'\t\t{[np.array(v).shape for v in values]}.')
    __temp__ = hc24.matrixtotimetable(np.array(data.TIMESTAMP),
                                      np.concatenate(values, axis=0), columns=φs_names)
    
    __temp__.set_index('TIMESTAMP', inplace=True)
    # logger.debug(f'\t\tpd.MultiIndex.from_tuples: {[tuple(c.split("_")) for c in __temp__.columns]}.')
    __temp__.columns = pd.MultiIndex.from_tuples([tuple(c.rsplit('_', 1)) for c in __temp__.columns])
    __temp__ = __temp__.stack(1).reset_index(1).rename(columns={"level_1": "natural_frequency"}).reset_index(drop=False)

    #pattern = re.compile(r"^(?P<variable>.+?)_?(?P<natural_frequency>(?<=_)\d+)?$")
    #__temp__ = __temp__.melt(['TIMESTAMP'] + φ.keys())
    #__temp__ = pd.concat([__temp__.pop('variable').str.extract(pattern, expand=True), __temp__], axis=1)
    __temp__['natural_frequency'] = __temp__['natural_frequency'].apply(lambda j: 1/hc24.j2sj(j, 1/dt) if j else np.nan)
    return __temp__


def _calculate_product_from_formula_(data, formula='w*co2|w*h2o'):
    logger = logging.getLogger('wvlt.pipeline._calculate_product_from_formula_')

    φs = {}
    logger.debug(
        f"\t\tformula: {formula} ({type(formula)}).")

    formulavar = formula_to_vars(formula) if isinstance(
        formula, str) else formula
    xy_name = ''.join(formulavar.xy)
    if xy_name not in data.columns:
        for ci, c in enumerate(formulavar.xy):
            XY = XY * np.array(data[c]).conjugate() if ci else data[c]
        φs[xy_name] = XY
        logger.debug(
            f"\t\tDecomposed covariance shape: {XY.shape}, named: {xy_name}.")
    # ({round(XY.shape[1] / (24*60*60/20), 2)} days, for dt=20Hz)
                        
    # logger.debug(f"\t\tformulavar.condsamp_pair: {formulavar.condsamp_pair}.")
    # logger.debug(f"\t\tφs: {φs}.")
    # φ = pd.DataFrame(φs)
    # logger.debug(f"\t\tφ: {φ.head()}.")
    for cs in formulavar.condsamp_pair:
        cs_name = ''.join(cs)
        # logger.debug(f"\t\tcs_name: {cs_name} (from {cs}) | φs.keys(): {φs.keys()}.")
        if (cs_name not in φs.keys()) and (cs_name not in data.columns):
            for ci, c in enumerate(cs):
                # logger.debug(f"\t\tCurrent c: {c}.")
                # logger.debug(f"\t\tCurrent data: {data.head()}.")
                # logger.debug(f"\t\tDecomposed covariance shape: {data[c].shape}.")
                CS = CS * np.array(data[c]).conjugate() if ci else data[c]
            φs[cs_name] = CS

    # data = pd.concat([data, pd.DataFrame(φs)], axis=1)
    return pd.DataFrame(φs)


def _calculate_conditional_sampling_from_formula_(data, formula='w*co2|w*h2o'):
    logger = logging.getLogger('wvlt.pipeline._calculate_conditional_sampling_from_formula_')

    logger.debug(f"\t\tformula: {formula}.")
    formulavar = formula_to_vars(formula) if isinstance(formula, str) else formula
    
    logger.debug(f"\t\tformulavar: {formulavar}.")
    names = [''.join(formulavar.xy)] + [''.join(cs)
                                        for cs in formulavar.condsamp_pair]

    logger.debug(f"\t\tnames: {names} ({data.columns.to_list()}).")

    φc = conditional_sampling(
        np.array(data[names[0]]), *[np.array(data[n]) for n in names], names=names, label={1: "+", -1: "-"}) if names else {}

    # data = pd.concat([data, pd.DataFrame(φs)], axis=1)
    return pd.DataFrame(φc)


def __save_cospectra__(data, dst_path, overwrite=False, **meta):
    logger = logging.getLogger('wvlt.pipeline.__save_cospectra__')

    # saved_files = []

    info_t_startsaveloop = time.time()

    # for __datea__, __tempa__ in data.groupby(data.TIMESTAMP):
    # dst_path = output_path.format(pd.to_datetime(__datea__).strftime('%Y%m%d%H%M'))
    logger.debug(f'\t\tSaving {dst_path} with shape {data.shape}.')
    # if os.path.exists(dst_path): continue
    use_header = False

    if overwrite or (not os.path.exists(dst_path)):
        use_header = True
        header  = "wavelet_based_(co)spectra\n"
        header += f"--------------------------------------------------------------\n"
        header += f"TIMESTAMP_START = {meta.get('TIMESTAMP_START', min(data.TIMESTAMP))}\n"
        header += f"TIMESTAMP_END = {meta.get('TIMESTAMP_END', max(data.TIMESTAMP))}\n"
        header += f"N: {meta.get('N', len(data.TIMESTAMP))}\n"
        header += f"TIME_BUFFER [min] = {meta.get('buffer', np.nan)/60}\n"
        header += f"frequency [Hz]\n"
        header += f"y-axis -> wavelet_reconstructed\n"
        header += f"mother_wavelet -> {meta.get('method', '')}\n"
        header += f"acquisition_frequency [Hz] = {1/meta.get('dt', np.nan)}\n"
        header += f"averaging_interval [Min] = {meta.get('averaging', '')}\n"
        hc24.mkdirs(dst_path)
        with open(dst_path, 'w+') as part: part.write(header)
        # legitimate_to_write = 1
        logger.debug(f'\t\tSaving header of DataFrame took {round(time.time() - info_t_startsaveloop)} s.')
        # saved_files.append(dst_path)
    
    # if not legitimate_to_write: continue
    
    data.drop('TIMESTAMP', axis=1, inplace=True)
    with open(dst_path, 'a+', newline='') as part:
        data.to_file(part, header=use_header, chunksize=500, index=False)
    logger.debug(f'\t\tSaving DataFrame took {round(time.time() - info_t_startsaveloop)} s.')
    
    # del data
        
    #arr_slice = np.unique(data.TIMESTAMP, return_index=True)
    #for __datea__ in arr_slice[0]:
    #    dst_path = output_path.format(suffix, pd.to_datetime(__datea__).strftime('%Y%m%d%H%M'))
    #    if os.path.exists(dst_path+'.part'): os.rename(dst_path+'.part', dst_path)
    
    # return saved_files
    return

def process(#ymd, raw_kwargs, 
            datetimerange, fileduration, input_path, acquisition_frequency,
            covariance=None, output_folderpath=None, verbosity=1,
            overwrite=False, processing_time_duration="1D", 
            internal_averaging=None, dt=0.05, wt_kwargs={}, 
            integration_period=None,
            method="dwt", averaging=30, meta={}, **kwargs):
    logger = logging.getLogger('wvlt.pipeline.process')
    local_args = locals()

    def _date_from_yl(date):
        date = re.sub('[-: ]', '', str(date))
        if processing_time_duration.endswith("D"):
            date = date[:8]
        if processing_time_duration.endswith("H") or processing_time_duration.endswith("Min"):
            date = date[:12]
        return date
    
    def _validate_run(date, yl, compare_start=True, compare_end=False):
        # recheck if files exist and overwrite option
        # doesn't save time (maybe only save 5min)
        file_name = os.path.basename(output_pathmodel.format(date))
        part_name0 = file_name.rsplit('_', 1)[0] + '_' if compare_start else ''
        part_name1 = file_name.rsplit('.', 1)[-1] if compare_end else ''
        current_files = [p for p in os.listdir(os.path.dirname(
            output_pathmodel)) if p.startswith(part_name0) and p.endswith(part_name1)]

        if not overwrite and current_files: #file_name in os.path.exists(output_pathmodel.format(date)):
            logger.warning(
                "UserWarning: Skipping, file already exists ({}).".format(date))
            return False

        # # if all([os.path.exists(output_pathmodel.format(_yl.strftime('%Y%m%d%H%M'))) for _yl in yl[:-1]]):
        # if all([output_pathmodel.format(_yl.strftime('%Y%m%d%H%M')) in current_files for _yl in yl[:-1]]):
        #     logger.warning(
        #         "UserWarning: Skipping, file already exists ({}).".format(date))
        #     return False
        # # elif any([os.path.exists(output_pathmodel.format(_yl.strftime('%Y%m%d%H%M'))) for _yl in yl[:-1]]):
        # if any([output_pathmodel.format(_yl.strftime('%Y%m%d%H%M')) in current_files for _yl in yl[:-1]]):
        #     logger.warning("UserWarning: Continuing but some files already exist ({}), others don't ({}).".format(
        #         ', '.join([_yl.strftime('%Y%m%d%H%M') for _yl in yl[:-1] if output_pathmodel.format(_yl.strftime('%Y%m%d%H%M')) in current_files]),
        #         ', '.join([_yl.strftime('%Y%m%d%H%M') for _yl in yl[:-1] if not output_pathmodel.format(_yl.strftime('%Y%m%d%H%M')) in current_files]),
        #     ))
        #     return True

        if hc24.checkifinprogress(curoutpath_inprog):
            return False
        return True
        
    def _load_data():
        start_time = time.time()
        data = loaddatawithbuffer(yl, d1=None, freq=_f, buffer=buffer, f_freq=_f, **load_kwargs)
        if data.empty:
            if verbosity > 1:
                logger.warning(f"UserWarning: No file found ({date}, path: {load_kwargs.get('path', 'default')}).")
            return None
        logger.info(f'\tLoading data took {round(time.time() - start_time)} s.')
        return data
    
    def _exit():
        if os.path.exists(curoutpath_inprog):
            os.remove(curoutpath_inprog)

    # Group parameters for each function
    load_kwargs = {
        # 'datetimerange': datetimerange,
        # 'fileduration': fileduration,
        'path': input_path,
        # 'acquisition_frequency': acquisition_frequency,
        'fkwargs': {'dt': 1/acquisition_frequency},
        'fmt': kwargs.get("fmt", {}),  # allow user to override or extend
        **kwargs.get("load_kwargs", {}),  # allow user to override or extend,
    }

    # kwargs['fmt'] = kwargs.get('fmt', {})
    if 'gas4_name' in kwargs.keys():
        load_kwargs['fmt'].update({kwargs.pop('gas4_name'): '4th gas'})

    # raw_kwargs = {'path': input_path, 'fkwargs': {
    #     'dt': 1/acquisition_frequency}}
    # raw_kwargs.update({k: v for k, v in kwargs.items() if k in ['fmt']})

    transform_kwargs = {
        'dt': 1/acquisition_frequency,
        'method': method,
        # 'averaging': averaging,
        'varstorun': covariance or hc24.available_combinations(hc24.DEFAULT_COVARIANCE),
        **kwargs.get("transform_kwargs", {}),
        **wt_kwargs
    }

    output_kwargs = {
        'output_folderpath': output_folderpath,
        'overwrite': overwrite,
        'meta': {'acquisition_frequency': acquisition_frequency},
        **kwargs.get("output_kwargs", {})
    }
    
    logging_kwargs = {
        **kwargs.get("logging_kwargs", {})
    }

    general_config = {
        'sitename': kwargs.get('sitename', '00000'),
        'verbosity': verbosity,
        'processing_time_duration': processing_time_duration,
    }

    
    run_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    output_path = ""
    output_pathmodel = ""
    curoutpath_inprog = ""

    # method = kwargs.get('method', 'dwt')
    # sitename = kwargs.get('sitename', '00000')

    # if covariance is None:
    #     # TODO: get variables available from eddypro
    #     covariance = hc24.available_combinations(hc24.DEFAULT_COVARIANCE)

    # kwargs.update({'covariance': covariance,
    #           'processduration': processduration, })
    
    if output_folderpath is not None:
        hc24.start_logging(output_folderpath, **logging_kwargs)
        if method == 'cov':
            output_pathmodel = str(os.path.join(output_folderpath, str(
                general_config['sitename'])+'_cov_{}_'+run_time+'.csv'))
        else:
            output_pathmodel = str(os.path.join(output_folderpath, 'wavelet_full_cospectra', str(
                general_config['sitename'])+'_full_cospectra_{}_'+run_time+'.csv'))
        hc24.mkdirs(output_pathmodel)
        try:
            hc24.save_locals(local_args, os.path.join(output_folderpath, f'log/setup_{run_time}.yml'))
        except Exception as e:
            logger.warning(f"Could not save local arguments to file: {e}")
        # with open(, 'w+') as stp:
        #     yaml.safe_dump(local_args, stp)
    else:
        output_pathmodel = ""
    
    logger.debug(f'Output path: {output_pathmodel}.')
    
    ymd = [datetimerange.split(
        '-')[0], datetimerange.split('-')[1], f'{fileduration}min']
    
    # if isinstance(averaging, (list, tuple)):
    #     averaging = averaging[-1]
    # if internal_averaging is None:
    #     internal_averaging = averaging
    
    fulldata = pd.DataFrame()
    info_t_start = time.time()
    logger.info(f'In load_main.')

    if verbosity:
        print(f'\nRUNNING WAVELET TRASNFORM ({method})')
    
    _, _, _f = ymd
    ymd = hc24.list_time_in_period(*ymd, processing_time_duration, include='both')

    buffer = 0 if method == 'cov' else (
        bufferforfrequency_dwt(n_=_f, **wt_kwargs) / 2 if method == 'dwt'
        else bufferforfrequency(wt_kwargs.get("f0", 1/(3*60*60))) / 2
    )
    meta.update({'buffer': buffer})
    logger.debug(f"Buffer (s): {buffer}.")
    logger.info(f'Start date loop at {round(time.time() - info_t_start)} s (load_main).')

    # Skip two line
    prev_print = '\n'
    for yl in ymd:
        date = _date_from_yl(yl[0])

        print(prev_print, date, 'reading', ' '*10, sep=' ', end='\n')
        prev_print = '\x1B[1A\r'

        if output_pathmodel:
            curoutpath_inprog = output_pathmodel.format(date).rsplit(".", 1)[
                0] + ".inprogress"
            logger.debug(f'In progress file: {curoutpath_inprog}.')
            if not _validate_run(date, yl):
                continue
        
        data = _load_data()
        if data is None:
            _exit()
            continue
        

        try:
            # main run
            output_path = output_pathmodel.format("{}")
            # # run directly
            # output_kwargs.update({'output_path': output_path})
            # fulldata = main(data, period=[min(yl), max(yl)],
            #                 output_kwargs=output_kwargs, **transform_kwargs)
            
            # run by varstorun
            output_kwargs.update({'output_path': output_path + '.part'})
            allvars = transform_kwargs['varstorun']
            saved_files = []
            for f in allvars:
                transform_kwargs['varstorun'] = [f]
                output = main(data, period=[min(yl), max(yl)], 
                              meta=meta,
                              output_kwargs=output_kwargs, **transform_kwargs)
                saved_files.append(output.saved)
                fulldata = pd.concat([fulldata, output.data], axis=0)
                
            for f in [s for s_ in saved_files for s in s_]:
                if os.path.exists(f):
                    os.rename(f, f.replace('.part', ''))
        except Exception as e:
            logger.critical(e)
            print(str(e))

        _exit()
    
    logger.debug(f'End date loop at {round(time.time() - info_t_start)} s.')
    logger.debug(f"integration_period: {integration_period}.")
    logger.debug(f"output_pathmodel: {output_pathmodel}.")
    logger.debug(f"fulldata: {fulldata.head()}.")
    if output_pathmodel and not fulldata.empty:
        # timestamp = pd.Timestamp.now().strftime('%Y%m%dT%H%M%S_%f')
        dst_path = os.path.join(output_folderpath, os.path.basename(
            output_pathmodel.format(run_time)))
        if integration_period:
            fulldata = integrate_cospectra(fulldata, 1/integration_period, dst_path=None)
        fulldata.to_csv(dst_path, index=False)
    return fulldata


def main(data, varstorun, period=None, average_period='30min', output_kwargs={}, meta={}, **kwargs):
    logger = logging.getLogger('wvlt.pipeline.main')
    logger.info('In main.')
    logger.debug(f'Input data shape: {data.shape}.')
    # logger.debug(f'Input data shape: {data.head(1)}.')

    vars_unique = list(set([var for f in varstorun for var in formula_to_vars(f).uniquevars]))
    logger.debug(f'Unique vars: {vars_unique}.')

    logger.debug(f'Input data is ready, data shape is {data.shape}.')
    logger.debug(
        f'\n{data.head()}\n\n{min(data["TIMESTAMP"])} - {max(data["TIMESTAMP"])}\n\n{period}')

    # decompose all required variables
    wvvar = decompose_data(data, vars_unique,
                           nan_tolerance=.3,
                           verbosity=1, identifier='0000',
                           **kwargs)
    meta.update({'averaging': average_period,
                 'method': f"{kwargs.get('method', '')} ~{kwargs.get('mother_wavelet', '')}",
                 'dt': kwargs.get('dt', np.nan)})
    
    logger.debug(f'Decompose data is over, data shape is {wvvar.shape}.')
    logger.debug(f'\n{wvvar.head()}\n')
    
    # select valid dates
    if period: wvvar = wvvar[(wvvar['TIMESTAMP'] > period[0]) & (wvvar['TIMESTAMP'] < period[1])]
    wvvar = wvvar.reset_index(drop=True)

    logger.debug(f'Screen data over period of interest yielded data shape {wvvar.shape}.')

    # calculate covariance
    # wvout = _calculate_product_from_formula_(wvvar, varstorun)
    logger.debug(f'varstorun. {varstorun}')
    uniquecovs = list(set(
        [c for f in varstorun for c in formula_to_vars(f).combinations]))
    logger.debug(f'uniquecovs. {uniquecovs}')

    wvout = pd.concat(
        # [wvvar[['TIMESTAMP', 'natural_frequency']]] +
        [(_calculate_product_from_formula_(wvvar, formula=f)
        #   .drop(columns=formula_to_vars(f).uniquevars if i == 0 else ['TIMESTAMP'] + formula_to_vars(f).uniquevars)
          )
         for i, f in enumerate(uniquecovs)], axis=1)
    
    logger.debug(f'Calclate covariance is over.')

    growingdata = pd.concat([wvvar, wvout], axis=1)
    logger.debug(f'Growing data shape {growingdata.shape}.')

    # calculate conditional sampling    
    logger.debug(f'Calclate _calculate_conditional_sampling_from_formula_ is over. 0')
    wvcsp = pd.concat(
        # [wvvar[['TIMESTAMP', 'natural_frequency']]] +
        [_calculate_conditional_sampling_from_formula_(growingdata, f)
         for f in varstorun], axis=1)
         
    logger.debug(f'Calclate _calculate_conditional_sampling_from_formula_ is over, with data: {wvcsp.head()}.')

    # despike
    # denoise
    # smoothing
    # Y12 = smooth_2d_data(Y12, method='repeat', smoothing=smoothing)
    # for i in range(len(φcs)):
    #     φcs[i] = smooth_2d_data(
    #         φcs[i], method='repeat', smoothing=smoothing)

    # assemble data
    growingdata = pd.concat([growingdata, wvcsp], axis=1)
    
    logger.debug(
        f'Data is assembled: {growingdata.head()}.')

    # average
    for thisdate, thisdata in growingdata.groupby(growingdata['TIMESTAMP'].dt.floor(
            average_period)):
        thisdate_ = thisdate.strftime('%Y%m%d%H%M')
        meta.update({thisdate_: meta})
        meta[thisdate_].update({
                    'TIMESTAMP_START': min(thisdata['TIMESTAMP']),
                    'TIMESTAMP_END': max(thisdata['TIMESTAMP']),
                    'N': len(thisdata['TIMESTAMP'])})

    growingdata['TIMESTAMP'] = growingdata['TIMESTAMP'].dt.floor(
        average_period)
    __ID_COLS__ = list(
        set(['TIMESTAMP', 'natural_frequency']) & set(growingdata.columns))
    growingdata = growingdata.groupby(__ID_COLS__).agg(
        np.nanmean).reset_index(drop=False)
    
    # integrate

    # save in dataframe and .csv
    growingdata = (growingdata.sort_values(by=__ID_COLS__)
                .melt(__ID_COLS__))
    
    logger.debug(f"\tSaving data in {output_kwargs['output_path']}.")
    saved_files = []
    if output_kwargs.get('output_path', None):
        for thisdate, thisdata in growingdata.groupby(growingdata.TIMESTAMP):
            thisdate_ = thisdate.strftime('%Y%m%d%H%M')
            dst_path = output_kwargs.get('output_path').format(thisdate_)
            logger.debug(f"\t\t\t... in {dst_path}.")
            __save_cospectra__(thisdata, dst_path, **meta[thisdate_])
            saved_files += [dst_path]
    
    # rename saved files when done
    # os.rename(output_kwargs['output_path'].format(suffix, pd.datetime.now().strftime('%Y%m%dT%H%M%S_%f')),
    #           )

    # save in .nc
    return type('var_', (object,), {'data': growingdata, 'saved': saved_files})
