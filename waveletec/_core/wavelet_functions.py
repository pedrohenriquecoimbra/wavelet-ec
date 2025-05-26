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

# 3rd party modules
from functools import reduce
import numpy as np
import pandas as pd
import itertools
from scipy.optimize import curve_fit
import pywt
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
from .commons import j2sj
from . import wavelet_misc as wlmisc

logger = logging.getLogger('wvlt')


def __wavemother_str_pycwt__(name):
    wavelets = {w.lower(): vars(pycwt.mothers)[w] for w in ['Morlet', 'Paul', 'DOG', 'MexicanHat']}
    mother = wavelets[re.subn('[0-9]', '',  name.lower())[0]]
    if re.findall('[0-9]+', name.lower()): 
        mother = mother(int(re.findall('[0-9]+', name.lower())[0]))
    else:
        mother = mother()
    return mother


try:
    import pywt
    def bufferforfrequency_dwt(N=0, n_=None, fs=20, level=None, f0=None, max_iteration=10**4, wavelet='db6'):
        if level is None and f0 is None: f0 = 1/(2*60*60)  # 18
        lvl = level if level is not None else int(np.ceil(np.log2(fs/f0)))
        if n_ is None: n_ = fs * 60 * 30
        n0 = N
        cur_iteration = 0
        while True:
            n0 += pd.to_timedelta(n_)/pd.to_timedelta("1s") * fs if isinstance(n_, str) else n_
            if lvl <= pywt.dwt_max_level(n0, wavelet):
                break
            cur_iteration += 1
            if cur_iteration > max_iteration:
                warnings.warn('Limit of iterations attained before buffer found. Current buffer allows up to {} levels.'.format(
                    pywt.dwt_max_level(n0, wavelet)))
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
        
        if isinstance(wavelet, str): wavelet = __wavemother_str_pycwt__(wavelet)
        c = wavelet.flambda() * wavelet.coi()
        n0 = 1 + (2 * (1/f0) * (c * dt)**-1)
        N = int(np.ceil(n0 * dt))
        return N
except Exception as e:
    print(e)


def formula_to_vars(formula):
    """
    function: parse formula to variables
    call: formula_to_vars()
    Input:
        formula: string
    Return:
        var_: object with attributes xy, condsamp_pair, uniquevars
    """
    # parse formula
    xy = formula.split('|')[0].split('*')
    condsamp_pair = [v.split('*') for v in formula.split('|')[1:]]
    condsamp_flat = [c for cs in condsamp_pair for c in cs]
    combinations = list(set(['*'.join(xy)] + ['*'.join(cs) for cs in condsamp_pair]))

    return type('var_', (object,), {'xy': xy, 'condsamp_pair': condsamp_pair, 'condsamp_flat': condsamp_flat, 
                                    'uniquevars': list(set(xy + condsamp_flat)), 'combinations': combinations})


def __cwt__(input, fs, f0, f1, fn, nthreads=1, scaling="log", fast=False, norm=True, Morlet=6.0):
    """
    function: performs Continuous Wavelet Transform
    call: __cwt__()
    Input:
        a: 
    Return:
        b: 
    """

    #check if input is array and not matrix
    if input.ndim > 1:
        raise ValueError("Input must be a vector")

    #check if input is single precision and change to single precision if not
    if input.dtype != 'single':
        input = input.astype('single')

    morl = fcwt.Morlet(Morlet) #use Morlet wavelet with a wavelet-parameter

    #Generate scales

    if scaling == "lin":
        scales = fcwt.Scales(morl,fcwt.FCWT_LINFREQS,fs,f0,f1,fn)
    elif scaling == "log":
        scales = fcwt.Scales(morl,fcwt.FCWT_LOGSCALES,fs,f0,f1,fn)
    else:
        scales = fcwt.Scales(morl,fcwt.FCWT_LOGSCALES,fs,f0,f1,fn)

    _fcwt = fcwt.FCWT(morl, int(nthreads), fast, norm)

    output = np.zeros((fn,input.size), dtype='csingle')
    freqs = np.zeros((fn), dtype='single')
    
    _fcwt.cwt(input,scales,output)
    scales.getFrequencies(freqs)

    return freqs, output


def __icwt__(W, sj, dt, dj, Cd=None, psi=None, wavelet=None):
    """
    function: performs Inverse Continuous Wavelet Transform
    call: __icwt__()
    Input:
        W: (cross-)spectra
        sj: scales
        dt: sampling rate
        dj: frequency resolution
        Cd, psi: wavelet-specific coefficients
        wavelet: mother wavelet (w/ cdelta and psi(0) callables). Ignored if Cd and psi are given.
    Return:
        x: array
    """
    if wavelet is None: wavelet = pycwt.wavelet.Morlet(6)
    if isinstance(wavelet, str): wavelet = __wavemother_str_pycwt__(wavelet)
    if Cd is None: Cd = wavelet.cdelta
    if psi is None: psi = wavelet.psi(0)
        
    a, b = W.shape
    c = sj.size
    if a == c:
        sj_ = (np.ones([b, 1]) * sj).transpose()
    elif b == c:
        sj_ = np.ones([a, 1]) * sj
    
    x = (W.real / (sj_ ** .5)) * ((dj * dt ** .5) / (Cd * psi))
    return x

   
def __dwt__(*args, level=None, wavelet="db6"):
    """
    function: performs Discrete Wavelet Transform
    call: __dwt__()
    Input:
        *args: arrays (1D) to be transformed
        level: maximum scale (power of 2)
        wavelet: mother wavelet (comprehensible to pywt)
    Return:
        Ws: list of 2D arrays
    """
    Ws = []
    for X in args:
        Ws += [pywt.wavedec(X, wavelet, level=level)]
    # level = len(Ws[-1])-1
    return Ws


def waverec_2d(coeffs, N, wavelet, mode='symmetric'):
    """
    function: reconstructs 2D signal from wavelet coefficients
    call: waverec_2d()
    Input:
        coeffs: list of wavelet coefficients
        N: data length
        wavelet: mother wavelet (comprehensible to pywt)
        mode: wavelet mode (default is 'symmetric')
    Return:
        rec: reconstructed signal
        """
    def reconstruct_level(coeffs, level_to_keep):
        # Make a copy of the coeffs
        coeffs_copy = [np.zeros_like(c) for c in coeffs]

        # Keep only the desired level coefficients
        coeffs_copy[level_to_keep] = coeffs[level_to_keep]

        # Reconstruct signal
        return pywt.waverec(coeffs_copy, wavelet=wavelet, mode=mode)

    # Example: Reconstruct each level
    reconstructed_levels = []
    for level in range(len(coeffs)):
        rec = reconstruct_level(coeffs, level)
        reconstructed_levels.append(rec[:N])  # Adjust length if needed

    # reconstructed_levels[0] = Approximation (lowest frequency)
    # reconstructed_levels[1], [2], etc. = Details (higher frequencies)
    return reconstructed_levels


def __idwt__(*args, N, wavelet='db6', mode='symmetric'):
    """
    function: performs Inverse Discrete Wavelet Transform
    call: __idwt__()
    Input:
        *args: 2D arrays containing wavelet coefficient
        N: data lenght
        wavelet: mother wavelet (comprehensible to pywt)
    Return:
        Ys: list of 2D arrays
    """
    Ys = []
    for W in args:
        reconstructed_signal = waverec_2d(W, N, wavelet, mode=mode)
        Ys += [np.array(reconstructed_signal[::-1])]
    return Ys


def prepare_signal(signal, nan_tolerance=0.3, identifier='0000'):
    signal = np.array(signal)
    signan = np.isnan(signal)
    N = len(signal)
    Nnan = np.sum(signan)
    if Nnan:
        if (nan_tolerance > 1 and Nnan > nan_tolerance) or (Nnan > nan_tolerance * N):
            logger.warning(
                f"UserWarning ({identifier}): Too much nans ({np.sum(signan)}, {np.round(100*np.sum(signan)/len(signal), 1)}%).")
    if Nnan and Nnan < N:
        signal = np.interp(np.linspace(0, 1, N),
                            np.linspace(0, 1, N)[
            signan == False],
            signal[signan == False])
    return type('var_', (object,), {'signal': signal, 'N': N, 'Nnan': Nnan})


def universal_wt(signal, method='dwt', fs=20, f0=1/(3*60*60), f1=10, fn=180, 
                 dj=1/12, inv=True, **kwargs):
    """
    function: performs Continuous Wavelet Transform
    call: universal_wt()
    Input:
        signal: 1D array
        method: 'dwt', 'cwt', 'fcwt' (cwt but uses fast algorithm)
        fs: sampling rate (Hz)
        f0: highest scale (becomes level for DWT)
        f1: lowest scale (2x sampling rate)
        fn: number of scales (only used for CWT)
        dj: frequency resolution (only used for CWT)
        inv: . Default is True
        **kwargs: keyword arguments sent to wavelet transform and inverse functions 
    Return:
        wave: 2D array
        sj: scales 
    """
    assert method in [
        'dwt', 'cwt', 'fcwt'], "Method not found. Available methods are: dwt, cwt, fcwt"
       
    if method == 'fcwt':
        if fcwt is not None:
            """Run Continuous Wavelet Transform, using fast algorithm"""
            _l, wave = __cwt__(signal, fs, f0, f1, fn, **kwargs)
            sj = np.log2(fs/_l)
            if inv:
                wave = __icwt__(wave, sj=sj, dt=fs, dj=dj, **kwargs, 
                            wavelet=pycwt.wavelet.Morlet(6))
            sj = list(sj)
        else:
            logger.warning('UserWarning: Fast continuous wavelet transform (fcwt) not found. Running slow version.')
            method = 'cwt'
    
    elif method == 'cwt':
        if pycwt is not None:
            """Run Continuous Wavelet Transform"""
            wave, sj, _, _, _, _ = pycwt.cwt(
                signal, dt=1/fs, s0=2/fs, dj=dj, J=fn-1, **kwargs)
            sj = np.log2(sj*fs)
            if inv:
                wave = __icwt__(wave, sj=sj, dt=fs**-1, dj=dj, **kwargs)
            sj = list(sj)
        else:
            logger.warning('UserWarning: Continuous wavelet transform (cwt) not found. Running discrete version.')
            method = 'dwt'
    
    elif method== "dwt":
        """Run Discrete Wavelet Transform"""
        lvl = kwargs.pop('level', int(np.ceil(np.log2(fs/f0))))
        # _l if s0*2^j; fs*2**(-_l) if Hz; (1/fs)*2**_l if sec.
        sj = [_l for _l in np.arange(1, lvl+2, 1)]
        waves = __dwt__(signal, level=lvl, **kwargs)
        if inv:
            N = np.array(signal).shape[-1]
            waves = __idwt__(*waves, N=N, **kwargs)
            waves = waves[0].real
        # wave = waves[0][0]
    return waves, sj


def integrate_cospectra(root, f0, pattern='', dst_path=None):
    if isinstance(root, str):
        saved_files = {}
        for name in os.listdir(root):
            dateparts = re.findall(pattern, name, flags=re.IGNORECASE)
            if len(dateparts) == 1:
                saved_files[dateparts[0]] = os.path.join(root, name)

        def __read__(date, path):
            r = pd.read_csv(path, skiprows=11, sep=',')
            if 'natural_frequency' not in r.columns: 
                logger.warn(f'Skipping spectral file. Natural frequency column not found ({path}).')
                return pd.DataFrame()
            if r.natural_frequency.dtype != float: print(date, r.natural_frequency.dtype)
            r['TIMESTAMP'] = pd.to_datetime(date, format='%Y%m%d%H%M')
            return r

        data = pd.concat([__read__(k, v) for k, v in saved_files.items()])
    else:
        data = root
    
    data0 = data[(np.isnan(data['natural_frequency'])==False) * (data['natural_frequency'] >= f0)].groupby(['variable', 'TIMESTAMP'])['value'].agg(np.nansum).reset_index(drop=False)
    data1 = data[np.isnan(data['natural_frequency'])].drop('natural_frequency', axis=1)

    datai = pd.concat([data1[np.isin(data1['variable'], data0['variable'].unique())==False], data0]).drop_duplicates()
    datai = datai.pivot_table('value', 'TIMESTAMP', 'variable').reset_index(drop=False)
    
    if dst_path: datai.to_file(dst_path, index=False)
    else: return datai
    return

    
def decompose_data(data, variables=['w', 'co2'], dt=0.05, method='dwt', wt_kwargs={}, nan_tolerance=.3, verbosity=1):
    """    Calculate data decomposed with wavelet transform
    """
    assert method in [
        'dwt', 'cwt', 'fcwt'], "Method not found. Available methods are: dwt, cwt, fcwt"
    
    φ = {}
    # run by couple of variables (e.g.: co2*w -> mean(co2'w'))
    info_t_startvarloop = time.time()
    
    # run wavelet transform
    for v in variables:
        if v not in φ.keys():
            signal = np.array(data[v])
            signan = np.isnan(signal)
            N = len(signal)
            Nnan = np.sum(signan)
            if Nnan:
                if (nan_tolerance > 1 and Nnan > nan_tolerance) or (Nnan > nan_tolerance * N):
                    logger.warning(
                        f"UserWarning: Too much nans ({np.sum(signan)}, {np.round(100*np.sum(signan)/len(signal), 1)}%) in {data.TIMESTAMP.head(-1)[0]}.")
            if Nnan and Nnan < N:
                signal = np.interp(np.linspace(0, 1, N), 
                        np.linspace(0, 1, N)[signan == False],
                        signal[signan==False])
            φ[v], sj = universal_wt(signal, method, **wt_kwargs, inv=True)
            N = len(signal)
        

    logger.debug(f'\t\tDecompose all variables took {round(time.time() - info_t_startvarloop)} s (run_wt).')
        
    φs_names = []
    for n in φ.keys():
        if φ[n].shape[0] > 1:
            for l in sj: # use '+ ['']' if __integrate_decomposedarrays_in_dictionary__
                if l: φs_names += [f'{n}_{l}'] 
                else: φs_names += [n] 
        else: φs_names += [n]
    logger.debug(f'\t\tTransform 2D arrays to DataFrame with columns `{"`; `".join(φs_names)}`.')
    logger.debug(f'\t\t{[np.array(v).shape for v in list(φ.values())]}.')
    __temp__ = hc24.matrixtotimetable(np.array(data.TIMESTAMP),
                                      np.concatenate(list(φ.values()), axis=0), columns=φs_names)
    
    __temp__.set_index('TIMESTAMP', inplace=True)
    __temp__.columns = pd.MultiIndex.from_tuples([tuple(c.split('_')) for c in __temp__.columns])
    __temp__ = __temp__.stack(1).reset_index(1).rename(columns={"level_1": "natural_frequency"}).reset_index(drop=False)

    #pattern = re.compile(r"^(?P<variable>.+?)_?(?P<natural_frequency>(?<=_)\d+)?$")
    #__temp__ = __temp__.melt(['TIMESTAMP'] + φ.keys())
    #__temp__ = pd.concat([__temp__.pop('variable').str.extract(pattern, expand=True), __temp__], axis=1)
    __temp__['natural_frequency'] = __temp__['natural_frequency'].apply(lambda j: 1/j2sj(j, 1/dt) if j else np.nan)
    return __temp__


def calculate_product(data, formula='w*co2|w*h2o', verbosity=1):
    """    Calculate (conditioned) product
    """
    var_ = formula_to_vars(formula)

    for ci, c in enumerate(var_.xy): Y12 = Y12 * np.array(data[c]).conjugate() if ci else np.array(data[c])
    φs = {''.join(var_.xy): Y12}

    # conditional sampling
    names = [''.join(var_.xy)] + [''.join(cs) for cs in var_.condsamp_pair]
    φcs = []
    if var_.condsamp_pair: 
        φcs += [Y12]
    for cs in var_.condsamp_pair:
        for ci, c in enumerate(cs): φc0 = φc0 * np.array(data[c]).conjugate() if ci else np.array(data[c])
        φcs += [φc0]
    φc = conditional_sampling(Y12, *φcs, names=names, label={1:"+", -1: "-"}) if φcs else {}
    φs.update(φc)

    data = pd.concat([data, pd.DataFrame(φs)], axis=1)
    return data


def loop_variables(data, varstorun, dt=0.05, wt_kwargs={}, 
                   method="dwt", nan_tolerance=.3, averaging=30, verbosity=1):
    """
    averaging & integrating at the same unit

    fs = 20, f0 = 1/(3*60*60), f1 = 10, fn = 100, agg_avg = 1, 
    suffix = "", wavelet = pycwt.wavelet.MexicanHat(),
    **kwargs):
    """
    wt = {'dt': dt, 'method': method, 'wt_kwargs': wt_kwargs, 'nan_tolerance': nan_tolerance}
    ## START HERE FUNCTION RUN_COV

    # ensure time is time
    data.TIMESTAMP = pd.to_datetime(data.TIMESTAMP)
    
    # ensure continuity
    info_t_startdatacontinuity = time.time()
    data = pd.merge(pd.DataFrame({"TIMESTAMP": pd.date_range(np.nanmin(data.TIMESTAMP), np.nanmax(data.TIMESTAMP), freq=f"{dt*1000}ms")}),
                        data, on="TIMESTAMP", how='outer').reset_index(drop=True)
    timestamp0 = data['TIMESTAMP'].copy()
    timestamp = timestamp0.copy()
    logger.debug(f'\tForce data to be continuous took {round(time.time() - info_t_startdatacontinuity)} s (run_wt).')
    
    assert len(varstorun), 'Empty list of covariances to run. Check available variables and covariances to be performed.'
    allvars = []
    for v in varstorun:
        allvars += formula_to_vars(v).uniquevars

    allvars = list(set(allvars))
    this_data = data[['TIMESTAMP'] + allvars].copy()

    if method == 'cov':
        for c in allvars:
            this_data[c] = this_data[c] - this_data[c].groupby(this_data.TIMESTAMP.dt.floor(f'{averaging}min')).transform(np.nanmean)
        
        this_data = pd.concat(
            [(calculate_product(this_data, formula=f)
                .drop(columns=formula_to_vars(f).uniquevars if i == 0 else ['TIMESTAMP'] + formula_to_vars(f).uniquevars))
                for i, f in enumerate(varstorun)], axis=1)
        #this_data = calculate_product(this_data, formula=var_.formula)
        return this_data

    else:
        info_t_startdecomposedata = time.time()
        data_decomp = decompose_data(this_data, variables=allvars, **wt)
        logger.debug(f'\tDecompose data took {round(time.time() - info_t_startdecomposedata)} s (run_wt).')

        info_t_startconcatenatedata = time.time()

        data_decomp = pd.concat(
            [(calculate_product(data_decomp, formula=f)
                .drop(columns=formula_to_vars(f).uniquevars if i == 0 else ['TIMESTAMP', 'natural_frequency'] + formula_to_vars(f).uniquevars))
                for i, f in enumerate(varstorun)], axis=1)
        return data_decomp

    #[os.remove(d) for a in avg_ for d in dat_fullspectra[a] if os.path.exists(d)]
    #if os.path.exists(curoutpath_inprog): os.remove(curoutpath_inprog)
    #prev_print = '\x1B[2A\r' + f' {date} {len(yl)} files {int(100*ymd_i/len(ymd))} % ({time.strftime("%d.%m.%y %H:%M:%S")})' + '\n'
        
    #logger.info(f'\tFinish {date} took {round(time.time() - info_t_startdateloop)} s, yielded {len(yl)} files (run_wt). Progress: {len(yl)} {int(100*ymd_i/len(ymd))} %')
    data = pd.concat([data_decomp, this_data], axis=0)
    logger.debug(f'\tConcatenate data took {round(time.time() - info_t_startconcatenatedata)} s (run_wt).')
    return data

def load_data_and_loop(ymd, raw_kwargs, output_path=None, verbosity=1,
                  overwrite=False, processing_time_duration="1D", 
                  internal_averaging=None, dt=0.05, wt_kwargs={}, 
                  method="dwt", averaging=30, **kwargs):
    if isinstance(averaging, (list, tuple)): averaging = averaging[-1]
    if internal_averaging is None: internal_averaging = averaging
    fulldata = pd.DataFrame()

    info_t_start = time.time()
    logger.info('Entered wavelet code (run_wt).')

    if verbosity: print(f'\nRUNNING WAVELET TRASNFORM ({method})')

    suffix = '_cov' if method == 'cov' else '_full_cospectra'

    _, _, _f = ymd
    ymd = hc24.list_time_in_period(*ymd, processing_time_duration, include='both')
    
    if method == 'cov':
        buffer = 0
    elif method in ['dwt']:
        buffer = wlmisc.bufferforfrequency_dwt(n_=_f, **wt_kwargs)/2
    else:
        buffer = wlmisc.bufferforfrequency(wt_kwargs.get("f0", 1/(3*60*60))) / 2
    logger.debug(f"Buffer (s): {buffer}.")
        
    logger.info(f'Start date loop at {round(time.time() - info_t_start)} s (run_wt).')
    
    # Skip two line
    prev_print = '\n'
    for ymd_i, yl in enumerate(ymd):
        legitimate_to_write = 0
        info_t_startdateloop = time.time()

        date = re.sub('[-: ]', '', str(yl[0]))
        if processing_time_duration.endswith("D"): date = date[:8]
        if processing_time_duration.endswith("H") or processing_time_duration.endswith("Min"): date = date[:12]
        
        print(prev_print, date, 'reading', ' '*10, sep=' ', end='\n')
        prev_print = '\x1B[1A\r'

        if output_path:
            # recheck if files exist and overwrite option
            # doesn't save time (maybe only save 5min)
            if not overwrite and os.path.exists(output_path.format(suffix, date)):
                logger.warning("UserWarning: Skipping, file already exists ({}).".format(date))
                continue
            
            if all([os.path.exists(output_path.format(suffix, _yl.strftime('%Y%m%d%H%M'))) for _yl in yl[:-1]]): 
                logger.warning("UserWarning: Skipping, file already exists ({}).".format(date))
                continue
            elif any([os.path.exists(output_path.format(suffix, _yl.strftime('%Y%m%d%H%M'))) for _yl in yl[:-1]]):
                logger.warning("UserWarning: Continuing but some files already exist ({}), others don't ({}).".format(
                    ', '.join([_yl.strftime('%Y%m%d%H%M') for _yl in yl[:-1] if os.path.exists(output_path.format(suffix, _yl.strftime('%Y%m%d%H%M')))]), 
                    ', '.join([_yl.strftime('%Y%m%d%H%M') for _yl in yl[:-1] if not os.path.exists(output_path.format(suffix, _yl.strftime('%Y%m%d%H%M')))]), 
                    ))
                pass
            
            curoutpath_inprog = output_path.format(suffix, str(date), "").rsplit(".", 1)[
                0] + ".inprogress"
            if hc24.checkifinprogress(curoutpath_inprog): continue
        
        # load files
        info_t_startloaddata = time.time()
        data = hc24.loaddatawithbuffer(
            yl, d1=None, freq=_f, buffer=buffer, f_freq=_f, **raw_kwargs)
        if data.empty:
            if verbosity>1: logger.warning("UserWarning: No file was found ({}, path: {}).".format(date, raw_kwargs.get('path', 'default')))
            if output_path and os.path.exists(curoutpath_inprog): os.remove(curoutpath_inprog)
            continue
        logger.info(f'\tLoading data took {round(time.time() - info_t_startloaddata)} s (run_wt).')
        
        # LOOP FOR VARIABLES
        #__save_cospectra__(data, averaging=internal_averaging,
        #                          dt=dt, wt_kwargs=wt_kwargs, method=method, 
        #                          verbosity=verbosity, **kwargs)
        try:
            data = loop_variables(data, averaging=internal_averaging,
                                  dt=dt, wt_kwargs=wt_kwargs, method=method, 
                                  verbosity=verbosity, **kwargs)
            data['TIMESTAMP'] = data['TIMESTAMP'].dt.floor(f'{averaging}min')
            __ID_COLS__ = ['TIMESTAMP'] if method == "cov" else ['TIMESTAMP', 'natural_frequency']
            data = data.groupby(__ID_COLS__).agg(np.nanmean).reset_index(drop=False)
        except Exception as e:
            logger.critical(str(e))
            print(str(e))

        #[os.remove(d) for a in avg_ for d in dat_fullspectra[a] if os.path.exists(d)]
        if output_path and os.path.exists(curoutpath_inprog): os.remove(curoutpath_inprog)
        prev_print = '\x1B[2A\r' + f' {date} {len(yl)} files {int(100*ymd_i/len(ymd))} % ({time.strftime("%d.%m.%y %H:%M:%S")})' + '\n'
            
        logger.info(f'\tFinish {date} took {round(time.time() - info_t_startdateloop)} s, yielded {len(yl)} files (run_wt). Progress: {len(yl)} {int(100*ymd_i/len(ymd))} %')

        logger.debug(f'\tSaving data starting at {round(time.time() - info_t_start)} s (run_wt).')
        
        if method == 'cov':
            fulldata = pd.concat([fulldata, data], axis=0)
        else:
            # FILTER OUT THE BUFFER DATA, KEEP ONLY GOOD DATA 
            #data = data[data.TIMESTAMP>= ]
            data = (data.sort_values(by=['TIMESTAMP', 'natural_frequency'])
                    .melt(['TIMESTAMP', 'natural_frequency']))
            if output_path:
                __save_cospectra__(data, suffix, output_path, **{'method': method, 'averaging': averaging, 'buffer': buffer, 'dt': dt})
            else:
                fulldata = pd.concat([fulldata, data], axis=0)
    
    if output_path and not fulldata.empty:
            fulldata.to_csv(output_path.format(suffix, pd.datetime.now().strftime('%Y%m%dT%H%M%S_%f')), index=False)
    return fulldata


def __save_cospectra__(data, suffix, output_path, **meta):
    info_t_startsaveloop = time.time()
    for __datea__, __tempa__ in data.groupby(data.TIMESTAMP):
        dst_path = output_path.format(suffix, pd.to_datetime(__datea__).strftime('%Y%m%d%H%M'))
        if os.path.exists(dst_path): continue
        use_header = False
        if not os.path.exists(dst_path):
            use_header = True
            header  = "wavelet_based_(co)spectra\n"
            header += f"--------------------------------------------------------------\n"
            header += f"TIMESTAMP_START = {min(__tempa__.TIMESTAMP)}\n"
            header += f"TIMESTAMP_END = {max(__tempa__.TIMESTAMP)}\n"
            header += f"N: {len(__tempa__.TIMESTAMP)}\n"
            header += f"TIME_BUFFER [min] = {meta.get('buffer', '')/60}\n"
            header += f"frequency [Hz]\n"
            header += f"y-axis_->_wavelet_coefficient_*_\n"
            header += f"mother_wavelet -> {meta.get('method', '')}\n"
            header += f"acquisition_frequency [Hz] = {1/meta.get('dt', '')}\n"
            header += f"averaging_interval [Min] = {meta.get('averaging', '')}\n"
            hc24.mkdirs(dst_path)
            with open(dst_path, 'w+') as part: part.write(header)
            legitimate_to_write = 1
            logger.debug(f'\t\tSaving header of DataFrame took {round(time.time() - info_t_startsaveloop)} s (run_wt).')
        
        if not legitimate_to_write: continue
        
        __tempa__.drop('TIMESTAMP', axis=1, inplace=True)
        with open(dst_path, 'a+', newline='') as part: __tempa__.to_file(part, header=use_header, chunksize=500, index=False)
        
        del __tempa__
        
    #arr_slice = np.unique(data.TIMESTAMP, return_index=True)
    #for __datea__ in arr_slice[0]:
    #    dst_path = output_path.format(suffix, pd.to_datetime(__datea__).strftime('%Y%m%d%H%M'))
    #    if os.path.exists(dst_path+'.part'): os.rename(dst_path+'.part', dst_path)
    return


def run_wt(ymd, varstorun, raw_kwargs, output_path, wt_kwargs={}, 
           method="dwt", Cφ=1, nan_tolerance=.3,
           averaging=30, condsamp_flat=[], integrating=30*60, 
           overwrite=False, saveraw=False, processing_time_duration="1D",
           preaverage=None,
           despike=False, denoise=False, noisecolor=1, deadband={}, verbosity=1):
    """
    averaging & integrating at the same unit

    fs = 20, f0 = 1/(3*60*60), f1 = 10, fn = 100, agg_avg = 1, 
    suffix = "", wavelet = pycwt.wavelet.MexicanHat(),
    **kwargs):
    """
    if isinstance(averaging, (list, tuple)): averaging = averaging[-1]

    info_t_start = time.time()
    logger.info('Entered wavelet code (run_wt).')

    assert method in [
        'dwt', 'cwt', 'fcwt'], "Method not found. Available methods are: dwt, cwt, fcwt"
    if verbosity: print(f'\nRUNNING WAVELET TRASNFORM ({method})')
    if False and (method in ["cwt", "fcwt"]):
        if method == "fcwt" or "wavelet" not in wt_kwargs.keys() or wt_kwargs.get("wavelet") in ['morlet', 'Morlet', pycwt.wavelet.Morlet(6)]:
            Cφ = 2.5
        else:
            Cφ = 16.568
    
    dt = 1 / wt_kwargs.get("fs", 20)
    suffix = raw_kwargs['suffix'] if 'suffix' in raw_kwargs.keys() else ''

    _, _, _f = ymd
    ymd = hc24.list_time_in_period(*ymd, processing_time_duration, include='both')
    
    if method in ['dwt']:
        buffer = wlmisc.bufferforfrequency_dwt(
            N=pd.to_timedelta(processing_time_duration)/pd.to_timedelta("1s") * dt**-1,
            n_=_f, **wt_kwargs)/2
    else:
        buffer = wlmisc.bufferforfrequency(wt_kwargs.get("f0", 1/(3*60*60))) / 2
    logger.debug(f"Buffer (s): {buffer}.")
        
    logger.info(f'Start date loop at {round(time.time() - info_t_start)} s (run_wt).')
    
    def __despike__(X):
        N = len(X)
        X = hc24.mauder2013(X)
        Xna = np.isnan(X)
        try:
            X = np.interp(np.linspace(0, 1, N), 
                            np.linspace(0, 1, N)[Xna == False],
                    X[Xna==False])
        except Exception as e:
            logger.error(f"UserWarning: {str(e)}")
        return X
    
    def __fit_whitenoise__(spec, fmax=5, a=noisecolor):
        r = hc24.structuredData()
        freqs = [1/j2sj(j, 1/dt) for j in np.arange(1,len(spec)+1)]
        r.curve_0 = lambda f, b: np.log((f**a)*b)
        specna = np.where(np.isnan(spec[:fmax]) | (spec[:fmax] <= 0), False, True)
        try:
            freqs = np.array(freqs)
            spec  = np.array(spec)
            r.params_0, _ = curve_fit(r.curve_0, freqs[:fmax][specna], np.log(spec[:fmax][specna]), bounds=(0, np.inf))
        except Exception as e:
            logger.error(f"UserWarning: {str(e)}")
            r.params_0 = [0]
        r.fit = np.array([(f**a)*r.params_0[0] for f in freqs])
        return r

    # Skip two line
    prev_print = '\n'
    for ymd_i, yl in enumerate(ymd):
        legitimate_to_write = 0
        info_t_startdateloop = time.time()

        date = re.sub('[-: ]', '', str(yl[0]))
        if processing_time_duration.endswith("D"): date = date[:8]
        if processing_time_duration.endswith("H") or processing_time_duration.endswith("Min"): date = date[:12]
        
        print(prev_print, date, 'reading', ' '*10, sep=' ', end='\n')
        prev_print = '\x1B[1A\r'

        # recheck if files exist and overwrite option
        # doesn't save time (maybe only save 5min)
        if not overwrite and os.path.exists(output_path.format(suffix + "_full_cospectra", date)):
            logger.warning("UserWarning: Skipping, file already exists ({}).".format(date))
            continue
        
        if all([os.path.exists(output_path.format(suffix + "_full_cospectra", _yl.strftime('%Y%m%d%H%M'))) for _yl in yl[:-1]]): 
            logger.warning("UserWarning: Skipping, file already exists ({}).".format(date))
            continue
        elif any([os.path.exists(output_path.format(suffix + "_full_cospectra", _yl.strftime('%Y%m%d%H%M'))) for _yl in yl[:-1]]):
            logger.warning("UserWarning: Continuing but some files already exist ({}), others don't ({}).".format(
                ', '.join([_yl.strftime('%Y%m%d%H%M') for _yl in yl[:-1] if os.path.exists(output_path.format(suffix + "_full_cospectra", _yl.strftime('%Y%m%d%H%M')))]), 
                ', '.join([_yl.strftime('%Y%m%d%H%M') for _yl in yl[:-1] if not os.path.exists(output_path.format(suffix + "_full_cospectra", _yl.strftime('%Y%m%d%H%M')))]), 
                ))
            pass

        """
        if not overwrite:
            avg_ = []
            for a in averaging:
                if not overwrite and os.path.exists(output_path.format(suffix + "_full_cospectra", date, str(a).zfill(2))):
                    avg_ += [a]
            avg_ = list(set(averaging)-set(avg_))
            if not avg_:
                if verbosity > 1: logger.warning("UserWarning: Skipping, file already exists ({}).".format(date))
                continue
        else:
            avg_ = [a for a in averaging]
        """
        
        curoutpath_inprog = output_path.format(suffix, str(date), "").rsplit(".", 1)[
            0] + ".inprogress"
        if hc24.checkifinprogress(curoutpath_inprog): continue
        
        # load files
        # data = get_rawdata.open_flux(lookup=yl, **raw_kwargs).data
        info_t_startloaddata = time.time()
        data = hc24.loaddatawithbuffer(
            yl, d1=None, freq=_f, buffer=buffer, f_freq=_f, **raw_kwargs)
        if data.empty:
            if verbosity>1: logger.warning("UserWarning: No file was found ({}, path: {}).".format(date, raw_kwargs.get('path', 'default')))
            if os.path.exists(curoutpath_inprog): os.remove(curoutpath_inprog)
            continue
        logger.info(f'\tLoading data took {round(time.time() - info_t_startloaddata)} s (run_wt).')
        
        ## START HERE FUNCTION RUN_COV

        # ensure time is time
        data.TIMESTAMP = pd.to_datetime(data.TIMESTAMP)
        
        # ensure continuity
        info_t_startdatacontinuity = time.time()
        data = pd.merge(pd.DataFrame({"TIMESTAMP": pd.date_range(np.nanmin(data.TIMESTAMP), np.nanmax(data.TIMESTAMP), freq=f"{dt*1000}ms")}),
                            data, on="TIMESTAMP", how='outer').reset_index(drop=True)
        timestamp0 = data['TIMESTAMP'].copy()
        timestamp = timestamp0.copy()
        logger.debug(f'\tForce data to be continuous took {round(time.time() - info_t_startdatacontinuity)} s (run_wt).')

        # main run
        # . collect all wavelet transforms
        # . calculate covariance
        # . conditional sampling (optional)
        # . save in dataframe and .csv
        φ = {}
        φ0 = {}
        σ = {}
        #ζ = {}
        ζb = {}
        S = {}
        μ = {}
        dat_fullspectra_file = []  # {a: [] for a in avg_}
        dat_fullspectra = []  # {a: [] for a in avg_}
        dat_fluxresult = []  # {a: [] for a in avg_}

        print(prev_print, date, 'decomposing', ' '*10, sep=' ', end='\n')
        # run by couple of variables (e.g.: co2*w -> mean(co2'w')) 
        logger.info(f'\tStarting variables loop at {round(time.time() - info_t_start)} s (run_wt).')
        info_t_dataframe     = 0 
        info_t_varstorunloop = 0
        info_c_varstorunloop = 0
        try:
            assert len(varstorun), 'Empty list of covariances to run. Check available variables and covariances to be performed.'
            for thisvar_i, thisvar in enumerate(varstorun):
                info_t_startvarloop = time.time()
                
                allvars = thisvar.replace('|', '*').split('*')
                xy = thisvar.split('|')[0].split('*')
                condsamp_pair = [v.split('*') for v in thisvar.split('|')[1:]]
                condsamp_flat = [c for cs in condsamp_pair for c in cs]
                #for xy, condsamp_flat in [(v.split('*')[:2], v.split('*')[2:]) for v in varstorun]:
                if '|' in thisvar: logger.debug(f"\t\tCall {thisvar} ({'*'.join(xy)} conditioned by {thisvar.split('|', 1)[1]}).")
                else: logger.debug(f"\t\tCall {thisvar}.")
                
                # run wavelet transform
                for v in xy + condsamp_flat:
                    if v not in φ.keys():
                        info_c_varstorunloop += 1
                        info_t_startdecomposition_v = time.time()  
                        signal = np.array(data[v])
                        signan = np.isnan(signal)
                        N = len(signal)
                        Nnan = np.sum(signan)
                        if Nnan:
                            if (nan_tolerance > 1 and Nnan > nan_tolerance) or (Nnan > nan_tolerance * N):
                                logger.warning(
                                    f"UserWarning: Too much nans ({np.sum(signan)}, {np.round(100*np.sum(signan)/len(signal), 1)}%) in {date}.")
                        if Nnan and Nnan < N:
                            signal = np.interp(np.linspace(0, 1, N), 
                                    np.linspace(0, 1, N)[signan == False],
                                    signal[signan==False])
                        φ[v], sj = universal_wt(signal, method, **wt_kwargs, inv=True)
                        levels_to_integrate = [dt*2**l < integrating for l in sj]
                        levels_notto_integrate = [l==False for l in levels_to_integrate]

                        # Strip buffer data
                        _isinfinalrange = (timestamp0 >= min(yl)) & (timestamp0 < max(yl))
                        signal = signal[_isinfinalrange.to_list()]
                        signan = signan[_isinfinalrange.to_list()]
                        φ[v] = φ[v][:, _isinfinalrange.to_list()]
                        N = len(signal)
                        S[v] = signal

                        # apply despiking (Mauder et al.)
                        if despike:
                            φ[v] = np.apply_along_axis(__despike__, 1, φ[v])
                        
                        φ0[v] = copy.deepcopy(φ[v])
                        μ[v] = signan *1
                        μ[v+'D'] = signan *1
                        μ[v+'T'] = signan *1
                        
                        """
                        # apply Allan deviation (ad-hoc)
                        frequencies = [dt*2**l for l in sj]
                        # read allan deviation file
                        # interpolate and find frequencies

                        allan_dev = {'co2': np.array([0.014, 0.014, 0.014, 0.014, 0.013, 0.01,  0.007, 0.005, 0.004, 0.004, 0.003, 0.002, 0.002, 0.001, 0.002, 0.003, 0.003, np.inf]),
                                     'co': np.array([3.095, 3.095, 3.095, 3.095, 2.853, 2.206, 1.393, 0.998, 0.674, 0.544, 0.418, 0.265, 0.168, 0.124, 0.103, 0.067, 0.086, np.inf]),
                                     'ch4': np.array([0.089, 0.089, 0.089, 0.089, 0.083, 0.065, 0.043, 0.034, 0.027, 0.023, 0.018, 0.013, 0.01,  0.011, 0.018, 0.027, 0.041, np.inf])}
                        allan_dev = {k: np.expand_dims(v, 1) for k, v in allan_dev.items()}

                        #filter values
                        if (N > 0) and (v.lower() in allan_dev.keys()):
                            # φ[v] needs to be (n, 17)
                            φ0[v] = np.where((abs(φ[v]) < np.sqrt(allan_dev[v])), 0, φ0[v])
                        
                        #if still some values
                        ##fit theorical (co)spectrum
                        ##fill spectra using theorical curve
                        """

                        # for file length, update later
                        σ[v]  = []
                        σ[v+'D'] = []
                        σ[v+'T'] = []
                        #ζ[v]  = []
                        ζb[v] = []

                        #arr_slice = np.unique(data[_isinfinalrange].TIMESTAMP.dt.floor(_f), return_index=True)
                        arr_slice = np.unique(timestamp0[_isinfinalrange].dt.floor(_f), return_index=True)
                        for this_group, this_data in list(zip(arr_slice[0], np.split(timestamp0[_isinfinalrange], arr_slice[1][1:]))):
                            #for this_group, this_data in data.groupby(data[_isinfinalrange].TIMESTAMP.dt.floor(_f)):
                            # Everytime creating working with the whole array, maybe faster to work on a split of the array (redability)
                            # List of T/F to select values that are in the current AVERAGING period
                            #this_hh = np.isin(data[_isinfinalrange].index, this_data.index)#np.where(data.TIMESTAMP > this_yl) & (data.TIMESTAMP <= (this_yl + pd.Timedelta(_f))), True, False)
                            this_hh = np.isin(timestamp0[_isinfinalrange].index, this_data.index)
                            this_Sv = np.array(S[v])
                            this_trend = np.nansum(np.array(φ[v])[levels_notto_integrate], axis=0)
                            this_N = len(this_Sv[this_hh])
                            σv = this_Sv - np.nanmean(this_Sv[this_hh])
                            σ[v]  += list(σv[this_hh])
                            σ[v+'D']  += list(this_Sv[this_hh] - this_trend[this_hh])
                            σ[v+'T']  += list(this_trend[this_hh])
                                
                            if v in deadband.keys():
                                this_hh = this_hh & (abs(σv) > abs(deadband[v])) #(σv > deadband[v])
                                φ0[v] = np.where((abs(σv) < abs(deadband[v])) * this_hh.reshape(1, -1), 0, φ0[v])

                            if denoise and (this_data.empty==False):
                                this_φv = np.array(φ[v])[:, this_hh]
                                σ2_v = np.nanmean(σv**2)
                                freq_noise_fit = 5
                                if dt==0.1: freq_noise_fit = 4
                                # np.nanmedian(abs(this_φv), axis=1)
                                noise_fitted = __fit_whitenoise__(np.nanmedian((this_φv**2)/σ2_v, axis=1), freq_noise_fit)
                                
                                this_noise = abs(np.sqrt(noise_fitted.fit * σ2_v).reshape(-1, 1))

                                #ζ[v]  += [this_noise]*len(this_Sv)
                                ζb[v] += [noise_fitted.params_0[0]]*this_N
                                
                                φ0[v] = np.where((abs(φ0[v]) > abs(this_noise)) * this_hh.reshape(1, -1) * (φ0[v] > 0), φ0[v] - this_noise, φ0[v])
                                φ0[v] = np.where((abs(φ0[v]) > abs(this_noise)) * this_hh.reshape(1, -1) * (φ0[v] < 0), φ0[v] + this_noise, φ0[v])
                                φ0[v] = np.where((abs(φ0[v]) <= abs(this_noise)) * this_hh.reshape(1, -1), 0, φ0[v])
                            else:
                                #ζ[v] += list(np.zeros((this_N, 1)))
                                ζb[v] += list(np.zeros((this_N, 1)) * np.nan)
                            
                        
                        # arrayify
                        φ[v] = np.array(φ[v])
                        φ0[v] = np.array(φ0[v])
                        σ[v] = np.array(σ[v]).reshape(1,-1)
                        σ[v+'D'] = np.array(σ[v+'D']).reshape(1,-1)
                        σ[v+'T'] = np.array(σ[v+'T']).reshape(1,-1)
                        #logger.debug(f'\t\tDecompose all variables took {σ[v].shape} / {σ[v+"D"].shape} s (run_wt).')
                        #ζ[v] = np.array(ζ[v]).T
                        ζb[v] = np.array(ζb[v]).reshape(1,-1)
                        S[v] = np.array(S[v]).reshape(1,-1)
                        #μ[v] = np.array(μ[v]).reshape(1,-1)
                logger.debug(f'\t\tDecompose all variables took {round(time.time() - info_t_startvarloop)} s (run_wt).')
                info_t_varstorunloop += time.time() - info_t_startvarloop

                info_t_startcovariance = time.time()
                # Strip definitively buffer data
                # data = data[_isinfinalrange]

                # calculate covariance
                for ci, c in enumerate(xy): Y12 = Y12 * φ[c].conjugate() if ci else φ[c] * Cφ
                for ci, c in enumerate(xy): Y120 = Y120 * φ0[c].conjugate() if ci else φ0[c] * Cφ
                for ci, c in enumerate(xy): Cov = Cov * σ[c] if ci else copy.deepcopy(σ[c])
                                
                CovT = {}
                for t in ['T', 'D']:
                    for co in set(itertools.combinations(['', t]*len(xy), len(xy))):
                        for ci, c in enumerate(xy): Cov_ = Cov_ * σ[c+co[ci]] if ci else copy.deepcopy(σ[c+co[0]])
                        CovT['/'.join([c+co[ci] for ci, c in enumerate(xy)])] = Cov_
                
                # addictive so it only remains which is > uncertainty
                #for ci, c in enumerate(xy): Y12ζ = Y12 * φ[c].conjugate() if ci else φ[c] * Cφ
                #for ci, c in enumerate(xy): Y12ζ = np.where((abs(φ[c]) > abs(ζ[c])) & (abs(φ[c])>0), 
                #                                            Y12ζ * (φ[c].conjugate() - (ζ[c] * φ[c]/abs(φ[c]))), 
                #                                            0) if ci else φ[c] * Cφ
                #Y12ζ = (abs(Y12) - abs(ζ[xy[-1]])) * Y12/abs(Y12)
                #print(date, f"\t call: {thisvar} ({'*'.join(xy)}", f"conditioned by {thisvar.split('|', 1)[1]})" if '|' in thisvar else ')',
                #    'main', '\t shape:', Y12.shape, '\t N (days):', round(Y12.shape[1] / (24*60*60/dt), 2), '\t buffer (s):', buffer)
                logger.debug(f"\t\tDecomposed covariance shape: {Y12.shape} ({round(Y12.shape[1] / (24*60*60/dt), 2)} days) (run_wt).")
                
                φs = {''.join(xy): Y12, '/'.join(xy)+'_cov': Cov.reshape(1,-1)}
                φs.update({f'{k}_cov': v.reshape(1,-1) for k, v in CovT.items()})
                φs.update({f'{k}_var': (v**2).reshape(1,-1) for k, v in σ.items()})
                μs = {''.join(xy): np.where(np.where(
                    np.array(μ[xy[0]]), 0, 1) * np.where(np.array(μ[xy[1]]), 0, 1), 0, 1).reshape(1,-1)}
                μs.update({'/'.join(xy)+'_cov': μs[''.join(xy)]})
                # Ad-hoc solution
                μs.update({f'{k}_cov': μs[k.replace('D', '').replace('T', '')+'_cov'] for k in CovT.keys()})
                μs.update({f'{k}_var': np.where(np.array(μ[k]), 0, 1).reshape(1,-1) for k in σ.keys()})
                
                logger.debug(f'\t\tCalculate covariances took {round(time.time() - info_t_startcovariance)} s (run_wt).')
                
                #Pre average to avoid (e.g.: 1 min)
                #It has to be divisor of 30 min to preserve 30 min average
                info_t_startpreaveraging = time.time()
                if preaverage:
                    arr_slice = np.unique(timestamp0[_isinfinalrange].dt.floor(preaverage), return_index=True)
                    #print(arr_slice[1][1:])
                    #print(len([np.nanmean(s) for s in np.split(signal, arr_slice[1][1:])]))
                    #print([len(s) for s in [s for s in np.split(signal, arr_slice[1][1:])]])
                    #signal = np.array([np.nanmean(s) for s in np.split(signal, arr_slice[1][1:])])
                    #signan = np.array([np.nanmean(s) for s in np.split(signan, arr_slice[1][1:])])
                    #φ[v] = np.array([np.nanmean(s, axis=0) for s in np.split(φ[v].T, arr_slice[1][1:])]).T
                    
                    φ0 = {k: np.array([np.nanmean(s, axis=0) for s in np.split(v.T, arr_slice[1][1:])]).T for k, v in φ0.items()}
                    φs = {k: np.array([np.nanmean(s, axis=0) for s in np.split(v.T, arr_slice[1][1:])]).T for k, v in φs.items()}
                    logger.debug(f'\tPreaveraging φs took {round(time.time() - info_t_startpreaveraging)} s (run_wt).')
                    μs = {k: np.array([np.nanmean(s, axis=0) for s in np.split(v.T, arr_slice[1][1:])]).T for k, v in μs.items()}
                    logger.debug(f'\tPreaveraging μs took {round(time.time() - info_t_startpreaveraging)} s (run_wt).')
                    ζb = {k: np.array([np.nanmean(s, axis=0) for s in np.split(v.T, arr_slice[1][1:])]).T for k, v in ζb.items()}
                    logger.debug(f'\tPreaveraging ζb took {round(time.time() - info_t_startpreaveraging)} s (run_wt).')
                    Y12 = np.array([np.nanmean(s, axis=0) for s in np.split(Y12.T, arr_slice[1][1:])]).T
                    Y120 = np.array([np.nanmean(s, axis=0) for s in np.split(Y120.T, arr_slice[1][1:])]).T
                    logger.debug(f'\tPreaveraging Y12 took {round(time.time() - info_t_startpreaveraging)} s (run_wt).')
                    timestamp = pd.Series(arr_slice[0])
                else:
                    timestamp = timestamp0[_isinfinalrange].copy()
                logger.debug(f'\tPreaveraging took {round(time.time() - info_t_startpreaveraging)} s (run_wt).')
                
                info_t_startconditionalsampling = time.time()
                # conditional sampling
                names = [''.join(xy)] + [''.join(cs) for cs in condsamp_pair]
                φcs = []
                if condsamp_pair: 
                    φcs += [Y120]#[Y12ζ]
                for cs in condsamp_pair:
                    for ci, c in enumerate(cs): φc0 = φc0 * φ0[c].conjugate() if ci else φ0[c]
                    # if variables to condition are uncertain do not consider them
                    # and put them in uncertain bag (Y12ζ) 
                    #for ci, c in enumerate(cs): φc0  = np.where(abs(φ[c]) > abs(ζ[c]), φc0, 0)
                    #for ci, c in enumerate(cs): Y12ζ = np.where(abs(φ[c]) > abs(ζ[c]), Y12ζ, 0)
                    φcs += [φc0]
                
                if ''.join(xy) in ['wco2', 'co2w', 'wh2o', 'h2ow']:
                    σcec = conditional_sampling(Cov, *[σ['w'], σ['co2'], σ['h2o']], 
                                                names=[''.join(xy), 'w', 'co2', 'h2o'],
                                                label={1: "+", -1: "-"})
                    φs.update(σcec)

                """
                names = [''.join(xy)] + [xy[0]+c for c in condsamp_flat]
                φc = [np.array(φ[xy[0]]) * np.array(φ[c]).conjugate() if xy[0] != c else np.array(φ[c]) for c in condsamp_flat]
                """
                #Y12ζ
                logger.debug(f'\t\tDifference between original and denoised {np.nanmean(abs(Y12 - Y120))} for array w/ shape {Y12.shape} and {Y120.shape} (run_wt).')
                φc = conditional_sampling(Y12, *φcs, names=names) if φcs else {}
                if φc: logger.debug(f'\t\tDifference between original and partitioned {np.nanmean(abs(Y12 - np.nansum(list(φc.values()), 0)))} for sum of {φc.keys()} (run_wt).')
                #φs.update({k.replace("xy", ''.join(xy)).replace('a', ''.join(condsamp_flat)): v for k, v in φc.items()})
                φs.update(φc)
                
                # ζY12 should be:
                # - 0, when  φ = 0
                # - φ, when |ζ|<|φ|
                # - ζ, when |ζ|>|φ|
                ζY12 = Y12 - Y120 #Y12ζ  # np.where(Y12ζ==0, np.max(Y12, ), 0)
                logger.debug(f'\t\tDone normal conditional sampling took {round(time.time() - info_t_startconditionalsampling)} s (run_wt).')
                names0 = [''.join(xy) + '·' + ''.join(condsamp_pair[0])] + [''.join(cs) for cs in condsamp_pair[1:]] if φcs else []
                φc0 = conditional_sampling(ζY12, *φcs[1:], names=names0) if φcs else {}
                #φs.update({k.replace("xy", ''.join(xy)).replace('a', ''.join(condsamp_flat)): v for k, v in φc.items()})
                φs.update(φc0)
                
                if φc and φc0: logger.debug(f'\t\tDifference between original and partitioned {np.nanmean(abs(Y12 - np.nansum(list(dict(φc, **φc0).values()), 0)))} for sum of {φc.keys()} (run_wt).')
                
                # add noise columns
                φs.update({''.join(xy) + '·': ζY12})

                logger.debug(f'\t\tConditional sampling took {round(time.time() - info_t_startconditionalsampling)} s (run_wt).')
                
                # repeats nan flag wo/ considering conditional sampling variables
                μs.update(
                    {k: μs[k if k in μs.keys() else [k_ for k_ in μs.keys() if k.startswith(k_)][0]] for k in φs.keys()})
                logger.debug(f'\t\tAdding nan flag took {round(time.time() - info_t_startconditionalsampling)} s (run_wt).')

                info_t_startdataframe = time.time()
                # array to dataframe for averaging
                
                # Integrate arrays are 2D inside dictionary (needed for concatenate)
                def __integrate_decomposedarrays_in_dictionary__(array_dic):
                    levels_to_integrate = [1/j2sj(l, 1/dt) >= 1/integrating for l in sj]
                    for k in array_dic.keys(): 
                        if array_dic[k].shape[0] == len(sj):
                            array_dic[k] = np.concatenate([
                                array_dic[k], np.nansum(array_dic[k][levels_to_integrate], axis=0).reshape(1,-1)], axis=0)
                    return array_dic
                φs = __integrate_decomposedarrays_in_dictionary__(φs)
                logger.debug(f'\t\tIntegration took {round(time.time() - info_t_startconditionalsampling)} s (run_wt).')
                
                # Guarantee arrays are 2D inside dictionary (needed for concatenate)
                def __guarantee_2d_arrays_in_dictionary__(array_dic):
                    for k in array_dic.keys(): 
                        if len(array_dic[k].shape) == 1:
                            array_dic[k] = array_dic[k].reshape(1, -1)
                    return array_dic
                φs = __guarantee_2d_arrays_in_dictionary__(φs)
                μs = __guarantee_2d_arrays_in_dictionary__(μs)
                ζb = __guarantee_2d_arrays_in_dictionary__(ζb)
                logger.debug(f'\t\tGuarantee took {round(time.time() - info_t_startconditionalsampling)} s (run_wt).')
                
                φs_names = []
                for n in φs.keys():  
                    if φs[n].shape[0] > 1:
                        for l in sj + ['']: # use if __integrate_decomposedarrays_in_dictionary__
                            if l: φs_names += [f'{n}_{l}'] 
                            else: φs_names += [n] 
                    else: φs_names += [n]            
                # [f'{n}_{l}' if l else n for n in φs.keys() for l in sj + [''] if φs[n].shape[0] > 1] +
                # [n for n in φs.keys() if φs[n].shape[0] <= 1] + 
                logger.debug(f'\t\tTransform 2D arrays to DataFrame with columns `{"`; `".join(φs_names + [f"{n}_qc" for n in μs.keys()] + [f"{n}_nfb" for n in ζb.keys()])}`.')
                logger.debug(f'\t\t{[np.array(v).shape for v in list(φs.values()) + list(μs.values()) + list(ζb.values())]}.')
                logger.debug(f'\t\t{timestamp.shape}.')
                __temp__ = hc24.matrixtotimetable(np.array(timestamp), #data.TIMESTAMP),
                                            np.concatenate(list(φs.values()) + list(μs.values()) + list(ζb.values()), axis=0),
                                            columns=φs_names + [f"{n}_qc" for n in μs.keys()] + [f"{n}_nfb" for n in ζb.keys()])
                logger.debug(f'\t\tMatrix to time table took {round(time.time() - info_t_startconditionalsampling)} s (run_wt).')
                
                ## END HERE FUNCTION RUN_COV 

                #__temp__ = __temp__[__temp__.TIMESTAMP >= min(yl)]
                #__temp__ = __temp__[__temp__.TIMESTAMP < max(yl)]
                
                #__temp__ = __temp__.melt('TIMESTAMP')
                #__temp__ = pd.concat([__temp__.pop('variable').str.extract("^(?P<variable>.+?)_?(?P<natural_frequency>(?<=_)\d+)?$", expand=True), __temp__], axis=1)
                #__temp__['natural_frequency'] = __temp__.variable.apply(lambda x: float(x.rsplit('_', 1)[-1]) if '_' in x and x[-1] in ['0','1','2','3','4','5','6','7','8','9'] else np.nan)
                #__temp__['natural_frequency'] = __temp__['natural_frequency'].apply(lambda j: 1/j2sj(j, 1/dt) if j else np.nan)
                #__temp__['variable'] = __temp__.variable.apply(lambda x: x.rsplit('_', 1)[0] if '_' in x and x[-1] in ['0','1','2','3','4','5','6','7','8','9'] else x)

                """
                def __arr2dataframe__(Y, qc=np.nan, prefix=''.join(xy), 
                                    id=np.array(data.TIMESTAMP), icolnames=sj):
                    if (len(Y.shape) == 1) or (Y.shape[1] == 1): colnames = [f"{prefix}"]
                    else: colnames = ["{}_{}".format(prefix, l) for l in icolnames] if icolnames is not None else None
                    __temp__ = hc24.matrixtotimetable(id, Y, columns=colnames)
                    if qc is not None: __temp__["{}_qc".format(prefix)] = qc
                    __temp__ = __temp__[__temp__.TIMESTAMP > min(yl)]
                    __temp__ = __temp__[__temp__.TIMESTAMP <= max(yl)]
                    return __temp__
                
                __temp__ = reduce(lambda left, right: pd.merge(left, right[['TIMESTAMP'] + list(right.columns.difference(left.columns))], 
                                                            on="TIMESTAMP", how="outer"),
                                [__arr2dataframe__(Y, μs[n], prefix=n) for n, Y in φs.items()] + 
                                [__arr2dataframe__(Y, None,  prefix=f'{n}_nfb') for n, Y in ζb.items()])
                #for k, v in ζb.items(): __temp__[f'{k}_nfb'] = v
                """
                
                info_t_startsaveloop = time.time()
                pattern = re.compile(r"^(?P<variable>.+?)_?(?P<natural_frequency>(?<=_)\d+)?$")
                arr_slice = np.unique(__temp__.TIMESTAMP.dt.floor(str(averaging)+'Min'), return_index=True)
                for __datea__, __tempa__ in list(zip(arr_slice[0], np.split(__temp__, arr_slice[1][1:]))):
                    #for __datea__, __tempa__ in __temp__.groupby(__temp__.TIMESTAMP.dt.floor(str(averaging)+'Min')):
                    dst_path = output_path.format(suffix + "_full_cospectra", pd.to_datetime(__datea__).strftime('%Y%m%d%H%M'))
                    if os.path.exists(dst_path): continue
                    use_header = False
                    if not os.path.exists(dst_path+'.part'):
                        use_header = True
                        header  = "wavelet_based_(co)spectra\n"
                        header += f"--------------------------------------------------------------\n"
                        header += f"TIMESTAMP_START = {min(__tempa__.TIMESTAMP)}\n"
                        header += f"TIMESTAMP_END = {max(__tempa__.TIMESTAMP)}\n"
                        header += f"N: {len(__tempa__.TIMESTAMP)}\n"
                        header += f"TIME_BUFFER [min] = {buffer/60}\n"
                        header += f"frequency [Hz]\n"
                        header += f"y-axis_->_wavelet_coefficient_*_\n"
                        header += f"mother_wavelet -> {method}\n"
                        header += f"acquisition_frequency [Hz] = {1/dt}\n"
                        header += f"averaging_interval [Min] = {averaging}\n"
                        hc24.mkdirs(dst_path)
                        with open(dst_path+'.part', 'w+') as part: part.write(header)
                        legitimate_to_write = 1
                        logger.debug(f'\t\tSaving header of DataFrame took {round(time.time() - info_t_startsaveloop)} ({round(time.time() - info_t_startdataframe)}) s (run_wt).')
                    
                    if not legitimate_to_write: continue
                    
                    __tempa__ = __tempa__.melt('TIMESTAMP')
                    __tempa__.drop('TIMESTAMP', axis=1, inplace=True)
                    __tempa__ = __tempa__.groupby(['variable'], dropna=False).agg(np.nanmean).reset_index(drop=False)
                    __tempa__ = pd.concat([__tempa__.pop('variable').str.extract(pattern, expand=True), __tempa__], axis=1)
                    __tempa__['natural_frequency'] = __tempa__['natural_frequency'].apply(lambda j: 1/j2sj(j, 1/dt) if j else np.nan)
                    with open(dst_path+'.part', 'a+', newline='') as part: __tempa__.to_file(part, header=use_header, chunksize=500, index=False)
                    
                    del __tempa__
                logger.debug(f'\t\tSaving DataFrame for all averaging period took {round(time.time() - info_t_startsaveloop)} ({round(time.time() - info_t_startdataframe)}) s (run_wt).')
                '''
                for a in avg_:
                    __tempa__ = copy.deepcopy(__temp__)
                    #__tempa__["TIMESTAMP"] = pd.to_datetime(np.array(__tempa__.TIMESTAMP)).floor(str(a)+'Min')
                    __tempa__["TIMESTAMP"] = __tempa__.TIMESTAMP.dt.floor(str(a)+'Min')
                    __tempa__ = __tempa__.groupby("TIMESTAMP").agg(np.nanmean).reset_index()

                    """
                    #maincols = ["TIMESTAMP", ''.join(xy)]
                    if φc:
                        #maincols += [names[0] + c + names[1] for c in ['++', '+-', '--', '-+']]
                        sign_label = {1: "+", -1: "-", 0: "·"}
                        for co in set(itertools.combinations([1, -1, 0]*len(names), len(names))):
                            #for c in ['++', '+-', '--', '-+']:
                            sign = ''.join([sign_label[c] for c in co])
                            #name = names[0] + sign[:2] + names[1] + ''.join([s + names[2+i]  for i, s in enumerate(sign[2:])])
                            name = ''.join([c for cs in zip(names, sign) for c in cs])
                            __tempa__.insert(1, name, np.sum(__tempa__[[
                                f"{name}_{l}" for l in sj if dt*2**l < integrating]], axis=1))
                            
                    __tempa__.insert(1, ''.join(xy)+'·', np.sum(__tempa__[[
                        "{}·_{}".format(''.join(xy), l) for l in sj if dt*2**l < integrating]], axis=1))
                    __tempa__.insert(1, ''.join(xy), np.sum(__tempa__[[
                        "{}_{}".format(''.join(xy), l) for l in sj if dt*2**l < integrating]], axis=1))
                    """
                    temporary_name = output_path.rsplit('/', 1)[0] + '/temp/' + output_path.rsplit('/', 1)[1].format(
                        suffix, str(date), str(a).zfill(2)).rsplit('.', 1)[0] + f'.{thisvar_i}.temporary'
                    hc24.mkdirs(temporary_name)
                    dat_fullspectra[a] += [temporary_name]
                    __tempa__.to_file(temporary_name, index=False)
                    #dat_fullspectra[a] += [__tempa__]
                    #dat_fluxresult[a] += [__tempa__[maincols]]
                    del __tempa__
                '''
                del __temp__
                
                    
                logger.debug(f'\t\tMoving from array to DataFrame took {round(time.time() - info_t_startdataframe)} s (run_wt).')
                info_t_dataframe += time.time() - info_t_startdataframe
                print('\x1B[1A\r', f"{date} {np.round(100 * thisvar_i / len(varstorun))}%" + " "*10, end='\n')
            
            logger.info(f'\tDecomposing all {info_c_varstorunloop} variables took {round(info_t_varstorunloop)} s ({round(info_t_varstorunloop/info_c_varstorunloop)} s/var) (run_wt).')
            logger.info(f'\tArray to DataFrame took {round(info_t_dataframe)} s ({round(info_t_dataframe/(thisvar_i+1))} s/loop) (run_wt).')
            logger.info(f'\tStarting averaging at {round(time.time() - info_t_start)} s (run_wt).')
            
            #data.TIMESTAMP
            arr_slice = np.unique(timestamp.dt.floor(str(averaging)+'Min'), return_index=True)
            for __datea__ in arr_slice[0]:
                #for __datea__, _ in data.groupby(data.TIMESTAMP.dt.floor(str(averaging)+'Min')):
                dst_path = output_path.format(suffix + "_full_cospectra", pd.to_datetime(__datea__).strftime('%Y%m%d%H%M'))
                if os.path.exists(dst_path+'.part'): os.rename(dst_path+'.part', dst_path)


            """
            for a in avg_:
                dat_fullspectra_file[a] = reduce(lambda left, right: pd.merge(left, right[['TIMESTAMP'] + list(right.columns.difference(left.columns))],
                                                                        on="TIMESTAMP", how="outer", suffixes=('', '_y')),
                                    [pd.read_file(d) for d in dat_fullspectra[a]])
                #dat_fluxresult[a] = reduce(lambda left, right: pd.merge(left, right[['TIMESTAMP'] + list(right.columns.difference(left.columns))],
                #                                                        on="TIMESTAMP", how="outer", suffixes=('', '_y')),
                #                     dat_fluxresult[a])
                
                hc24.mkdirs(output_path.format(suffix, str(date), str(a).zfill(2)))
                dat_fullspectra_file[a].to_csv(output_path.format(
                    suffix + "_full_cospectra", str(date), str(a).zfill(2)), index=False)
                #dat_fluxresult[a].to_csv(output_path.format(
                #    suffix, str(date), str(a).zfill(2)), index=False)
            """
        except Exception as e:
            logger.critical(str(e))
            print(str(e))
        
        #[os.remove(d) for a in avg_ for d in dat_fullspectra[a] if os.path.exists(d)]
        if os.path.exists(curoutpath_inprog): os.remove(curoutpath_inprog)
        prev_print = '\x1B[2A\r' + f' {date} {len(yl)} files {int(100*ymd_i/len(ymd))} % ({time.strftime("%d.%m.%y %H:%M:%S")})' + '\n'
            
        logger.info(f'\tFinish {date} took {round(time.time() - info_t_startdateloop)} s, yielded {len(yl)} files (run_wt). Progress: {len(yl)} {int(100*ymd_i/len(ymd))} %')
    
    logger.info(f'Finish wavelets at {round(time.time() - info_t_start)} s (run_wt).')
    return
