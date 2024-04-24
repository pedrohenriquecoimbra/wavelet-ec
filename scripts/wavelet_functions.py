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

# standard modules
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

# Project modules
from . import coimbra2024_scripts as hc24
j2sj = hc24.j2sj

logger = logging.getLogger('wvlt')

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

   
def __dwt__(*args, level=None, wave="db6"):
    """
    function: performs Discrete Wavelet Transform
    call: __dwt__()
    Input:
        *args: arrays (1D) to be transformed
        level: maximum scale (power of 2)
        wave: mother wavelet (comprehensible to pywt)
    Return:
        Ws: list of 2D arrays
    """
    Ws = []
    for X in args:
        Ws += [pywt.wavedec(X, wave, level=level)]
    level = len(Ws[-1])-1
    return Ws


def __idwt__(*args, N, level=None, wave="db6"):
    """
    function: performs Inverse Discrete Wavelet Transform
    call: __idwt__()
    Input:
        *args: 2D arrays contianing wavelet coefficient
        N: data lenght
        level: maximum scale (power of 2)
        wave: mother wavelet (comprehensible to pywt)
    Return:
        Ws: list of 2D arrays
        level: maximum scale (power of 2)
    """
    #assert sum([s==level for s in W.shape]), "Coefficients don't have the same size as level."
    def wrcoef(N, coef_type, coeffs, wavename, level):
        a, ds = coeffs[0], list(reversed(coeffs[1:]))

        if coef_type == 'a':
            return pywt.upcoef('a', a, wavename, level=level, take=N)  # [:N]
        elif coef_type == 'd':
            return pywt.upcoef('d', ds[level-1], wavename, level=level, take=N)  # [:N]
        else:
            raise ValueError("Invalid coefficient type: {}".format(coef_type))
    
    Ys = []
    for W in args:
        A1 = wrcoef(N, 'a', W, wave, level)
        D1 = [wrcoef(N, 'd', W, wave, i) for i in range(1, level+1)]
        Ys += [np.array(D1 + [A1])]
    return Ys, level


def universal_wt(signal, method, fs=20, f0=1/(3*60*60), f1=10, fn=100, 
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
        if callable(fcwt):
            """Run Continuous Wavelet Transform, using fast algorithm"""
            _l, wave = __cwt__(signal, fs, f0, f1, fn, **kwargs)
            sj = np.log2(fs/_l)
            if inv:
                wave = __icwt__(wave, sj=sj, dt=fs, dj=dj, **kwargs, 
                            wavelet=pycwt.wavelet.Morlet(6))
        else:
            logger.warning('UserWarning: Fast continuous wavelet transform (fcwt) not found. Running slow version.')
            method = 'cwt'
    
    elif method == 'cwt':
        if callable(pycwt):
            """Run Continuous Wavelet Transform"""
            wave, sj, _, _, _, _ = pycwt.cwt(
                signal, dt=1/fs, s0=2/fs, dj=dj, J=fn-1)
            sj = np.log2(sj*fs)
            if inv:
                wave = __icwt__(wave, sj=sj, dt=fs**-1, dj=dj, **kwargs)
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
            waves = __idwt__(*waves, N=N, level=lvl, **kwargs)
        wave = waves[0][0]
    return wave, sj


def conditional_sampling(Y12, *args, names=['xy', 'a'], label = {1: "+", -1: "-", 0: "·"}, false=0):
    # guarantee names are enough to name all arguments
    nargs = len(args)# + 1
    if nargs < len(names): names = names[:nargs]
    if nargs > len(names): names = names + ['b']* (nargs-len(names))
    # [Y12] + list(args) 
    YS = list(args)
    Ys = {}

    # run for all unique combinations of + and - for groups of size n
    # (e.g., n=2: ++, +-, -+, --, n=3 : +++, ++-, ...)
    for co in set(itertools.combinations(list(label.keys())*nargs, nargs)):
        sign = ''.join([label[c] for c in co])
        name = ''.join([c for cs in zip(names, sign) for c in cs])
        Ys[name] = Y12
        # condition by sign
        for i, c in enumerate(co):
            if c: xy = 1 * (c*YS[i] > 0)
            else: xy = 1 * (YS[i] == 0)
            #xy[xy==0] = false
            xy = np.where(xy == 0, false, xy)
            Ys[name] = Ys[name] * xy
    return Ys


def split_sampling(Y12, *args, names=['xy', 'a'], false=0):
    return

def integrate_cospectra(root, pattern, f0, dst_path):
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
    data0 = data[(np.isnan(data['natural_frequency'])==False) * (data['natural_frequency'] >= f0)].groupby(['variable', 'TIMESTAMP'])['value'].agg(np.nansum).reset_index(drop=False)
    data1 = data[np.isnan(data['natural_frequency'])].drop('natural_frequency', axis=1)

    datai = pd.concat([data1, data0[np.isin(data0['variable'], data1['variable'].unique())==False]]).drop_duplicates()
    datai = datai.pivot_table('value', 'TIMESTAMP', 'variable').reset_index(drop=False)
    
    if dst_path: datai.to_file(dst_path, index=False)
    else: return datai
    return

    
def run_wt(ymd, varstorun, raw_kwargs, output_path, wt_kwargs={}, 
           method="dwt", Cφ=1, nan_tolerance=.3,
           averaging=30, condsamp_flat=[], integrating=30*60, 
           overwrite=False, saveraw=False, file_duration="1D",
           denoise=False, noisecolor=1, deadband={}, verbosity=1):
    """
    file_duration -> processing_time_duration
    averaging & integrating at the same unit

    fs = 20, f0 = 1/(3*60*60), f1 = 10, fn = 100, agg_avg = 1, 
    suffix = "", mother = pycwt.wavelet.MexicanHat(),
    **kwargs):
    """
    if isinstance(averaging, (list, tuple)): averaging = averaging[-1]

    info_t_start = time.time()
    logger.info('Entered wavelet code (run_wt).')

    assert method in [
        'dwt', 'cwt', 'fcwt'], "Method not found. Available methods are: dwt, cwt, fcwt"
    if verbosity: print(f'\nRUNNING WAVELET TRASNFORM ({method})')
    if method in ["cwt", "fcwt"]:
        if method == "fcwt" or "mother" not in wt_kwargs.keys() or wt_kwargs.get("mother") in ['morlet', 'Morlet', pycwt.wavelet.Morlet(6)]:
            Cφ = 5.271
        else:
            Cφ = 16.568
    
    dt = 1 / wt_kwargs.get("fs", 20)
    suffix = raw_kwargs['suffix'] if 'suffix' in raw_kwargs.keys() else ''

    _, _, _f = ymd
    ymd = hc24.list_time_in_period(*ymd, file_duration, include='both')
    
    if method in ['dwt']:
        buffer = hc24.bufferforfrequency_dwt(
            N=pd.to_timedelta(file_duration)/pd.to_timedelta("1S") * dt**-1,
            n_=_f, **wt_kwargs)/2
    else:
        buffer = hc24.bufferforfrequency(wt_kwargs.get("f0", 1/(3*60*60))) / 2
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
        if file_duration.endswith("D"): date = date[:8]
        if file_duration.endswith("H") or file_duration.endswith("Min"): date = date[:12]
        
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
        
        # ensure time is time
        data.TIMESTAMP = pd.to_datetime(data.TIMESTAMP)
        
        # ensure continuity
        info_t_startdatacontinuity = time.time()
        data = pd.merge(pd.DataFrame({"TIMESTAMP": pd.date_range(np.nanmin(data.TIMESTAMP), np.nanmax(data.TIMESTAMP), freq=f"{dt}S")}),
                            data, on="TIMESTAMP", how='outer').reset_index(drop=True)
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
                        _isinfinalrange = (data.TIMESTAMP >= min(yl)) & (data.TIMESTAMP < max(yl))
                        signal = signal[_isinfinalrange.to_list()]
                        signan = signan[_isinfinalrange.to_list()]
                        N = len(signal)
                        φ[v] = φ[v][:, _isinfinalrange.to_list()]
                        φ0[v] = copy.deepcopy(φ[v])
                        S[v] = signal

                        # apply despiking (Mauder et al.)
                        φ0[v] = np.apply_along_axis(__despike__, 1, φ0[v])
                        μ[v] = signan *1
                        μ[v+'D'] = signan *1
                        μ[v+'T'] = signan *1
                        
                        """
                        # apply Allan deviation (ad-hoc)
                        allan_dev = {'co2': np.array([0.014, 0.014, 0.014, 0.014, 0.013, 0.01,  0.007, 0.005, 0.004, 0.004, 0.003, 0.002, 0.002, 0.001, 0.002, 0.003, 0.003, np.inf]),
                                     'co': np.array([3.095, 3.095, 3.095, 3.095, 2.853, 2.206, 1.393, 0.998, 0.674, 0.544, 0.418, 0.265, 0.168, 0.124, 0.103, 0.067, 0.086, np.inf]),
                                     'ch4': np.array([0.089, 0.089, 0.089, 0.089, 0.083, 0.065, 0.043, 0.034, 0.027, 0.023, 0.018, 0.013, 0.01,  0.011, 0.018, 0.027, 0.041, np.inf])}
                        allan_dev = {k: np.expand_dims(v, 1) for k, v in allan_dev.items()}
                        if (N > 0) and (v.lower() in allan_dev.keys()):
                            # φ[v] needs to be (n, 17)
                            φ0[v] = np.where((abs(φ[v]) < np.sqrt(allan_dev[v])), 0, φ0[v])                        
                        """

                        # for file length, update later
                        σ[v]  = []
                        σ[v+'D'] = []
                        σ[v+'T'] = []
                        #ζ[v]  = []
                        ζb[v] = []

                        for this_group, this_data in data.groupby(data[_isinfinalrange].TIMESTAMP.dt.floor(_f)):
                            # Everytime creating working with the whole array, maybe faster to work on a split of the array (redability)
                            # List of T/F to select values that are in the current AVERAGING period
                            this_hh = np.isin(data[_isinfinalrange].index, this_data.index)#np.where(data.TIMESTAMP > this_yl) & (data.TIMESTAMP <= (this_yl + pd.Timedelta(_f))), True, False)
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
                        logger.debug(f'\t\tDecompose all variables took {σ[v].shape} / {σ[v+"D"].shape} s (run_wt).')
                        #ζ[v] = np.array(ζ[v]).T
                        ζb[v] = np.array(ζb[v]).reshape(1,-1)
                        S[v] = np.array(S[v]).reshape(1,-1)
                        #μ[v] = np.array(μ[v]).reshape(1,-1)
                logger.debug(f'\t\tDecompose all variables took {round(time.time() - info_t_startvarloop)} s (run_wt).')
                info_t_varstorunloop += time.time() - info_t_startvarloop

                info_t_startcovariance = time.time()
                # Strip definitively buffer data
                data = data[_isinfinalrange]

                # calculate covariance
                for ci, c in enumerate(xy): Y12 = Y12 * φ[c].conjugate() if ci else φ[c] * Cφ
                for ci, c in enumerate(xy): Y120 = Y120 * φ0[c].conjugate() if ci else φ0[c] * Cφ
                for ci, c in enumerate(xy): Cov = Cov * σ[c] if ci else copy.deepcopy(σ[c])
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
                φs.update({f'{k}_var': (v**2).reshape(1,-1) for k, v in σ.items()})
                μs = {''.join(xy): np.where(np.where(
                    np.array(μ[xy[0]]), 0, 1) * np.where(np.array(μ[xy[1]]), 0, 1), 0, 1).reshape(1,-1)}
                μs.update({'/'.join(xy)+'_cov': μs[''.join(xy)]})
                μs.update({f'{k}_var': np.where(np.array(μ[k]), 0, 1).reshape(1,-1) for k in σ.keys()})
                
                logger.debug(f'\t\tCalculate covariances took {round(time.time() - info_t_startcovariance)} s (run_wt).')
                
                info_t_startconditionalsampling = time.time()
                # conditional sampling
                names = [''.join(xy)] + [''.join(cs) for cs in condsamp_pair]
                φcs = []
                if condsamp_pair: φcs += [Y120]#[Y12ζ]
                for cs in condsamp_pair:
                    for ci, c in enumerate(cs): φc0 = φc0 * φ0[c].conjugate() if ci else φ0[c]
                    # if variables to condition are uncertain do not consider them
                    # and put them in uncertain bag (Y12ζ) 
                    #for ci, c in enumerate(cs): φc0  = np.where(abs(φ[c]) > abs(ζ[c]), φc0, 0)
                    #for ci, c in enumerate(cs): Y12ζ = np.where(abs(φ[c]) > abs(ζ[c]), Y12ζ, 0)
                    φcs += [φc0]
                
                """
                names = [''.join(xy)] + [xy[0]+c for c in condsamp_flat]
                φc = [np.array(φ[xy[0]]) * np.array(φ[c]).conjugate() if xy[0] != c else np.array(φ[c]) for c in condsamp_flat]
                """
                #Y12ζ
                φc = conditional_sampling(Y120, *φcs, names=names) if φcs else {}
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
                φs.update({k.replace("xy", ''.join(xy)).replace('a', ''.join(condsamp_flat)): v for k, v in φc.items()})
                φs.update(φc0)

                logger.debug(f'\t\tConditional sampling took {round(time.time() - info_t_startconditionalsampling)} s (run_wt).')
                
                # add noise columns
                φs.update({''.join(xy) + '·': ζY12})

                # repeats nan flag wo/ considering conditional sampling variables
                μs.update(
                    {k: μs[k if k in μs.keys() else [k_ for k_ in μs.keys() if k.startswith(k_)][0]] for k in φs.keys()})

                info_t_startdataframe = time.time()
                # array to dataframe for averaging
                
                # Integrate arrays are 2D inside dictionary (needed for concatenate)
                def __integrate_decomposedarrays_in_dictionary__(array_dic):
                    levels_to_integrate = [dt*2**l < integrating for l in sj]
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
                __temp__ = hc24.matrixtotimetable(np.array(data.TIMESTAMP),
                                            np.concatenate(list(φs.values()) + list(μs.values()) + list(ζb.values()), axis=0),
                                            columns=φs_names + [f"{n}_qc" for n in μs.keys()] + [f"{n}_nfb" for n in ζb.keys()])
                logger.debug(f'\t\tMatrix to time table took {round(time.time() - info_t_startconditionalsampling)} s (run_wt).')
                
                #__temp__ = __temp__[__temp__.TIMESTAMP >= min(yl)]
                #__temp__ = __temp__[__temp__.TIMESTAMP < max(yl)]
                
                __temp__ = __temp__.melt('TIMESTAMP')
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
                for __datea__, __tempa__ in __temp__.groupby(__temp__.TIMESTAMP.dt.floor(str(averaging)+'Min')):
                    dst_path = output_path.format(suffix + "_full_cospectra", __datea__.strftime('%Y%m%d%H%M'))
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
                    
                    if not legitimate_to_write: continue
                    
                    __tempa__.drop('TIMESTAMP', axis=1, inplace=True)
                    __tempa__ = __tempa__.groupby(['variable'], dropna=False).agg(np.nanmean).reset_index(drop=False)
                    __tempa__ = pd.concat([__tempa__.pop('variable').str.extract(pattern, expand=True), __tempa__], axis=1)
                    __tempa__['natural_frequency'] = __tempa__['natural_frequency'].apply(lambda j: 1/j2sj(j, 1/dt) if j else np.nan)
                    with open(dst_path+'.part', 'a+', newline='') as part: __tempa__.to_file(part, header=use_header, index=False)
                    
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
            
            for __datea__, _ in data.groupby(data.TIMESTAMP.dt.floor(str(averaging)+'Min')):
                dst_path = output_path.format(suffix + "_full_cospectra", __datea__.strftime('%Y%m%d%H%M'))
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
        prev_print = '\x1B[2A\r' + f' {date} {len(yl)} files {int(100*ymd_i/len(ymd))} %' + '\n'
            
        logger.info(f'\tFinish {date} took {round(time.time() - info_t_startdateloop)} s, yielded {len(yl)} files (run_wt). Progress: {len(yl)} {int(100*ymd_i/len(ymd))} %')
    
    logger.info(f'Finish wavelets at {round(time.time() - info_t_start)} s (run_wt).')
    return
