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

logger = logging.getLogger(__name__)


def __wavemother_str_pycwt__(name):
    wavelets = {w.lower(): vars(pycwt.mothers)[w] for w in ['Morlet', 'Paul', 'DOG', 'MexicanHat']}
    mother = wavelets[re.subn('[0-9]', '',  name.lower())[0]]
    if re.findall('[0-9]+', name.lower()): 
        mother = mother(int(re.findall('[0-9]+', name.lower())[0]))
    else:
        mother = mother()
    return mother


def bufferforfrequency_dwt(N=0, n_=None, fs=20, level=None, f0=None, max_iteration=10**4, wavelet='db6'):
    try:
        import pywt
    except Exception as e:
        logger.error(f"Error in bufferforfrequency_dwt:\n{e}")
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


def bufferforfrequency(f0, dt=0.05, param=6, mother="MORLET", wavelet=None):
    try:
        import pywt
    except Exception as e:
        logger.error(f"Error in bufferforfrequency:\n{e}")
    wavelet = wavelet or pycwt.Morlet(6)
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


def __fcwt__(input, fs, f0, f1, fn, nthreads=1, scaling="log", fast=False, norm=True, Morlet=6.0):
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

   
def __dwt__(signal, level=None, wavelet="db6", **kwargs):
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
    return pywt.wavedec(signal, wavelet, level=level, **kwargs)


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


def __idwt__(coef, N, wavelet='db6', mode='symmetric'):
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
    reconstructed_signal = waverec_2d(coef, N, wavelet, mode=mode)
    reconstructed_signal = np.array(reconstructed_signal[::-1])
    return reconstructed_signal


def prepare_signal(signal, nan_tolerance=0.3):
    signal = np.array(signal)
    signan = np.isnan(signal)
    N = signal.size
    Nnan = np.sum(signan)
    if Nnan:
        if (nan_tolerance > 1 and Nnan > nan_tolerance) or (Nnan > nan_tolerance * N):
            logger.warning(
                f"UserWarning: Too much nans ({np.sum(signan)}, {np.round(100*np.sum(signan)/len(signal), 1)}%).")
    if Nnan and Nnan < N:
        signal = np.interp(np.linspace(0, 1, N),
                            np.linspace(0, 1, N)[
            signan == False],
            signal[signan == False])
    return type('var_', (object,), {'signal': signal, 'signan': signan, 'N': N, 'Nnan': Nnan})


def cone_of_influence(n0, dt, wavelet=None):
    """ function: calculates the cone of influence.
    Uses triangualr Bartlett window with non-zero end-points.
    call: cone_of_influence()
    Input:
        n0: number of points in the signal
        dt: sampling rate (in seconds)
        wavelet: mother wavelet (w/ cdelta and psi(0) callables). If None, uses Morlet.
    Return:
        coi: array with the cone of influence in Fourier periods    
    """

    if wavelet is None: wavelet = pycwt.wavelet.Morlet(6)
    if isinstance(wavelet, str): wavelet = __wavemother_str_pycwt__(wavelet)

    coi = (n0 / 2 - np.abs(np.arange(0, n0) - (n0 - 1) / 2))
    coi = wavelet.flambda() * wavelet.coi() * dt * coi    
    return coi


def cone_of_influence_mask(sj, **kwargs):
    """
    function: calculates the cone of influence for a given wavelet transform
    call: cone_of_influence_mask()
    Input:
        sj: scales
        kwargs: keyword arguments for wavelet transform (e.g., dt, wavelet)
    Return:
        coi: array with True if inside cone of influence, False otherwise
    """
    
    coi = np.array([[s >= (1 / c)
                   for c in cone_of_influence(**kwargs)] for s in sj])

    return coi


def universal_wt(signal, method='dwt', fs=20, f0=1/(3*60*60), f1=10, fn=180, 
                 dj=1/12, wt=True, iwt=True, coi=True, **kwargs):
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
        iwt: . Default is True
        **kwargs: keyword arguments sent to wavelet transform and inverse functions 
    Return:
        wave: 2D array
        sj: scales 
    """
    assert method in [
        'dwt', 'cwt', 'fcwt'], "Method not found. Available methods are: dwt, cwt, fcwt"

    if method == 'fcwt':
        if fcwt is not None:
            pass
        else:
            logger.warning(
                'UserWarning: Fast continuous wavelet transform (fcwt) not found. Running slow version.')
            method = 'cwt'

    if method == 'cwt':
        if pycwt is not None:
            pass
        else:
            logger.warning(
                'UserWarning: Continuous wavelet transform (cwt) not found. Running discrete version.')
            method = 'dwt'

    if wt:
        # Wavelet Transform
        if method == 'fcwt':
            """Run Continuous Wavelet Transform, using fast algorithm"""
            _l, wave = __fcwt__(signal, fs, f0, f1, fn, **kwargs)
            sj = np.log2(fs/_l)
            sj = list(sj)

        elif method == 'cwt':
            """Run Continuous Wavelet Transform"""
            wave, sj, _, _, _, _ = pycwt.cwt(
                signal, dt=1/fs, s0=2/fs, dj=dj, J=fn-1, **kwargs)
            sj = np.log2(sj*fs)
            sj = list(sj)

        elif method == "dwt":
            """Run Discrete Wavelet Transform"""
            lvl = kwargs.pop('level', int(np.ceil(np.log2(fs/f0))))
            # _l if s0*2^j; fs*2**(-_l) if Hz; (1/fs)*2**_l if sec.
            sj = [(fs*(2**-float(_l))) for _l in np.arange(1, lvl+2, 1)]
            wave = __dwt__(signal, level=lvl, **kwargs)
    else:
        wave = signal

    if iwt:
        # Inverse Wavelet Transform
        if method == 'fcwt':
            sj = kwargs.pop('sj', None)
            wave = __icwt__(wave, sj=sj, dt=fs, dj=dj, **kwargs, 
                        wavelet=pycwt.wavelet.Morlet(6))
        
        elif method == 'cwt':
            sj = kwargs.pop('sj', None)
            wave = __icwt__(wave, sj=sj, dt=fs**-1, dj=dj, **kwargs)
        
        elif method== "dwt":
            N = np.array(signal).shape[-1]
            wave = __idwt__(wave, N=N, **kwargs).real

    if coi:
        # Cone of Influence (COI)
        if method == "dwt":
            coneoi = np.ones_like(wave)
        else:
            coneoi = cone_of_influence_mask(sj=sj, dt=1/fs,
                                        n0=len(signal), 
                                        #wavelet=kwargs.get('mother_wavelet', '')
                                        )
    else:
        coneoi = None
    
    return type('var_', (object,), {
        'wave': wave, 'sj': sj, 'coi': coneoi, 'method': method, 'fs': fs, 'f0': f0, 'f1': f1, 'fn': fn})
