# built-in modules
import re
import warnings

# 3rd party modules
import numpy as np
import pandas as pd

##########################################
###     WAVELET-RELATED                            
##########################################

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
