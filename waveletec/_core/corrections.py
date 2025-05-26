
# built-in modules

# 3rd party modules
import numpy as np
import pandas as pd
import logging
from scipy.optimize import curve_fit

# project modules
from .commons import j2sj

logger = logging.getLogger('corrections')


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


def __despike__(X, method=mauder2013):
    N = len(X)
    X = method(X)
    Xna = np.isnan(X)
    try:
        X = np.interp(np.linspace(0, 1, N), 
                        np.linspace(0, 1, N)[Xna == False],
                X[Xna==False])
    except Exception as e:
        logger.error(f"UserWarning: {str(e)}")
    return X


def __fit_whitenoise__(spec, freqs=None, fmax=5, a=1):
    """Fit a white noise spectrum to the data
    
    Parameters
    ----------
    spec : array
        The spectrum to fit
    fmax : int
        The maximum frequency to fit
    a : float
        The exponent of the spectrum ("color" of the noise)
    Returns
    -------
    fit : array
        The fitted spectrum
    """
    if freqs is None:
        dt = 20
        freqs = [1/j2sj(j, 1/dt) for j in np.arange(1, len(spec)+1)]
    curve_0 = lambda f, b: np.log((f**a)*b)
    specna = np.where(np.isnan(spec[:fmax]) | (
        spec[:fmax] <= 0), False, True)
    try:
        freqs = np.array(freqs)
        spec = np.array(spec)
        params_0, _ = curve_fit(curve_0, freqs[:fmax][specna], np.log(
            spec[:fmax][specna]), bounds=(0, np.inf))
    except Exception as e:
        logger.error(f"UserWarning: {str(e)}")
        params_0 = [0]
    fit = np.array([(f**a)*params_0[0] for f in freqs])
    return type('var_', (object,), {'curve_0': curve_0, 'params_0': params_0, 'fit': fit})


def smooth_2d_data(data, **kwargs):
    if data.shape[0] == 1:
        return smooth_data(data, **kwargs)
    else:
        for i in range(data.shape[0]):
            data[i] = smooth_data(data[i], **kwargs)
        return data


def smooth_data(data, method='convolve', smoothing=3, **kwargs):
    if smoothing == 0:
        return data
    if method == 'convolve':
        return np.convolve(data, np.ones((smoothing,)) /
                           smoothing, mode='same')
    # if method == 'gaussian':
    #     return gaussian_filter1d(data, sigma=smoothing, **kwargs)
    # if method == 'savgol':
    #     return savgol_filter(data, window_length=smoothing, polyorder=2, **kwargs)
    if method == 'moving':
        stat = kwargs.get('stat', 'mean')
        if stat == 'mean':
            return data.rolling(smoothing, min_periods=1).mean()
        if stat == 'std':
            return data.rolling(smoothing, min_periods=1).std()
        if stat == 'max':
            return data.rolling(smoothing, min_periods=1).max()
        if stat == 'quantile':
            return data.rolling(smoothing, min_periods=1).quantile(0.5)
    if method == 'repeat':
        # Create array of indices to group by
        chunk_indices = np.floor(np.arange(len(data)) / smoothing).astype(int)
        # Group by those indices and apply the function
        return pd.Series(data).groupby(
            chunk_indices).transform(np.nanmean)

    return data
    for i in range(len(φcs)):
        φcs[i] = np.convolve(φcs[i], np.ones(
            (smoothing,))/smoothing, mode='same')
