
# built-in modules

# 3rd party modules
import numpy as np
import logging

# project modules

logger = logging.getLogger('corrections')


##########################################
###     SPECTRA                            
##########################################


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