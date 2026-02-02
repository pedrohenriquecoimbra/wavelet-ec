
# built-in modules
import os
import re
import warnings
import logging
import copy
import time
import datetime
import glob
import zipfile
from io import StringIO

# 3rd party modules
from functools import reduce
import numpy as np
import pandas as pd
import xarray as xr

# local modules
from ..core import registry


READERS = registry.Register()


@READERS.register('raw')
def read_raw(path, **kwargs):
    kw = dict(na_values=[-9999, 'NAN'])
    kw.update(kwargs)
    df = pd.read_csv(path, **kw)

    df["TIMESTAMP_START"] = pd.to_datetime(
        df["TIMESTAMP_START"], format='%Y%m%d%H%M')
    df["TIMESTAMP_END"] = pd.to_datetime(
        df["TIMESTAMP_END"], format='%Y%m%d%H%M')
    df["TIMESTAMP"] = df["TIMESTAMP_END"]

    ds = df.set_index('TIMESTAMP').to_xarray()
    ds = ds.set_coords(['TIMESTAMP_START', 'TIMESTAMP_END'])
    return ds


@READERS.register('fluxnet')
def read_fluxnet(path, **kwargs):
    kw = dict(na_values=[-9999, 'NAN'])
    kw.update(kwargs)
    df = pd.read_csv(path, **kw)

    df["TIMESTAMP_START"] = pd.to_datetime(
        df["TIMESTAMP_START"], format='%Y%m%d%H%M')
    df["TIMESTAMP_END"] = pd.to_datetime(
        df["TIMESTAMP_END"], format='%Y%m%d%H%M')
    df["TIMESTAMP"] = df["TIMESTAMP_END"]

    ds = df.set_index('TIMESTAMP').to_xarray()
    ds = ds.set_coords(['TIMESTAMP_START', 'TIMESTAMP_END'])
    return ds
