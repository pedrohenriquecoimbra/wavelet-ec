
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
import itertools
from scipy.optimize import curve_fit
import pywt
import yaml


def __universal_reader__(path, reader, **kwargs):
    reader(path, **kwargs)

    return
