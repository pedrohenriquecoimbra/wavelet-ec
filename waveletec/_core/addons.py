# built-in modules
import re
import os
import sys
import warnings
import logging
from contextlib import contextmanager

# 3rd party modules
import pandas as pd

logger = logging.getLogger('wvlt.addons')

# Add-ons
def _warning(
    message,
    category = UserWarning,
    filename = '',
    lineno = -1,
    file = None, 
    line = None):
    print("%s: %s" % (category.__name__, message))
warnings.showwarning = _warning

import matplotlib.pyplot as plt
# Reads styles in /styles
stylesheets = plt.style.core.read_style_directory(os.path.join(os.getcwd(), 'style'))
# Update dictionary of styles
plt.style.core.update_nested_dict(plt.style.library, stylesheets)
plt.style.core.available[:] = sorted(plt.style.library.keys())

import pandas as pd
from pandas.api.types import is_numeric_dtype, is_object_dtype
pd.DataFrame.columnstartswith = lambda self, x: [c for c in self.columns if c.startswith(x)]
pd.DataFrame.columnsmatch = lambda self, x: [c for c in self.columns if re.findall(x, c)]
def columnsconditioned(self, start, *args):
    columns = self.columnsmatch(f'^{start}[^_]+$')
    if args:
        for a in args:
            for c in [c for c in columns]:
                if not re.findall(a, c):
                    columns.pop(columns.index(c))
    
    return columns
pd.DataFrame.columnsconditioned = columnsconditioned
def df_to_file(self, file_name, *a, **k): 
    to_functions = {'csv': pd.DataFrame.to_csv,
                    'xlsx': pd.DataFrame.to_excel,
                    'txt': pd.DataFrame.to_csv,
                    'parquet': pd.DataFrame.to_parquet,
                    'temporary': pd.DataFrame.to_parquet,
                    'json': pd.DataFrame.to_json}
    
    file_str = file_name if isinstance(file_name, str) else file_name.name
    if file_str.endswith('.part'):
        file_str = file_str.replace('.part', '')

    for file_ext, to in to_functions.items():
        if file_str.endswith(file_ext):
            return to(self, file_name, *a, **k)
        
    logger.warning(f"File extension not recognized: {file_str}")
    return None
pd.DataFrame.to_file = df_to_file
def pd_read_file(file_name, *a, **k):
    read_functions = {'csv': pd.read_csv,
                    'xlsx': pd.read_excel,
                    'txt': pd.read_csv,
                    'parquet': pd.read_parquet,
                    'temporary': pd.read_parquet,
                    'json': pd.read_json}
    for file_ext, read in read_functions.items():
        if file_name.endswith(file_ext):
            return read(file_name, *a, **k)         
    return None
pd.read_file = pd_read_file

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout