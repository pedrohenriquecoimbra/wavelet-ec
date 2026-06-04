# built-in modules
import re
import os
import sys
import warnings
import logging
from contextlib import contextmanager

# 3rd party modules
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "configure_warnings",
    "patch_pandas",
    "read_file",
    "to_file",
    "columnstartswith",
    "columnsmatch",
    "columnsconditioned",
    "suppress_stdout",
]


# Add-ons
def _warning(
    message,
    category = UserWarning,
    filename = '',
    lineno = -1,
    file = None, 
    line = None):
    logger.warning("%s: %s", category.__name__, message)


def configure_warnings():
    """Route Python warnings through this module's logger.

    Not applied on import; call explicitly (e.g. from an application entry
    point) if you want ``warnings.showwarning`` overridden.
    """
    warnings.showwarning = _warning


# --- DataFrame helpers (module-level; opt into pandas methods via patch_pandas) ---

def columnstartswith(df, prefix):
    """Return columns of *df* whose names start with *prefix*."""
    return [c for c in df.columns if c.startswith(prefix)]


def columnsmatch(df, pattern):
    """Return columns of *df* whose names match the regex *pattern*."""
    return [c for c in df.columns if re.findall(pattern, c)]


def columnsconditioned(df, start, *patterns):
    """Return columns matching ``^{start}[^_]+$`` and every regex in *patterns*."""
    columns = columnsmatch(df, f'^{start}[^_]+$')
    for pattern in patterns:
        columns = [c for c in columns if re.findall(pattern, c)]
    return columns


_TO_WRITERS = {'csv': pd.DataFrame.to_csv,
               'xlsx': pd.DataFrame.to_excel,
               'txt': pd.DataFrame.to_csv,
               'parquet': pd.DataFrame.to_parquet,
               'temporary': pd.DataFrame.to_parquet,
               'json': pd.DataFrame.to_json}
_READERS = {'csv': pd.read_csv,
            'xlsx': pd.read_excel,
            'txt': pd.read_csv,
            'parquet': pd.read_parquet,
            'temporary': pd.read_parquet,
            'json': pd.read_json}


def to_file(df, file_name, *args, **kwargs):
    """Write *df* to *file_name*, dispatching on the file-name extension.

    Returns the writer's result, or ``None`` (with a warning) if the extension
    is unrecognized. A trailing ``.part`` suffix is ignored when matching.
    """
    file_str = file_name if isinstance(file_name, str) else file_name.name
    if file_str.endswith('.part'):
        file_str = file_str.replace('.part', '')
    for ext, writer in _TO_WRITERS.items():
        if file_str.endswith(ext):
            return writer(df, file_name, *args, **kwargs)
    logger.warning("File extension not recognized: %s", file_str)
    return None


def read_file(file_name, *args, **kwargs):
    """Read *file_name* into a DataFrame, dispatching on the file-name extension.

    Returns ``None`` if the extension is unrecognized.
    """
    for ext, reader in _READERS.items():
        if file_name.endswith(ext):
            return reader(file_name, *args, **kwargs)
    return None


def patch_pandas():
    """Attach the helpers above onto pandas as methods/functions.

    Opt-in convenience, **not** applied on import. After calling this you can
    use ``df.to_file(...)``, ``pd.read_file(...)``, ``df.columnsmatch(...)``,
    etc. Library code should prefer the module-level functions directly.
    """
    pd.DataFrame.columnstartswith = lambda self, prefix: columnstartswith(self, prefix)
    pd.DataFrame.columnsmatch = lambda self, pattern: columnsmatch(self, pattern)
    pd.DataFrame.columnsconditioned = lambda self, start, *patterns: columnsconditioned(self, start, *patterns)
    pd.DataFrame.to_file = lambda self, *args, **kwargs: to_file(self, *args, **kwargs)
    pd.read_file = read_file

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout