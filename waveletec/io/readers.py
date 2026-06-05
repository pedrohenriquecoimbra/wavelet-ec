
# built-in modules

# 3rd party modules
import pandas as pd

# local modules
import regorator as registry


READERS = registry.Register()


def _read_timestamped_csv(path, **kwargs):
    """Read a CSV with ``TIMESTAMP_START``/``TIMESTAMP_END`` columns into a Dataset.

    Args:
        path: Path to the CSV file.
        **kwargs: Extra keyword arguments forwarded to :func:`pandas.read_csv`
            (override the ``na_values`` default).

    Returns:
        xarray.Dataset indexed by ``TIMESTAMP`` (set to ``TIMESTAMP_END``) with
        ``TIMESTAMP_START``/``TIMESTAMP_END`` as coordinates.
    """
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


@READERS.register('raw')
def read_raw(path, **kwargs):
    """Reader for ``raw`` files (see :func:`_read_timestamped_csv`)."""
    return _read_timestamped_csv(path, **kwargs)


@READERS.register('fluxnet')
def read_fluxnet(path, **kwargs):
    """Reader for ``fluxnet`` files (see :func:`_read_timestamped_csv`)."""
    return _read_timestamped_csv(path, **kwargs)
