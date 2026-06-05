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
import logging
import warnings
import time
import datetime
import glob

# 3rd party modules
import numpy as np
import xarray as xr

# project modules
from . import commons
from ..io import READERS
# from .read_data import loaddatawithbuffer
from .wavelet_functions import universal_wt, formula_to_vars, prepare_signal
from ..extra.partitioning.coimbra_et_al_2025 import conditional_sampling, partition_DWCS, partition_DWCS_CO, partition_DWCS_H2O
from ..extra import eddypro as eddypro


logger = logging.getLogger(__name__)

__all__ = [
    # decomposition & covariance on xarray.Dataset
    "decompose_variables",
    "data_statistics",
    "data_compute_product",
    "data_conditional_sampling",
    "data_partition",
    "data_average_dims",
    "data_integrate_in_frequency",
    "open_files_in_folder",
    # run entry points
    "process",
    "data_run",
    "run_from_eddypro",
    # re-exported analysis API (kept public for backwards compatibility)
    "universal_wt",
    "formula_to_vars",
    "prepare_signal",
    "conditional_sampling",
    "partition_DWCS",
    "partition_DWCS_CO",
    "partition_DWCS_H2O",
]


def decompose_variables(data, variables=None,
                        nan_tolerance=.3, **kwargs):
    """
    Calculate data decomposed with wavelet transform for xarray.Dataset.

    Parameters:
    - data: xarray.Dataset
    - variables: list of variable names to decompose (defaults to ['w', 'co2'])
    - nan_tolerance: tolerance for NaN values
    - **kwargs: additional arguments for universal_wt

    Returns:
    - xarray.Dataset with decomposed variables
    """
    if variables is None:
        variables = ['w', 'co2']
    # Initialize output dataset
    result = xr.Dataset()

    # Add the original coordinates
    result = result.assign_coords(data.coords)

    # # Placeholder for wavelet scales and COI
    # sj = None
    # coi = None

    try:
        for var in variables:
            if var not in data.data_vars:
                raise ValueError(f"Variable {var} not found in dataset.")
            
            # Prepare signal (assuming prepare_signal is adapted for xarray)
            ready_signal = prepare_signal(
                data[var], nan_tolerance=nan_tolerance)
            
            # Perform wavelet transform
            wt_signal = universal_wt(
                signal=ready_signal.signal, **kwargs, iwt=True)
            
            # Store results
            wt_signal = wt_signal.rename(
                {'wave': var,
                 'approximation': f'{var}_lf',
                 'coi': f'{var}_qc'}
            )
            wt_signal = wt_signal.drop_vars('signal')
            wt_signal[f'{var}_qc'] = wt_signal[f'{var}_qc'].where(
                ready_signal.signan != 0, 0)
            result = xr.merge([wt_signal, result], compat='override')

        # # Add COI as a variable
        # result['coi'] = wt_signal.coi

    except Exception as e:
        logger.error(f"Error in decompose_variables: {e}")
        raise

    return result


def data_statistics(data, formula='w*co2'):
    formulavar = formula_to_vars(formula) if isinstance(
        formula, str) else formula
    xy_name = ''.join(formulavar.xy)

    if len(formulavar.xy) == 2:
        data[f'cov_{xy_name}'] = xr.cov(
            data[formulavar.xy[0]], data[formulavar.xy[1]], 
            dim=[d for d in data.dims if d not in {
                'TIMESTAMP', 'natural_frequency'}])

        data[f'adv_{xy_name}'] = (
            data[formulavar.xy[0]].mean(dim=[d for d in data.dims if d not in {
                'TIMESTAMP', 'natural_frequency'}]) *
            data[formulavar.xy[1]].mean(dim=[d for d in data.dims if d not in {
                'TIMESTAMP', 'natural_frequency'}]))
    return data


def data_compute_product(data, formula='w*co2|w*h2o', name=None):
    """
    Calculate the product of variables specified by a formula for xarray.Dataset.

    Parameters:
    - data: xarray.Dataset
    - formula: string or object specifying the formula (e.g., 'w*co2|w*h2o')

    Returns:
    - xarray.Dataset with the calculated products
    """
    formulavar = formula_to_vars(formula) if isinstance(
        formula, str) else formula
    xy_name = name or ''.join(formulavar.xy)

    # Calculate the product for the main variables
    if xy_name not in data.data_vars:
        XY = data[formulavar.xy[0]]
        for c in formulavar.xy[1:]:
            XY = XY * data[c].conj()
        data[xy_name] = XY

    # Calculate the product for conditional sampling pairs
    for cs in formulavar.condsamp_pair:
        cs_name = ''.join(cs)
        if cs_name not in data.data_vars:
            CS = data[cs[0]]
            for c in cs[1:]:
                CS = CS * data[c].conj()
            data[cs_name] = CS

    # Convert the dictionary to an xarray.Dataset
    return data


def data_conditional_sampling(data, formula='w*co2|w*h2o'):
    """
    Calculate conditional sampling from a formula for xarray.Dataset.

    Parameters:
    - data: xarray.Dataset
    - formula: string or object specifying the formula (e.g., 'w*co2|w*h2o')

    Returns:
    - xarray.Dataset with the conditional sampling results
    """
    formulavar = formula_to_vars(formula) if isinstance(formula, str) else formula

    # Generate names for variables
    names = [''.join(formulavar.xy)] + [''.join(cs)
                                        for cs in formulavar.condsamp_pair]

    # If name in names not in data, calculate it
    for n in names:
        data = data_compute_product(data, formula=n)

    # Extract data arrays for conditional sampling
    data_arrays = [data[names[0]]] + [data[n] for n in names[1:]]

    # Perform conditional sampling
    φc = conditional_sampling(
        data_arrays[0], *[da for da in data_arrays[:]], names=names, label={1: "+", -1: "-"}
    ) if names else {}

    return φc


def data_partition(data, dst=None,
                   id_columns=None,
                   variables_available=None, **kwargs):
    """Apply every applicable wavelet conditional-sampling partition to *data*.

    Each gas-channel partition is attempted independently ("best effort"): a
    channel whose required variables are absent is skipped and logged at DEBUG,
    while a channel that raises is logged with a full traceback at ERROR and
    skipped, so the remaining channels still run.

    Args:
        data: xarray.Dataset of conditionally-sampled covariances.
        dst: Optional path stem; each channel writes ``f"{dst}.<tag>"`` if set.
        id_columns: Reserved; currently informational only.
        variables_available: Variables present in the source data. Defaults to
            ``['u', 'v', 'w', 'ts', 'co2', 'h2o']`` when ``None``.
        **kwargs: Ignored; accepted for forward compatibility.

    Returns:
        xarray.Dataset: *data* merged with the partition outputs. Channels other
        than the primary H2O one are suffixed (``_pH2O_CO``, ``_pCO``,
        ``_pCH4``) so their NEE/GPP/Reco variables don't collide on merge.
    """
    if variables_available is None:
        variables_available = ['u', 'v', 'w', 'ts', 'co2', 'h2o']
    id_columns = id_columns or ['TIMESTAMP'] + \
        [c for c in ['natural_frequency'] if c in data]

    def _run(label, required, fn, columns, suffix, tag):
        missing = [v for v in required if v not in variables_available]
        if missing:
            logger.debug("Skipping %s: missing variables %s.",
                         label, ', '.join(missing))
            return xr.Dataset()
        try:
            out = fn()[columns]
            if suffix:
                out = out.rename(
                    {var: f"{var}{suffix}" for var in out.data_vars})
            if dst:
                out.to_netcdf(f"{dst}.{tag}")
            return out
        except Exception:
            logger.exception("Partition %s failed; skipping.", label)
            return xr.Dataset()

    ds_pH2O = _run(
        "DWCS_H2O", ['w', 'co2', 'h2o'],
        lambda: partition_DWCS_H2O(
            data, NEE='NEE', GPP='GPP', Reco='Reco', CO2='wco2',
            CO2neg_H2Opos='wco2-wh2o+', CO2neg_H2Oneg='wco2-wh2o-', NIGHT=None),
        ['NEE', 'GPP', 'Reco'], '', "FCO2_condH2O")

    ds_pH2O_CO = _run(
        "DWCS_H2O_CO", ['w', 'co2', 'h2o', 'co'],
        lambda: partition_DWCS_CO(
            data, NEE='NEE', GPP='GPP', Reco='Reco', ffCO2='ffCO2', CO2='wco2',
            CO2neg_H2Opos='wco2-wh2o+', CO2neg_H2Oneg='wco2-wh2o-',
            CO2pos_COpos='wco2+wco+', CO2pos_COneg='wco2+wco-', NIGHT=None),
        ['NEE', 'GPP', 'Reco', 'ffCO2'], '_pH2O_CO', "FCO2_condH2O_CO")

    ds_pCO = _run(
        "DWCS_CO", ['w', 'co2', 'co'],
        lambda: partition_DWCS_CO(
            data, NEE='NEE', GPP='GPP', Reco='Reco', ffCO2='ffCO2', CO2='wco2',
            CO2neg_H2Opos=['wco2-wco+', 'wco2-wco-'], CO2neg_H2Oneg=None,
            CO2pos_COpos='wco2+wco+', CO2pos_COneg='wco2+wco-', NIGHT=None),
        ['NEE', 'GPP', 'Reco', 'ffCO2'], '_pCO', "FCO2_condCO")

    ds_pCH4 = _run(
        "DWCS_CH4", ['w', 'co2', 'ch4'],
        lambda: partition_DWCS_CO(
            data, NEE='NEE', GPP='GPP', Reco='Reco', ffCO2='ffCO2', CO2='wco2',
            CO2neg_H2Opos=['wco2-wch4+', 'wco2-wch4-'], CO2neg_H2Oneg=None,
            CO2pos_COpos='wco2+wch4+', CO2pos_COneg='wco2+wch4-', NIGHT=None),
        ['NEE', 'GPP', 'Reco', 'ffCO2'], '_pCH4', "FCO2_condCH4")

    return xr.merge([data, ds_pH2O, ds_pH2O_CO, ds_pCO, ds_pCH4], compat='override')


def open_files_in_folder(path):
    ds = xr.open_mfdataset(
        os.path.join(path, '*.nc'), combine='nested', concat_dim='TIMESTAMP')
    return ds


def data_average_dims(data, id_cols=None):
    if id_cols is None:
        id_cols = {'TIMESTAMP', 'natural_frequency'}
    ds = data.mean(dim=[d for d in data.dims if d not in id_cols])
    return ds


def data_integrate_in_frequency(data, f0, freq='natural_frequency'):
    assert freq in data.dims, f'Dim `{freq}` not found in data.'
    ds = data.sel({freq: slice(f0)}
                  ).sum(dim=freq)
    return ds


def process(datetimerange, fileduration, input_path, acquisition_frequency,
            covariance=None, output_folderpath=None, verbosity=1,
            overwrite=False, processing_time_duration="1D",
            reader_method='ep_raw_lvl',
            internal_averaging=None, dt=0.05,
            integration_period=None,
            identifier=None,
            filter_criteria=None,
            method="dwt", averaging=30, **kwargs):
    if filter_criteria is None:
        filter_criteria = {}
    logger.debug('--- Starting process ---')
    local_args = locals()
    info_t_start = time.time()

    def _date_from_yl(date):
        date = re.sub('[-: ]', '', str(date))
        if processing_time_duration.endswith("D"):
            date = date[:8]
        if processing_time_duration.endswith("H") or processing_time_duration.endswith("Min"):
            date = date[:12]
        return date
    
    def _validate_run(date, yl, compare_start=True, compare_end=False):
        # recheck if files exist and overwrite option
        # doesn't save time (maybe only save 5min)
        file_name = os.path.basename(output_path.format(date))
        part_name0 = file_name.rsplit('_', 1)[0] + '_' if compare_start else ''
        part_name1 = file_name.rsplit('.', 1)[-1] if compare_end else ''
        current_files = [p for p in os.listdir(os.path.dirname(
            output_path)) if p.startswith(part_name0) and p.endswith(part_name1)]

        if not overwrite and current_files: #file_name in os.path.exists(output_pathmodel.format(date)):
            logger.warning(
                "UserWarning: Skipping, file already exists ({}).".format(date))
            return False

        # # if all([os.path.exists(output_pathmodel.format(_yl.strftime('%Y%m%d%H%M'))) for _yl in yl[:-1]]):
        # if all([output_pathmodel.format(_yl.strftime('%Y%m%d%H%M')) in current_files for _yl in yl[:-1]]):
        #     logger.warning(
        #         "UserWarning: Skipping, file already exists ({}).".format(date))
        #     return False
        # # elif any([os.path.exists(output_pathmodel.format(_yl.strftime('%Y%m%d%H%M'))) for _yl in yl[:-1]]):
        # if any([output_pathmodel.format(_yl.strftime('%Y%m%d%H%M')) in current_files for _yl in yl[:-1]]):
        #     logger.warning("UserWarning: Continuing but some files already exist ({}), others don't ({}).".format(
        #         ', '.join([_yl.strftime('%Y%m%d%H%M') for _yl in yl[:-1] if output_pathmodel.format(_yl.strftime('%Y%m%d%H%M')) in current_files]),
        #         ', '.join([_yl.strftime('%Y%m%d%H%M') for _yl in yl[:-1] if not output_pathmodel.format(_yl.strftime('%Y%m%d%H%M')) in current_files]),
        #     ))
        #     return True

        if commons.checkifinprogress(curoutpath_inprog):
            return False
        return True
        
    def _load_data():
        def date2name(date):
            return date.strftime('%Y%m%d-%H%M')

        start_time = time.time()
        paths = [
            file
            for name in [date2name(d) for d in yl]
            for file in glob.glob(f'{os.path.join(input_path, name)}*')
        ]

        data = []

        for p in paths:
            try:
                data += [READERS.get(reader_method)(p)]
            except Exception as e:
                logger.error(f"Error when reading {p}: {e}")

        try:
            data = xr.concat(data, dim='TIMESTAMP')
        except Exception as e:
            data = xr.Dataset()
            logger.error(str(e))
        # if data.size == 0:
        #     logger.warning(
        #             f"UserWarning: No file found ({date}, path: {input_path}).")
        #     return None
        logger.debug(f'\tLoading data took {round(time.time() - start_time)} s.')
        return data
    
    def _exit():
        if os.path.exists(curoutpath_inprog):
            os.remove(curoutpath_inprog)

    # raw_kwargs = {'path': input_path, 'fkwargs': {
    #     'dt': 1/acquisition_frequency}}
    # raw_kwargs.update({k: v for k, v in kwargs.items() if k in ['fmt']})

    run_kwargs = {
        'fs': acquisition_frequency,
        'method': method,
        # 'averaging': averaging,
        'varstorun': covariance or commons.available_combinations(commons.DEFAULT_COVARIANCE),
        **kwargs.get("run_kwargs", {}),
        **kwargs.get("wt_kwargs", {})
    }

    logging_kwargs = kwargs.get("logging_kwargs", {})

    identifier = identifier or kwargs.get('sitename', '00000')
    
    run_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    # output_path = ""
    # output_pathmodel = ""
    curoutpath_inprog = ""

    
    if output_folderpath is not None:
        commons.start_logging(output_folderpath, **logging_kwargs)
        try:
            commons.save_locals(local_args, os.path.join(output_folderpath, f'log/setup_{run_time}.yml'))
        except Exception as e:
            logger.warning(f"Could not save local arguments to file: {e}")
        
    ymd = [datetimerange.split(
        '-')[0], datetimerange.split('-')[1], f'{fileduration}min']
    
    # if isinstance(averaging, (list, tuple)):
    #     averaging = averaging[-1]
    # if internal_averaging is None:
    #     internal_averaging = averaging
    
    # fulldata = pd.DataFrame()
    ds_collection = []
    
    _, _, _f = ymd
    ymd = commons.list_time_in_period(*ymd, processing_time_duration, include='right')
    # ymd = {y[-1]: y for y in ymd}
    
    logger.debug(
        f'Start date loop at {round(time.time() - info_t_start)} s.')

    # Skip two line
    for yl in ymd:
        info_t_yl_ymd = time.time()
        
        try:
            date = _date_from_yl(yl[0])

            logger.info("%s reading", date)

            if output_folderpath is not None:
                output_path = str(os.path.join(
                    output_folderpath,
                    "wavelet_full_cospectra",
                    f"{identifier}_full_cospectra_{date}_{run_time}.nc"))
                commons.mkdirs(output_path)
                curoutpath_inprog = f"{output_path}.inprogress"
                logger.debug(f'In progress file: {curoutpath_inprog}.')
                if not _validate_run(date, yl):
                    continue
        except Exception as e:
            logger.critical(e)
            warnings.warn(str(e))
            continue

        try:
            data = _load_data()
        except Exception as e:
            logger.critical(e)
            raise
        
        if data is None:
            _exit()
            continue

        try:
            # main run
            for var, criteria in filter_criteria.items():
                if var in data:
                    data[var] = data[var].where(
                        (data[var] >= criteria.start) & (data[var] < criteria.stop))
                else:
                    logger.warning(f"Trying to filter {var}, but variable not found in data.")

            ds = data_run(data, sel={'TIMESTAMP': slice(min(yl), max(yl) + datetime.timedelta(minutes=fileduration))},
                      dst=None, **run_kwargs)
            ds_average = ds.mean(dim=[d for d in ds.dims if d not in {
                'TIMESTAMP', 'natural_frequency'}])
            ds_average.to_netcdf(output_path)
            ds_collection += [ds_average]

            # allvars = run_kwargs['varstorun']
            # saved_files = []
            # for f in allvars:
            #     run_kwargs['varstorun'] = [f]
            #     output = data_run(data, period=[min(yl), max(yl)],
            #                   meta=meta,
            #                   dst=output_path, **run_kwargs)
            #     saved_files.append(output.saved)
            #     fulldata = pd.concat([fulldata, output.data], axis=0)

            # for f in [s for s_ in saved_files for s in s_]:
            #     if os.path.exists(f):
            #         os.rename(f, f.replace('.part', ''))

        except Exception as e:
            logger.critical(e)

        logger.debug(
            f"Date loop ({yl[0].strftime('%Y%m%d-%H%M')}:{yl[-1].strftime('%Y%m%d-%H%M')}) took {round(time.time() - info_t_yl_ymd)} s.")
        _exit()
    
    logger.debug(f'End date loop at {round(time.time() - info_t_start)} s.')
    logger.debug(f"integration_period: {integration_period}.")

    # if output_pathmodel and not fulldata.empty:
    #     # timestamp = pd.Timestamp.now().strftime('%Y%m%dT%H%M%S_%f')
    #     dst_path = os.path.join(output_folderpath, os.path.basename(
    #         output_pathmodel.format(run_time)))
    #     if integration_period:
    #         fulldata = integrate_cospectra(fulldata, 1/integration_period, dst_path=None)
    #     fulldata.to_csv(dst_path, index=False)

    # Concatenate all data
    if ds_collection:
        ds_collection = xr.concat(ds_collection, dim='TIMESTAMP')
        ds_collection = ds_collection.rename(
            {var: f"csp_{var}" for var in ds_collection.data_vars if 'natural_frequency' in ds_collection[var].dims})

        if integration_period:
            ds_collection_i = data_integrate_in_frequency(
                ds_collection, 1/(integration_period*60))
            ds_collection_i = ds_collection_i.rename(
                {var: f"{var}_int" for var in ds_collection_i.data_vars})

            ds_collection = ds_collection.merge(ds_collection_i)
    else:
        logger.error('No data was collected.')
        ds_collection = xr.Dataset()

    # Include metadata
    meta = {
        'identifier': identifier,
        'processing_time_duration': processing_time_duration,
        'acquisition_frequency': acquisition_frequency,
    }
    meta.update(ds_collection.attrs)
    ds_collection.attrs.update(meta)

    logger.debug(
        f'\t\tFull process took {round(time.time() - info_t_start)} s (run_wt).')
    return ds_collection


def data_run(data, varstorun, sel=None, average_period='30min', dst=None, **kwargs):
    """
    Main function to decompose, calculate covariance, and average data using xarray.Dataset.

    Parameters:
    - data: xarray.Dataset
    - varstorun: list of formulas to run
    - sel: dict with keys matching dimensions and values (c.f. xarray.Dataset.sel)
    - average_period: period for averaging
    - output_kwargs: output configuration
    - meta: metadata dictionary
    - **kwargs: additional arguments for decompose_data

    Returns:
    - xarray.Dataset with processed data
    """
    logger.debug(
        'Start data_run.')
    info_t_main = time.time()
    vars_unique = list(
        set([var for f in varstorun for var in formula_to_vars(f).uniquevars]))
    
    # Decompose all required variables
    data_decomposed = decompose_variables(
        data.stack(z=data.dims), vars_unique, nan_tolerance=.3, **kwargs).unstack()
    data_decomposed.attrs.update({
        'averaging': average_period,
        'method': f"{kwargs.get('method', '')} ~{kwargs.get('mother_wavelet', '')}",
        'dt': kwargs.get('dt', np.nan)
    })

    # Select valid dates
    if sel:
        data_decomposed = data_decomposed.sel(sel)

    # Calculate covariance
    info_t_calc_product = time.time()
    uniquecovs = list(
        set([c for f in varstorun for c in formula_to_vars(f).combinations]))

    # Calculate statistics
    data_stats = [
        data_statistics(data, formula=f)
        for f in uniquecovs]
    data_stats = xr.merge(data_stats)

    # Calculate product from formula for each unique covariance
    data_product = [
        data_compute_product(data_decomposed, formula=f)
        for f in uniquecovs]
    logger.debug(
        f'\tCalculate product from formula took {round(time.time() - info_t_calc_product)} s.')
    data_product = xr.merge(data_product)
    data_product = xr.merge([data_decomposed, data_product])

    # Calculate conditional sampling
    info_t_calc_cond_samp = time.time()
    data_condsamp = [
        data_conditional_sampling(data_product, f)
        for f in varstorun
    ]
    logger.debug(
        f'\tCalculate conditional sampling took {round(time.time() - info_t_calc_cond_samp)} s.')
    data_condsamp = xr.merge(data_condsamp)

    # Low-frequency
    xycovs = list(
        set(["*".join([f"{c}_lf" for c in formula_to_vars(f).xy]) for f in varstorun]))
    data_lf = [
        data_compute_product(data_decomposed, formula=f, name=f.replace('_lf', '') + "_lf")
        for f in xycovs]
    logger.debug(
        f'\tCalculate product from formula took {round(time.time() - info_t_calc_product)} s.')
    data_lf = xr.merge(data_lf)

    # Average data
    # data_averaged = data_product.groupby('TIMESTAMP').sum('natural_frequency')
    # data_integrated = data_product.where(data_product.natural_frequency >=
    #                                      1/(30*60), drop=True).sum('natural_frequency')

    # Rename variables
    data_product = data_product.rename(
        {var: f"{var}~" for var in data_product.data_vars})

    # Merge data
    response = xr.merge([data, data_stats, data_product, data_lf,
                        data_condsamp], compat='override')
    
    # Merge attrs
    for ds in [data, data_stats, data_product, data_lf, data_condsamp]:
        response.attrs.update(ds.attrs)

    # Save data
    if dst:
        response.to_netcdf(dst)
        response.attrs.setdefault('pipeout', []).append(f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}    Saved in: {dst}.")

    logger.debug(
        f'\tMain took {round(time.time() - info_t_main)} s.')
    return response


def run_from_eddypro(path,
                     # ="input/EP/FR-Gri_sample.eddypro",
                     #  covariance=["w*co2|w|co2|h2o", "w*co2|w*h2o", "w*h2o",],
                     #  processduration='6H',
                     **kwargs):
    c = eddypro.extract_info_from_eddypro_setup(eddypro=path)
    c.update(**kwargs)

    for path in ['input_path', 'output_folderpath']:
        if c.get(path, None) is not None:
            c[path] = os.path.abspath(c[path])

    # if covariance is None:
    #     # TODO: get variables available from eddypro
    #     covariance = commons.available_combinations(commons.DEFAULT_COVARIANCE)

    return process(**c)
