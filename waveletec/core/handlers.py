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
import datetime
import glob

# 3rd party modules
from functools import reduce
import numpy as np
import pandas as pd
import xarray as xr

# project modules
from . import commons
from ..io import READERS
# from .read_data import loaddatawithbuffer
from .wavelet_functions import universal_wt, formula_to_vars, prepare_signal, bufferforfrequency_dwt, bufferforfrequency
from ..extra.partitioning.coimbra_et_al_2025 import conditional_sampling, partition_DWCS, partition_DWCS_CO, partition_DWCS_H2O
from ..extra import eddypro as eddypro


logger = logging.getLogger(__name__)

def decompose_variables(data, variables=['w', 'co2'],
                        nan_tolerance=.3, **kwargs):
    """
    Calculate data decomposed with wavelet transform for xarray.Dataset.

    Parameters:
    - data: xarray.Dataset
    - variables: list of variable names to decompose
    - nan_tolerance: tolerance for NaN values
    - **kwargs: additional arguments for universal_wt

    Returns:
    - xarray.Dataset with decomposed variables
    """
    # Initialize output dataset
    result = xr.Dataset()

    # Add the original coordinates
    result = result.assign_coords(data.coords)

    # Placeholder for wavelet scales and COI
    sj = None
    coi = None

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
            result[var] = (('natural_frequency', *data[var].dims), wt_signal.wave)
            result[f'{var}_qc'] = (('natural_frequency', *data[var].dims), np.where(
                ready_signal.signan, 0, wt_signal.coi))

            # Update scales and COI (assuming they are the same for all variables)
            sj = wt_signal.sj
            coi = wt_signal.coi

        # Add scales as a coordinate
        result = result.assign_coords({'natural_frequency': sj})

        # Add COI as a variable
        result['coi'] = (('natural_frequency', *data[var].dims), coi)

    except Exception as e:
        logger.error(f"Error in decompose_variables: {e}")
        raise

    return result


def data_compute_product(data, formula='w*co2|w*h2o'):
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
    xy_name = ''.join(formulavar.xy)

    # Calculate the product for the main variables
    if xy_name not in data.data_vars:
        XY = data[formulavar.xy[0]]
        for c in formulavar.xy[1:]:
            XY = XY * data[c].conj()
        data[xy_name] = XY

    # Calculate the product for conditional sampling pairs
    for cs in formulavar.condsamp_pair:
        cs_name = ''.join(cs)
        if (cs_name not in data.data_vars) and (cs_name not in data.data_vars):
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


def open_files_in_folder(path):
    ds = xr.open_mfdataset(
        os.path.join(path, '*.nc'), combine='nested', concat_dim='TIMESTAMP')
    ds = ds.mean(dim=[d for d in ds.dims if d not in {
                'TIMESTAMP', 'natural_frequency'}])
    # ds = ds.compute()
    return ds


def data_partition(data, dst=None,
                   id_columns=None,
                   variables_available=['u', 'v', 'w', 'ts', 'co2', 'h2o'], **kwargs):
    id_columns = id_columns or ['TIMESTAMP'] + \
        [c for c in ['natural_frequency'] if c in data]

    ds_pH2O = xr.Dataset()
    ds_pH2O_CO = xr.Dataset()
    ds_pCO = xr.Dataset()
    ds_pCH4 = xr.Dataset()

    h2o_dw_required_variables = ['w', 'co2', 'h2o']
    is_lacking_variable = sum(
        [v not in variables_available for v in h2o_dw_required_variables])
    if not is_lacking_variable:
        logger.debug("partition_DWCS_H2O")
        try:
            ds_pH2O = partition_DWCS_H2O(
                data, NEE='NEE', GPP='GPP', Reco='Reco', CO2='wco2',
                CO2neg_H2Opos='wco2-wh2o+',
                CO2neg_H2Oneg='wco2-wh2o-', NIGHT=None)\
                .filter(id_columns + ['NEE', 'GPP', 'Reco'])

            if dst:
                ds_pH2O.to_netcdf(dst + ".FCO2_condH2O")
        except Exception as e:
            logging.warning(str(e))
    else:
        logger.debug(
            f"Missing variables {', '.join([v for v in h2o_dw_required_variables if v not in variables_available])}.")

    h2o_co_dw_required_variables = ['w', 'co2', 'h2o', 'co']
    is_lacking_variable = sum(
        [v not in variables_available for v in h2o_co_dw_required_variables])
    if not is_lacking_variable:
        try:
            ds_pH2O_CO = partition_DWCS_CO(
                data, NEE='NEE', GPP='GPP', Reco='Reco', ffCO2='ffCO2',
                CO2='wco2',
                CO2neg_H2Opos='wco2-wh2o+',
                CO2neg_H2Oneg='wco2-wh2o-',
                CO2pos_COpos='wco2+wco+',
                CO2pos_COneg='wco2+wco-',
                NIGHT=None)\
                .filter(id_columns + ['NEE', 'GPP', 'Reco', 'ffCO2'])

            if dst:
                ds_pH2O_CO.to_netcdf(dst + ".FCO2_condH2O_CO")
        except Exception as e:
            logging.warning(str(e))
    else:
        logger.debug(
            f"Missing variables {', '.join([v for v in h2o_co_dw_required_variables if v not in variables_available])}.")

    co_dw_required_variables = ['w', 'co2', 'co']
    is_lacking_variable = sum(
        [v not in variables_available for v in co_dw_required_variables])
    if not is_lacking_variable:
        try:
            ds_pCO = partition_DWCS_CO(
                data, NEE='NEE', GPP='GPP', Reco='Reco', ffCO2='ffCO2',
                CO2='wco2',
                CO2neg_H2Opos=['wco2-wco+', 'wco2-wco-'],
                CO2neg_H2Oneg=None,
                CO2pos_COpos='wco2+wco+',
                CO2pos_COneg='wco2+wco-',
                NIGHT=None)\
                .filter(id_columns + ['NEE', 'GPP', 'Reco', 'ffCO2'])

            if dst:
                ds_pCO.to_netcdf(dst + ".FCO2_condCO")
        except Exception as e:
            logging.warning(str(e))
    else:
        logger.debug(
            f"Missing variables {', '.join([v for v in co_dw_required_variables if v not in variables_available])}.")

    ch4_dw_required_variables = ['w', 'co2', 'ch4']
    is_lacking_variable = sum(
        [v not in variables_available for v in ch4_dw_required_variables])
    if not is_lacking_variable:
        try:
            ds_pCH4 = partition_DWCS_CO(
                data, NEE='NEE', GPP='GPP', Reco='Reco', ffCO2='ffCO2',
                CO2='wco2',
                CO2neg_H2Opos=['wco2-wch4+', 'wco2-wch4-'],
                CO2neg_H2Oneg=None,
                CO2pos_COpos='wco2+wch4+',
                CO2pos_COneg='wco2+wch4-',
                NIGHT=None)\
                .filter(id_columns + ['NEE', 'GPP', 'Reco', 'ffCO2'])

            if dst:
                ds_pCH4.to_netcdf(dst + ".FCO2_condCH4")
        except Exception as e:
            logging.warning(str(e))
    else:
        logger.debug(
            f"Missing variables {', '.join([v for v in ch4_dw_required_variables if v not in variables_available])}.")

    return xr.merge([data, ds_pH2O, ds_pH2O_CO, ds_pCO, ds_pCH4])


def open_files_in_folder(path):
    ds = xr.open_mfdataset(
        os.path.join(path, '*.nc'), combine='nested', concat_dim='TIMESTAMP')
    return ds


def data_average_dims(data, id_cols={'TIMESTAMP', 'natural_frequency'}):
    ds = data.mean(dim=[d for d in data.dims if d not in id_cols])
    return ds


def data_integrate_in_frequency(data, f0, freq='natural_frequency'):
    assert freq in data.dims, f'Dim `{freq}` not found in data.'
    ds = data.where(data[freq] >= f0, drop=True).sum(dim=freq)
    return ds


def process(datetimerange, fileduration, input_path, acquisition_frequency,
            covariance=None, output_folderpath=None, verbosity=1,
            overwrite=False, processing_time_duration="1D",
            reader_method='ep_raw_lvl',
            internal_averaging=None, dt=0.05,
            integration_period=None,
            identifier=None,
            filter_criteria={},
            method="dwt", averaging=30, **kwargs):
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
    ymd = commons.list_time_in_period(*ymd, processing_time_duration, include='both')
    # ymd = {y[-1]: y for y in ymd}
    
    logger.debug(
        f'Start date loop at {round(time.time() - info_t_start)} s.')

    # Skip two line
    prev_print = '\n'
    for yl in ymd:
        info_t_yl_ymd = time.time()
        date = _date_from_yl(yl[0])

        print(prev_print, date, 'reading', ' '*10, sep=' ', end='\n')
        prev_print = '\x1B[1A\r'

        if output_folderpath is not None:
            output_path = str(os.path.join(
                output_folderpath,
                f"wavelet_full_cospectra",
                f"{identifier}_full_cospectra_{date}_{run_time}.nc"))
            commons.mkdirs(output_path)
            curoutpath_inprog = f"{output_path}.inprogress"
            logger.debug(f'In progress file: {curoutpath_inprog}.')
            if not _validate_run(date, yl):
                continue

        try:
            data = _load_data()
        except Exception as e:
            logger.critical(e)
            raise(e)
        
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

            ds = data_run(data, sel={'TIMESTAMP': slice(min(yl), max(yl))},
                      dst=output_path, **run_kwargs)
            ds_collection += [ds.mean(dim=[d for d in ds.dims if d not in {
                'TIMESTAMP', 'natural_frequency'}])]

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
            ds_collection_i = (
                ds_collection[
                    [var for var in ds_collection.data_vars
                    if 'natural_frequency' in ds_collection[var].dims]]
                .where(ds_collection.natural_frequency >= 1/(integration_period*60), drop=True)
                .mean('natural_frequency')
            )
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
        f'Start data_run.')
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

    # Average data
    # data_averaged = data_product.groupby('TIMESTAMP').sum('natural_frequency')
    # data_integrated = data_product.where(data_product.natural_frequency >=
    #                                      1/(30*60), drop=True).sum('natural_frequency')

    # Rename variables
    data_product = data_product.rename(
        {var: f"{var}~" for var in data_product.data_vars})

    response = xr.merge([data, data_product,
                        data_condsamp], compat='override')

    # Save data
    if dst:
        response.to_netcdf(dst)
        response.attrs.setdefault('pipeout', []).append(f'{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}    Saved in: {dst}.')

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
