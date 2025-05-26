# standard modules
import os
import sys
import re
import logging
import datetime
# Project modules
import yaml
from . import _core as hc24
from ._core import partition as ptt
from ._core import wavelet_functions as wavelet_functions


def __possible_combinations__(interesting_combinations, variables_available):
        # Reduce interesting to possible
        possible_combinations = [sum([v not in variables_available for v in re.split('[*|]', t)])==0 for t in interesting_combinations]
        # Limit run to the realm of possible 
        varstorun = [t for t, p in zip(interesting_combinations, possible_combinations) if p]
        return varstorun


def sample_raw_data(inputpath, datetimerange, acquisition_frequency=20, fileduration=30, **kwargs):
    raw_kwargs = {'path': inputpath, 'fkwargs': {'dt': 1/acquisition_frequency}}
    kwargs['fmt'] = kwargs.get('fmt', {})
    if 'gas4_name' in kwargs.keys(): kwargs['fmt'].update({kwargs.pop('gas4_name'): '4th gas'})
    raw_kwargs.update({k: v for k, v in kwargs.items() if k in ['fmt']})

    ymd = [datetimerange.split('-')[0], datetimerange.split('-')[1], f'{fileduration}min']
    _, _, _f = ymd
    ymd = hc24.list_time_in_period(*ymd, '1D', include='both')

    for ymd_i, yl in enumerate(ymd):
        data = hc24.loaddatawithbuffer(
            yl, d1=None, freq=_f, buffer=0, f_freq=_f, **raw_kwargs)
        break
    return data


def eddypro_wavelet_run(sitename, inputpath, outputpath, datetimerange, acquisition_frequency=20, fileduration=30, 
         processduration='1D', integratioperiod=None, preaverage=None,
         covariance = None, variables_available=['u', 'v', 'w', 'ts', 'co2', 'h2o'], denoise=0, deadband=[], 
         method = 'dwt', wave_mother='db6', **kwargs):
    local_args = locals()

    if outputpath is not None:
        hc24.start_logging(outputpath)

        # Select output file path
        if method == 'cov':
            outputpath = str(os.path.join(outputpath, str(sitename)+'{}_{}.csv'))
        else:
            outputpath = str(os.path.join(outputpath, 'wavelet_full_cospectra', str(sitename)+'_CDWT{}_{}.csv'))

        # Save args for run
        hc24.mkdirs(outputpath)
        with open(os.path.join(os.path.dirname(os.path.dirname(outputpath)), f'log/setup_{datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")}.yml'), 'w+') as stp:
            yaml.safe_dump(local_args, stp)

    # Select covariances
    # x*y → Cov(x, y)
    # x*y|x*z|x*... → Cov(x, y)|Cov(x, z),Cov(x, ...)
    if covariance is None:
        interesting_combinations = ['co2*co2', 'h2o*h2o', 'ts*ts', 'co*co',  'ch4*ch4', 'n2o*n2o',
                                    'w*co2', 'w*h2o', 'w*ts', 'w*co', 'w*ch4',  'w*n2o',
                                    'w*co2|w*h2o', 'w*co2|w*co', 'w*co2|w*ch4', 'w*co2|w*ts', 'w*co2|w*h2o|w*co', 
                                    'w*h2o|w*co2', 'w*h2o|w*co', 'w*h2o|w*ch4', 'w*h2o|w*ts', 
                                    'w*co|w*co2',  'w*co|w*ts', 'w*co|w*ch4', 'w*co|w*h2o', 
                                    'w*ch4|w*co2',  'w*ch4|w*co', 'w*ch4|w*ts', 'w*ch4|w*h2o', 
                                    'w*ts|w*co2',  'w*ts|w*co', 'w*ts|w*ch4', 'w*ts|w*h2o',
                                    ]        
        # Reduce interesting to possible
        # Limit run to the realm of possible 
        covariance = __possible_combinations__(interesting_combinations, variables_available)

    # RUN WAVELET FLUX PROCESSING
    # ymd = [START_DATE, END_DATE, FILE_FREQUENCY]
    raw_kwargs = {'path': inputpath, 'fkwargs': {'dt': 1/acquisition_frequency}}
    kwargs['fmt'] = kwargs.get('fmt', {})
    if 'gas4_name' in kwargs.keys(): kwargs['fmt'].update({kwargs.pop('gas4_name'): '4th gas'})
    raw_kwargs.update({k: v for k, v in kwargs.items() if k in ['fmt']})
    data = wavelet_functions.load_data_and_loop(ymd = [datetimerange.split('-')[0], datetimerange.split('-')[1], f'{fileduration}min'],
                                         output_path = outputpath,
                                         varstorun = covariance,
                                         averaging = [fileduration],
                                         processing_time_duration = processduration,
                                         method = method,
                                         wt_kwargs = {'fs': acquisition_frequency, 'wavelet': wave_mother},
                                         raw_kwargs = raw_kwargs,
                                         verbosity=5)
    return data


def integrate_full_spectra_into_file(sitename, outputpath, integratioperiod=60*30, **kwargs):
    # CONCAT INTO SINGLE FILE
    dst_path = os.path.join(outputpath, str(sitename)+f'_CDWT_full_cospectra.csv')
    
    wavelet_functions.integrate_cospectra(os.path.join(outputpath, 'wavelet_full_cospectra'),
                                          1/integratioperiod, '_CDWT_full_cospectra_([0-9]{12}).csv$', dst_path)
    #hc24.concat_into_single_file(
    #    os.path.join(outputpath, 'wavelet_full_cospectra'), str(sitename)+f'_CDWT_full_cospectra.+.{fileduration}mn.csv', 
    #    output_path=dst_path, skiprows=10)
    
def condition_sampling_partition(sitename, outputpath, variables_available=['u', 'v', 'w', 'ts', 'co2', 'h2o'], **kwargs):
    # RUN PARTITIONING
    dst_path = os.path.join(outputpath, str(sitename)+f'_CDWT_full_cospectra.csv')

    h2o_dw_required_variables = ['w','co2','h2o']
    is_lacking_variable = sum([v not in variables_available for v in h2o_dw_required_variables])
    if not is_lacking_variable:
        try:
            ptt.partition_DWCS_H2O(str(dst_path), 
                                        NEE='NEE', GPP='GPP', Reco='Reco', CO2='wco2', 
                                        CO2neg_H2Opos='wco2-wh2o+', 
                                        CO2neg_H2Oneg='wco2-wh2o-', NIGHT=None)\
                                    .filter(['TIMESTAMP', 'NEE', 'GPP', 'Reco'])\
                                    .to_file(os.path.join(outputpath, str(sitename)+f'_CDWT_partitioning_H2O.csv'), index=False)
        except Exception as e:
            logging.warning(str(e))
    
    h2o_co_dw_required_variables = ['w','co2','h2o','co']
    is_lacking_variable = sum([v not in variables_available for v in h2o_co_dw_required_variables])
    if not is_lacking_variable:
        try:
            ptt.partition_DWCS_CO(str(dst_path), 
                                        NEE='NEE', GPP='GPP', Reco='Reco', ffCO2='ffCO2',
                                        CO2='wco2', 
                                        CO2neg_H2Opos='wco2-wh2o+', 
                                        CO2neg_H2Oneg='wco2-wh2o-', 
                                        CO2pos_COpos='wco2+wco+', 
                                        CO2pos_COneg='wco2+wco-',
                                        NIGHT=None)\
                                        .filter(['TIMESTAMP', 'NEE', 'GPP', 'Reco', 'ffCO2'])\
                                        .to_file(os.path.join(outputpath, str(sitename)+f'_CDWT_partitioning_H2O_CO.csv'), index=False)
        except Exception as e:
            logging.warning(str(e))
    
    co_dw_required_variables = ['w','co2','co']
    is_lacking_variable = sum([v not in variables_available for v in co_dw_required_variables])
    if not is_lacking_variable:
        try:
            ptt.partition_DWCS_CO(str(dst_path), 
                                        NEE='NEE', GPP='GPP', Reco='Reco', ffCO2='ffCO2',
                                        CO2='wco2', 
                                        CO2neg_H2Opos=['wco2-wco+', 'wco2-wco-'], 
                                        CO2neg_H2Oneg=None, 
                                        CO2pos_COpos='wco2+wco+', 
                                        CO2pos_COneg='wco2+wco-',
                                        NIGHT=None)\
                                        .filter(['TIMESTAMP', 'NEE', 'GPP', 'Reco', 'ffCO2'])\
                                        .to_file(os.path.join(outputpath, str(sitename)+f'_CDWT_partitioning_CO.csv'), index=False)
        except Exception as e:
            logging.warning(str(e))
        
    ch4_dw_required_variables = ['w','co2','ch4']
    is_lacking_variable = sum([v not in variables_available for v in ch4_dw_required_variables])
    if not is_lacking_variable:
        try:
            ptt.partition_DWCS_CO(str(dst_path), 
                                        NEE='NEE', GPP='GPP', Reco='Reco', ffCO2='ffCO2',
                                        CO2='wco2', 
                                        CO2neg_H2Opos=['wco2-wch4+', 'wco2-wch4-'], 
                                        CO2neg_H2Oneg=None, 
                                        CO2pos_COpos='wco2+wch4+', 
                                        CO2pos_COneg='wco2+wch4-',
                                        NIGHT=None)\
                                        .filter(['TIMESTAMP', 'NEE', 'GPP', 'Reco', 'ffCO2'])\
                                        .to_file(os.path.join(outputpath, str(sitename)+f'_CDWT_partitioning_CH4.csv'), index=False)
        except Exception as e:
            logging.warning(str(e))


def handle_eddypro_setup(**args):
    if args['eddypro']:
        eddypro_setup = hc24.eddypro_tools.read_eddypro_metadata_file(args['eddypro'])
        if not 'sitename' in args.keys() or args['sitename'] is None: args['sitename'] = eddypro_setup['Project']['project_id']
        if not 'inputpath' in args.keys() or args['inputpath'] is None: args['inputpath'] = eddypro_setup['Project']['out_path'] + '/eddypro_raw_datasets/level_6'
        if not 'outputpath' in args.keys() or args['outputpath'] is None: args['outputpath'] = eddypro_setup['Project']['out_path'] + '/wavelet_flux'
        if not 'datetimerange' in args.keys() or args['datetimerange'] is None: args['datetimerange'] = eddypro_setup['Project']['pr_start_date'].replace('-', '') + eddypro_setup['Project']['pr_start_time'].replace(':', '') + '-' + \
            eddypro_setup['Project']['pr_end_date'].replace('-', '') + eddypro_setup['Project']['pr_end_time'].replace(':', '')
        if not 'fileduration' in args.keys() or args['fileduration'] is None: args['fileduration'] = int(eddypro_setup['RawProcess_Settings']['avrg_len'])

        if not 'metadata' in args.keys() or args['metadata'] is None: 
            if eddypro_setup['Project']['proj_file']: args['metadata'] = eddypro_setup['Project']['proj_file']
            else: args['metadata'] = args['eddypro'].rsplit('.', 1)[0] + '.metadata'
        
        try:
            if not 'gas4_name' in args.keys() or args['gas4_name'] is None: 
                gas4_col = eddypro_setup['Project']['col_n2o']
                eddypro_metad = hc24.eddypro_tools.read_eddypro_metadata_file(args['metadata'])
                args['gas4_name'] = eddypro_metad['FileDescription'][f'col_{gas4_col}_variable']
        except Exception as e:
            print(e)
    
    if args['metadata']:
        eddypro_metad = hc24.eddypro_tools.read_eddypro_metadata_file(args['metadata'])
        if not 'acquisition_frequency' in args.keys() or args['acquisition_frequency'] is None: args['acquisition_frequency'] = int(float(eddypro_metad['Timing']['acquisition_frequency']))
        #if args['fileduration'] is None: args['fileduration'] = int(eddypro_metad['Timing']['file_duration'])
    
    if not 'variables_available' in args.keys() or args['variables_available'] is None:
        if args['eddypro']:
            args['variables_available'] = ['u', 'v', 'w'] + [k for k in ['co2', 'h2o', 'ch4'] if float(eddypro_setup['Project'][f'col_{k}']) > 0]
        if args['metadata']:
            if float(eddypro_setup['Project']['col_n2o']) > 0:
                gas4 =  eddypro_metad['FileDescription'][f"col_{eddypro_setup['Project']['col_n2o']}_variable"]
                if gas4: args['variables_available'] = args['variables_available'] + [gas4]
    return args
