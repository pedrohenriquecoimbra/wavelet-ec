# standard modules
import os
import sys
import re
import logging
import datetime
# Project modules
import yaml
import scripts.coimbra2024_scripts as hc24
import scripts.wavelet_functions as wavelet_functions


def main(sitename, inputpath, outputpath, datetimerange, acquisition_frequency=20, fileduration=30, processduration='1D', integratioperiod=None, preaverage=None,
         covariance = None, variables_available=['u', 'v', 'w', 'ts', 'co2', 'h2o'], denoise=0, deadband=[], 
         method = 'dwt', wave_mother='db6', **kwargs):
    local_args = locals()

    logname = str(os.path.join(outputpath, f"log/current_{datetime.datetime.now().strftime('%y%m%dT%H%M%S')}.log"))
    hc24.mkdirs(logname)
    #with open(logname, "w+"): pass
    logging.basicConfig(filename=logname,
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.DEBUG, 
                        force=True)

    logging.captureWarnings(True)
    logging.info("STARTING THE RUN")

    # Create setup
    configure = hc24.structuredData()

    configure.processing_time_duration = processduration
    configure.preaverage = preaverage

    # Select output file path
    configure.output_path = str(os.path.join(outputpath, 'wavelet_full_cospectra', str(sitename)+'_CDWT{}_{}.csv'))

    # Save args for run
    hc24.mkdirs(configure.output_path)
    with open(os.path.join(os.path.dirname(os.path.dirname(configure.output_path)), f'wavelet_{datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")}.setup'), 'w+') as stp:
        yaml.safe_dump(local_args, stp)

    # Select raw input file path
    # e.g.: "<PROJECT FOLDER>/eddypro_output/eddypro_raw_datasets/level_6/"
    if isinstance(inputpath, dict):
        configure.raw_kwargs = {'path': {k: str(os.path.join(v, 'eddypro_raw_datasets/level_6')) for k, v in inputpath.items()}}
    else: configure.raw_kwargs = {'path': str(os.path.join(inputpath, 'eddypro_raw_datasets/level_6'))}

    # Select covariances
    # x*y → Cov(x, y)
    # x*y*z*... → Cov(x, y)|Cov(x, z),Cov(x, ...)
    if covariance is None:
        interesting_combinations = ['co2*co2', 'h2o*h2o', 'ts*ts', 'co*co',  'ch4*ch4', 'n2o*n2o',
                                    'w*co2', 'w*h2o', 'w*ts', 'w*co', 'w*ch4',  'w*n2o',
                                    'w*co2|w*h2o', 'w*co2|w*co', 'w*co2|w*ch4', 'w*co2|w*ts', 'w*co2|w*h2o|w*co', 
                                    'w*h2o|w*co2', 'w*h2o|w*co', 'w*h2o|w*ch4', 'w*h2o|w*ts', 
                                    'w*co|w*co2',  'w*co|w*ts', 'w*co|w*ch4', 'w*co|w*h2o', 
                                    'w*ch4|w*co2',  'w*ch4|w*co', 'w*ch4|w*ts', 'w*ch4|w*h2o', 
                                    'w*ts|w*co2',  'w*ts|w*co', 'w*ts|w*ch4', 'w*ts|w*h2o',
                                    #'u*w', 'v*w'
                                    ]

        # Reduce interesting to possible
        possible_combinations = [sum([v not in variables_available for v in re.split('[*|]', t)])==0 for t in interesting_combinations]
        # Limit run to the realm of possible 
        configure.varstorun = [t for t, p in zip(interesting_combinations, possible_combinations) if p]
    else:
        configure.varstorun = covariance

    # Select period of interest
    # [START_DATE, END_DATE, FILE_FREQUENCY]
    configure.ymd = [datetimerange.split('-')[0], datetimerange.split('-')[1], f'{fileduration}min']

    # Averaging (and integrating) time
    configure.averaging = [fileduration]
    configure.integrating = integratioperiod if integratioperiod is not None else fileduration * 60

    # Select wavelet method
    configure.method = method
    configure.denoise = bool(denoise)

    # Dead band
    if deadband: configure.deadband = {v: abs(float(d)) for v, d in zip(variables_available, deadband) if abs(float(d))>0}

    # Select dt
    configure.wt_kwargs = {'fs': acquisition_frequency, 'wavelet': wave_mother}
    configure.raw_kwargs.update({'fkwargs': {'dt': 1/acquisition_frequency},
                                 'fmt': {'co': '4th'}})

    # RUN WAVELET FLUX PROCESSING
    wavelet_functions.run_wt(**vars(configure), verbosity=5)


def __concat__(sitename, outputpath, integratioperiod=60*30, **kwargs):
    # CONCAT INTO SINGLE FILE
    dst_path = os.path.join(outputpath, str(sitename)+f'_CDWT_full_cospectra.csv')
    
    wavelet_functions.integrate_cospectra(os.path.join(outputpath, 'wavelet_full_cospectra'),
                                          '_CDWT_full_cospectra_([0-9]{12}).csv$',
                                          1/integratioperiod, dst_path)
    #hc24.concat_into_single_file(
    #    os.path.join(outputpath, 'wavelet_full_cospectra'), str(sitename)+f'_CDWT_full_cospectra.+.{fileduration}mn.csv', 
    #    output_path=dst_path, skiprows=10)
    
def __partition__(sitename, outputpath, fileduration=30, variables_available=['u', 'v', 'w', 'ts', 'co2', 'h2o'], **kwargs):
    # RUN PARTITIONING
    dst_path = os.path.join(outputpath, str(sitename)+f'_CDWT_full_cospectra.csv')

    h2o_dw_required_variables = ['w','co2','h2o']
    is_lacking_variable = sum([v not in variables_available for v in h2o_dw_required_variables])
    if not is_lacking_variable:
        try:
            hc24.partition_DWCS_H2O(str(dst_path), 
                                        NEE='NEE', GPP='GPP', Reco='Reco', CO2='wco2', 
                                        CO2neg_H2Opos='wco2-wh2o+', 
                                        CO2neg_H2Oneg='wco2-wh2o-', NIGHT=None)\
                                    .filter(['TIMESTAMP', 'NEE', 'GPP', 'Reco'])\
                                    .to_file(os.path.join(outputpath, str(sitename)+f'_CDWT_partitioning_H2O.{fileduration}mn.csv'), index=False)
        except Exception as e:
            logging.warning(str(e))
    
    h2o_co_dw_required_variables = ['w','co2','h2o','co']
    is_lacking_variable = sum([v not in variables_available for v in h2o_co_dw_required_variables])
    if not is_lacking_variable:
        try:
            hc24.partition_DWCS_CO(str(dst_path), 
                                        NEE='NEE', GPP='GPP', Reco='Reco', ffCO2='ffCO2',
                                        CO2='wco2', 
                                        CO2neg_H2Opos='wco2-wh2o+', 
                                        CO2neg_H2Oneg='wco2-wh2o-', 
                                        CO2pos_COpos='wco2+wco+', 
                                        CO2pos_COneg='wco2+wco-',
                                        NIGHT=None)\
                                        .filter(['TIMESTAMP', 'NEE', 'GPP', 'Reco', 'ffCO2'])\
                                        .to_file(os.path.join(outputpath, str(sitename)+f'_CDWT_partitioning_H2O_CO.{fileduration}mn.csv'), index=False)
        except Exception as e:
            logging.warning(str(e))
    
    co_dw_required_variables = ['w','co2','co']
    is_lacking_variable = sum([v not in variables_available for v in co_dw_required_variables])
    if not is_lacking_variable:
        try:
            hc24.partition_DWCS_CO(str(dst_path), 
                                        NEE='NEE', GPP='GPP', Reco='Reco', ffCO2='ffCO2',
                                        CO2='wco2', 
                                        CO2neg_H2Opos=['wco2-wco+', 'wco2-wco-'], 
                                        CO2neg_H2Oneg=None, 
                                        CO2pos_COpos='wco2+wco+', 
                                        CO2pos_COneg='wco2+wco-',
                                        NIGHT=None)\
                                        .filter(['TIMESTAMP', 'NEE', 'GPP', 'Reco', 'ffCO2'])\
                                        .to_file(os.path.join(outputpath, str(sitename)+f'_CDWT_partitioning_CO.{fileduration}mn.csv'), index=False)
        except Exception as e:
            logging.warning(str(e))
        
    ch4_dw_required_variables = ['w','co2','ch4']
    is_lacking_variable = sum([v not in variables_available for v in ch4_dw_required_variables])
    if not is_lacking_variable:
        try:
            hc24.partition_DWCS_CO(str(dst_path), 
                                        NEE='NEE', GPP='GPP', Reco='Reco', ffCO2='ffCO2',
                                        CO2='wco2', 
                                        CO2neg_H2Opos=['wco2-wch4+', 'wco2-wch4-'], 
                                        CO2neg_H2Oneg=None, 
                                        CO2pos_COpos='wco2+wch4+', 
                                        CO2pos_COneg='wco2+wch4-',
                                        NIGHT=None)\
                                        .filter(['TIMESTAMP', 'NEE', 'GPP', 'Reco', 'ffCO2'])\
                                        .to_file(os.path.join(outputpath, str(sitename)+f'_CDWT_partitioning_CH4.{fileduration}mn.csv'), index=False)
        except Exception as e:
            logging.warning(str(e))


def handle_eddypro_setup(**args):
    if args['eddypro']:
        eddypro_setup = hc24.read_eddypro_metadata_file(args['eddypro'])
        if args['sitename'] is None: args['sitename'] = eddypro_setup['Project']['project_title']
        if args['inputpath'] is None: args['inputpath'] = eddypro_setup['Project']['out_path']
        if args['outputpath'] is None: args['outputpath'] = eddypro_setup['Project']['out_path'] + '/wavelet_flux'
        if args['datetimerange'] is None: args['datetimerange'] = eddypro_setup['Project']['pr_start_date'].replace('-', '') + eddypro_setup['Project']['pr_start_time'].replace(':', '') + '-' + \
            eddypro_setup['Project']['pr_end_date'].replace('-', '') + eddypro_setup['Project']['pr_end_time'].replace(':', '')
        if args['fileduration'] is None: args['fileduration'] = int(eddypro_setup['RawProcess_Settings']['avrg_len'])
    
        if args['metadata'] is None: 
            if eddypro_setup['Project']['proj_file']: args['metadata'] = eddypro_setup['Project']['proj_file']
            else: args['metadata'] = args['eddypro'].rsplit('.', 1)[0] + '.metadata'
    
    if args['metadata']:
        eddypro_metad = hc24.read_eddypro_metadata_file(args['metadata'])
        if args['acquisition_frequency'] is None: args['acquisition_frequency'] = int(float(eddypro_metad['Timing']['acquisition_frequency']))
        #if args['fileduration'] is None: args['fileduration'] = int(eddypro_metad['Timing']['file_duration'])
    
    return args

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-ep', '--eddypro',   type=str)
    parser.add_argument('-m', '--metadata',   type=str)
    parser.add_argument('-s', '--sitename',   type=str)
    parser.add_argument('-i', '--inputpath',  type=str)
    parser.add_argument('-o', '--outputpath', type=str)
    parser.add_argument('-d', '--datetimerange', type=str)
    parser.add_argument('-af', '--acquisition_frequency', type=int)
    parser.add_argument('-fd', '--fileduration', type=int)
    parser.add_argument('-ip', '--integratioperiod', type=int)
    parser.add_argument('-v', '--variables_available', type=str, nargs='+')
    parser.add_argument('-dk', '--despike', type=int)#, nargs=1)
    parser.add_argument('-dn', '--denoise', type=int)#, nargs=1)
    parser.add_argument('-db', '--deadband', type=str, nargs='+')
    parser.add_argument('-cov', '--covariance', type=str, nargs='+')
    parser.add_argument('--method', type=str, default='dwt')
    parser.add_argument('--wave_mother', type=str, default='db6')
    parser.add_argument('--run', type=int, default=1)
    parser.add_argument('--concat', type=int, default=1)
    parser.add_argument('--partition', type=int, default=1)
    parser.add_argument('--processduration', type=str, default='1D')
    parser.add_argument('--preaverage', type=str, default=None)   
    args = parser.parse_args()
    args = handle_eddypro_setup(**vars(args))

    # default
    args['integratioperiod'] = args['integratioperiod'] if args['integratioperiod'] is not None else args['fileduration'] * 60
    
    args.pop('eddypro')
    args.pop('metadata')
    run = args.pop('run')
    concat = args.pop('concat')
    partition = args.pop('partition')

    print('Start run w/')
    # replace os.get_cwd() for '' if str
    print('\n'.join([f'{k}:\t{v[:5] + "~" + v[-25:] if isinstance(v, str) and len(v) > 30 else v}' for k, v in args.items()]), end='\n\n')
    
    # Assert variables have been assigned
    missing_args = [f'`{k}`' for k in ['sitename', 'inputpath', 'outputpath', 'datetimerange', 'acquisition_frequency', 'fileduration'] if args[k] is None]
    assert len(missing_args) == 0, f'Missing argument in: {", ".join(missing_args)}.'

    if run: main(**args)
    if concat: __concat__(**args)
    if partition: __partition__(**args)
"""
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('sitename',   type=str)
    parser.add_argument('inputpath',  type=str)
    parser.add_argument('outputpath', type=str)
    parser.add_argument('datetimerange', type=str)
    parser.add_argument('acquisition_frequency', type=int, nargs='?', default=20)
    parser.add_argument('fileduration', type=int, nargs='?', default=30)
    parser.add_argument('-sr', '--acquisition_frequency', type=int, default=20)
    parser.add_argument('-fd', '--fileduration', type=int, default=30)
    #parser.add_argument('-cov', '--covariance', type=str, default=None)
    args = parser.parse_args()

    #main(**vars(args))
    print(vars(args))
"""