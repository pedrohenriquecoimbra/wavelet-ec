# standard modules
import os
import sys
import re
import logging
import time
import datetime
import pandas as pd
import numpy as np
# Project modules
import yaml
from waveletec._core import commons as hc24
from waveletec._core import wavelet_functions as wavelet_functions

logger = logging.getLogger('cond_ec')

def run_cec(ymd, raw_kwargs, output_path, 
            w='w', c='co2', q='h2o', averaging=30,
            processing_time_duration="1D", verbosity=1):
    """
    Zahn et al. (2022)
    """
    if isinstance(averaging, (list, tuple)): averaging = averaging[-1]

    info_t_start = time.time()
    logger.info('Entered wavelet code (run_cec).')
    
    suffix = raw_kwargs['suffix'] if 'suffix' in raw_kwargs.keys() else ''

    _, _, _f = ymd
    ymd = hc24.list_time_in_period(*ymd, processing_time_duration, include='both')
            
    logger.info(f'Start date loop at {round(time.time() - info_t_start)} s (run_cec).')
    
    # Skip two line
    prev_print = '\n'
    for ymd_i, yl in enumerate(ymd):
        info_t_startdateloop = time.time()

        date = re.sub('[-: ]', '', str(yl[0]))
        if processing_time_duration.endswith("D"): date = date[:8]
        if processing_time_duration.endswith("H") or processing_time_duration.endswith("Min"): date = date[:12]
        
        print(prev_print, date, 'reading', ' '*10, sep=' ', end='\n')
        prev_print = '\x1B[1A\r'
        
        # load files
        # data = get_rawdata.open_flux(lookup=yl, **raw_kwargs).data
        info_t_startloaddata = time.time()
        data = hc24.loaddatawithbuffer(
            yl, d1=None, freq=_f, buffer=0, f_freq=_f, **raw_kwargs)
        if data.empty:
            if verbosity>1: logger.warning("UserWarning: No file was found ({}, path: {}).".format(date, raw_kwargs.get('path', 'default')))
            continue
        logger.info(f'\tLoading data took {round(time.time() - info_t_startloaddata)} s (run_cec).')
        
        # ensure time is time
        data.TIMESTAMP_ns = pd.to_datetime(data.TIMESTAMP)
        data.TIMESTAMP = data.TIMESTAMP_ns.dt.floor(_f)
        
        try:
            for v in [w, c, q]:
                data[v+'_'] = data.groupby('TIMESTAMP')[v].transform(np.nanmean)
            
            σcec = wavelet_functions.conditional_sampling((data[w]-data[f'{w}_'])*(data[c]-data[f'{c}_']),
                                        *[(data[w]-data[f'{w}_']), (data[c]-data[f'{c}_']), (data[q]-data[f'{q}_'])],
                                        names=['w', 'c', 'q'],
                                        label={1: "+", -1: "-"})
            data['wc'] = (data['w']-data['w_'])*(data['co2']-data['co2_'])
            data['fR'] = σcec['w+c+q+']
            data['fP'] = σcec['w+c-q+']

            σcec = wavelet_functions.conditional_sampling((data[w]-data[f'{w}_'])*(data[q]-data[f'{q}_']),
                                        *[(data[w]-data[f'{w}_']), (data[c]-data[f'{c}_']), (data[q]-data[f'{q}_'])],
                                        names=['w', 'c', 'q'],
                                        label={1: "+", -1: "-"})
            data['wq'] = (data['w']-data['w_'])*(data['h2o']-data['h2o_'])
            data['fE'] = σcec['w+c+q+']
            data['fT'] = σcec['w+c-q+']

            data = data[['TIMESTAMP', w, c, q, 'wc', 'wq', 'fR', 'fP', 'fE', 'fT']].groupby('TIMESTAMP').agg(np.nanmean).reset_index(drop=False)
            data['rFc'] = data['fR'] / data['fP']
            data['Rcec'] = data['wc'] / (1 + ( 1 / data['rFc'] ))
            data['Pcec'] = data['wc'] / (1 + data['rFc'] )
            data['rET'] = data['fE'] / data['fT']
            data['Ecec'] = data['wq'] / (1 + ( 1 / data['rET'] ))
            data['Tcec'] = data['wq'] / (1 + data['rET'] )
            
            __date__ = data['TIMESTAMP'].dt.floor(processing_time_duration).unique()[0]
            dst_path = output_path.format(suffix + "_cec", pd.to_datetime(__date__).strftime('%Y%m%d%H%M'))
            data.to_file(dst_path, index=False)
        except Exception as e:
            logger.critical(str(e))
            print(str(e))
        
        prev_print = '\x1B[2A\r' + f' {date} {len(yl)} files {int(100*ymd_i/len(ymd))} % ({time.strftime("%d.%m.%y %H:%M:%S")})' + '\n'
            
        logger.info(f'\tFinish {date} took {round(time.time() - info_t_startdateloop)} s, yielded {len(yl)} files (run_cec). Progress: {len(yl)} {int(100*ymd_i/len(ymd))} %')
    
    logger.info(f'Finish conditional EC (Zahn et al., 2022) at {round(time.time() - info_t_start)} s (run_cec).')
    return


def main(sitename, inputpath, outputpath, datetimerange, acquisition_frequency=20, fileduration=30, processduration='1D',
         **kwargs):
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

    # Select output file path
    configure.output_path = str(os.path.join(outputpath, 'processing', str(sitename)+'_CEC{}_{}.csv'))

    # Save args for run
    hc24.mkdirs(configure.output_path)
    with open(os.path.join(os.path.dirname(os.path.dirname(configure.output_path)), f'cec_{datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")}.setup'), 'w+') as stp:
        yaml.safe_dump(local_args, stp)

    # Select raw input file path
    # e.g.: "<PROJECT FOLDER>/eddypro_output/eddypro_raw_datasets/level_6/"
    if isinstance(inputpath, dict):
        configure.raw_kwargs = {'path': {k: str(os.path.join(v, 'eddypro_raw_datasets/level_6')) for k, v in inputpath.items()}}
    else: configure.raw_kwargs = {'path': str(os.path.join(inputpath, 'eddypro_raw_datasets/level_6'))}

    # Select period of interest
    # [START_DATE, END_DATE, FILE_FREQUENCY]
    configure.ymd = [datetimerange.split('-')[0], datetimerange.split('-')[1], f'{fileduration}min']

    # Averaging (and integrating) time
    configure.averaging = [fileduration]

    # Select dt
    configure.raw_kwargs.update({'fkwargs': {'dt': 1/acquisition_frequency},
                                 'fmt': {'co': '4th'}})

    # RUN WAVELET FLUX PROCESSING
    run_cec(**vars(configure), verbosity=5)


def __concat__(sitename, outputpath, **kwargs):
    # CONCAT INTO SINGLE FILE
    dst_path = os.path.join(outputpath, str(sitename)+f'_CEC_partition.csv')

    root = os.path.join(outputpath, 'processing')
    print(os.listdir(root))
    saved_files = {}
    for name in os.listdir(root):
        dateparts = re.findall('_([0-9]{12}).csv$', name, flags=re.IGNORECASE)
        if len(dateparts) == 1:
            saved_files[dateparts[0]] = os.path.join(root, name)

    data = pd.concat([pd.read_csv(v, sep=',') for k, v in saved_files.items()])

    if dst_path: data.to_file(dst_path, index=False)
    else: return data
    return


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
    parser.add_argument('--run', type=int, default=1)
    parser.add_argument('--concat', type=int, default=1)
    parser.add_argument('--processduration', type=str, default='1D')
    args = parser.parse_args()
    args = handle_eddypro_setup(**vars(args))
    
    args['sitename'] = args['sitename'].replace('/', '_').replace('\\', '_')
    
    args.pop('eddypro')
    args.pop('metadata')
    run = args.pop('run')
    concat = args.pop('concat')

    print('Start run w/')
    # replace os.get_cwd() for '' if str
    print('\n'.join([f'{k}:\t{v[:5] + "~" + v[-25:] if isinstance(v, str) and len(v) > 30 else v}' for k, v in args.items()]), end='\n\n')
    
    # Assert variables have been assigned
    missing_args = [f'`{k}`' for k in ['sitename', 'inputpath', 'outputpath', 'datetimerange', 'acquisition_frequency', 'fileduration'] if args[k] is None]
    assert len(missing_args) == 0, f'Missing argument in: {", ".join(missing_args)}.'

    if run: main(**args)
    if concat: __concat__(**args)