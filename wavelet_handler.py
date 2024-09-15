# Project modules
from waveletec._core import wavelet_functions as wavelet_functions
from waveletec.handler import main, __concat__, __partition__, handle_eddypro_setup


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
    args['sitename'] = args['sitename'].replace('/', '_').replace('\\', '_')

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

    #with open(args['outputpath']+f'/wavelet_processing_{datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")}.py', 'w+') as p:
    #    p.write('python wavelet_handler.py ' + 
    #            ''.join([f'--{k} {" ".join(v)}' if isinstance(v, list) else f'--{k} {v}' for k, v in args.items()]))

    if run: main(**args)
    if concat: __concat__(**args)
    if partition: __partition__(**args)
