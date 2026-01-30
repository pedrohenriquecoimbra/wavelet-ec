# standard modules
import os
import sys
import re
import logging
import datetime
import argparse
# 3rd party modules
import yaml
# Project modules
from . import core as hc24
from .extra import eddypro as eddypro
from .core import handlers


logger = logging.getLogger(__name__)


def set_cwd(workingdir=None):
    """
    Set the current working directory to the specified path.
    
    Parameters:
    - workingdir: Path to the directory to set as current working directory.
    """
    if workingdir is not None:
        os.chdir(workingdir)
        print(f"Current working directory set to: {os.getcwd()}")
        logger.info(f"Current working directory set to: {os.getcwd()}")
    else:
        logger.debug("No working directory specified, using current directory.")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-ep', '--eddypro',   type=str,
                        help='Path to EddyPro setup file')
    parser.add_argument('-m', '--metadata',   type=str,
                        help='Path to EddyPro metadata file')
    parser.add_argument('-s', '--site_name',   type=str,
                        help='Site name for the processing')
    parser.add_argument('-i', '--input_path',  type=str,
                        help='Path to the input data folder')
    parser.add_argument('-o', '--output_folderpath', type=str,
                        help='Path to the output folder where results will be saved')
    parser.add_argument('-d', '--datetimerange', type=str,
                        help='Date and time range for processing, format: YYYYMMDD-HHMM-YYYYMMDD-HHMM')
    parser.add_argument('-af', '--acquisition_frequency', type=int, default=20,
                        help='Acquisition frequency in Hz (default: 20 Hz)')
    parser.add_argument('-fd', '--fileduration', type=int, default=30,
                        help='File duration in minutes (default: 30 minutes)')
    parser.add_argument('-ip', '--integration_period', type=int, default=None,
                        help='Integration period in seconds (default: None)')
    parser.add_argument('-v', '--variables_available', type=str, nargs='+',
                        help='List of available variables (default: ["u", "v", "w", "ts", "co2", "h2o"])')
    # parser.add_argument('-dk', '--despike', type=int)  # , nargs=1)
    # parser.add_argument('-dn', '--denoise', type=int)  # , nargs=1)
    # parser.add_argument('-db', '--deadband', type=str, nargs='+')
    parser.add_argument('-cov', '--covariance', type=str, nargs='+',
                        help='List of covariances to compute (default: None)')
    parser.add_argument('--method', type=str, default='dwt', choices=['cov', 'dwt', 'cwt'],
                        help='Method to use for wavelet processing (default: "dwt")')
    parser.add_argument('--wave_mother', type=str, default='db6',
                        help='Mother wavelet to use for wavelet processing (default: "db6")')
    parser.add_argument('--run', type=int, default=1,
                        help='Run the wavelet processing (default: 1)')
    parser.add_argument('--concat', type=int, default=1,
                        help='Concatenate results into a single file (default: 1)')
    parser.add_argument('--partition', type=int, default=1,
                        help='Run partitioning on the results (default: 1)')
    parser.add_argument('--processing_time_duration', type=str, default='1D',
                        help='Processing time duration for the wavelet processing (default: "1D")')
    # parser.add_argument('--preaverage', type=str, default=None)
    parser.add_argument('-cwd', '-workingdir', type=str, default=None,
                        help='Set the current working directory (default: None)')
    parser.add_argument('--log_level', type=str, default='INFO', choices=[
                        'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level')

    args = parser.parse_args()
    return vars(args)


def prepare_args(args):
    # Set the current working directory if specified
    if args.get('cwd'):
        set_cwd(os.chdir(args['cwd']))

    # Set logging level
    log_level = getattr(logging, args.pop(
        'log_level', 'INFO').upper(), logging.INFO)
    args['logging_kwargs'] = {'level': log_level}

    # Extract and pop run flags
    run = args.pop('run', 1)
    concat = args.pop('concat', 1)
    partition = args.pop('partition', 1)
    ep_setup = args.pop('eddypro')
    ep_meta = args.pop('metadata')

    # Retrieve eddypro setup
    # exta = eddypro.extract_info_from_eddypro_setup(ep_setup, ep_meta)
    # args = eddypro.update_args_with_extracted_info(args, exta)

    # Default integration period
    logger.debug(
        f"Integration period {args['integration_period']}, setting to {args['fileduration']} minutes.")
    if args['integration_period'] is None:
        args['integration_period'] = args['fileduration'] * 60

    # Sanitize site name
    args['site_name'] = args['site_name'].replace('/', '_').replace('\\', '_')

    # Convert paths
    for path in ['input_path', 'output_folderpath', 'eddypro', 'metadata']:
        if args.get(path):
            args[path] = os.path.abspath(args[path])

    # Return args and control flags
    return args, run, concat, partition, ep_setup


def validate_args(args):
    required = ['site_name', 'input_path', 'output_folderpath',
                'datetimerange', 'acquisition_frequency', 'fileduration']
    missing = [f'`{k}`' for k in required if args.get(k, None) is None]
    if missing:
        raise ValueError(f'Missing argument(s): {", ".join(missing)}')


def log_args(args):
    # print('\n'.join(
    #     [f'{k}:\t{v[:5] + "~" + v[-25:] if isinstance(v, str) and len(v) > 30 else v}' for k, v in args.items()]), end='\n\n')

    print('Start run with:')
    for k, v in args.items():
        if isinstance(v, str) and len(v) > 30:
            v = f"{v[:5]}~{v[-25:]}"
        print(f"{k}:\t{v}")
    print()


# consider local folder + "/wavelet_flux" as default
# 

def __custom_params__(unknown_args):
    def convert_to_number(value):
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value

    # Print or process unknown arguments
    custom_params = {}
    i = 0
    while i < len(unknown_args):
        logger.debug(f"Custom/unknown arguments: {unknown_args}")
        # You can store or process these as needed, e.g.:
        arg = unknown_args[i]
        if arg.startswith("--"):
            key = arg[2:]
            # Check if the next argument is a value (not starting with '--')
            if i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith("--"):
                custom_params[key] = convert_to_number(unknown_args[i + 1])
                i += 1  # Skip the next argument as it's the value
            else:
                custom_params[key] = True  # Flag argument
        else:
            # Handle values for the previous key if needed
            custom_params[key].append(convert_to_number(arg))
            pass
        i += 1
    return custom_params


def eddypro_run():
    # waveletEC-eddypro_run -p "input/EP/FR-Gri_sample.eddypro" -o "output/wavelet_flux/" -d "20220513T0000-20220516T0000" --processing_time_duration 3H
    parser = argparse.ArgumentParser(description="Run wavelet-based edy covariance workflows.")     
    parser.add_argument('-p', '--path',   type=str,
                        help='Path to EddyPro setup file')
    parser.add_argument('-f', '--folder', type=str,
                        help='Path to the output folder where results will be saved')
    parser.add_argument('-d', '--datetimerange', type=str,
                        help='Date and time range for processing, format: YYYYMMDDTHHMM-YYYYMMDDTHHMM') 
    parser.add_argument('-cov', '--covariance', type=str, nargs='+',
                        help='List of covariances to compute (e.g.: "w*co2", default: None)')
    parser.add_argument("--verbosity", default="INFO", choices=[
                        'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help="Logging level (e.g., DEBUG, INFO, WARNING)")
    # Parse known arguments and capture the rest
    args, unknown_args = parser.parse_known_args()

    # Validate logging level
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if args.verbosity.upper() not in valid_levels:
        logger.error(
            f"Invalid verbosity level. Choose from: {valid_levels}")
        args.verbosity = "0"

    try:
        custom_params = __custom_params__(unknown_args)
    except UnboundLocalError:
        raise(SyntaxError, 'Check command. Possibly kwargs error, kwargs must be passed as `--key value`.')

    logging.basicConfig(level=args.verbosity.upper(),
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    passed_args = {"path": args.path,
                   "output_folderpath": args.folder,
                   "datetimerange": args.datetimerange,
                   "covariance": args.covariance,
                   **custom_params}
    log_args(vars(args))
    handlers.run_from_eddypro(**passed_args)


def integrate():
    parser = argparse.ArgumentParser(
        description="Run wavelet-based edy covariance workflows.")
    parser.add_argument('-f', '--folder', type=str,
                        help='Path to the output folder where results will be saved')
    parser.add_argument('-d', '--dst_path', type=str,
                        help='Path to the output file where results will be saved')    
    parser.add_argument('-ip', '--integration_period', type=int, default=None,
                        help='Integration period in seconds (default: None)')
    parser.add_argument("--verbosity", default="INFO", choices=[
                        'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help="Logging level (e.g., DEBUG, INFO, WARNING)")
    # Parse known arguments and capture the rest
    args, unknown_args = parser.parse_known_args()

    args.dst_path = args.dst_path or os.path.join(
        args.folder, '00000_full_cospectra.csv')

    # Validate logging level
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if args.verbosity.upper() not in valid_levels:
        logger.error(
            f"Invalid verbosity level. Choose from: {valid_levels}")
        args.verbosity = "0"

    custom_params = __custom_params__(unknown_args)

    logging.basicConfig(level=args.verbosity.upper(),
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    passed_args = {"root": os.path.join(args.folder, 'wavelet_full_cospectra'),
                   "f0": 1/args.integration_period,
                   "dst_path": args.dst_path,
                   **custom_params}
    log_args(vars(args))
    handlers.integrate_cospectra_from_file(**passed_args)
    
def partition():
    parser = argparse.ArgumentParser(
        description="Run wavelet-based edy covariance workflows.")
    parser.add_argument('-f', '--folder', type=str,
                        help='Path to the output folder where results will be saved')
    parser.add_argument("--verbosity", default="INFO", choices=[
                        'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help="Logging level (e.g., DEBUG, INFO, WARNING)")
    # Parse known arguments and capture the rest
    args, unknown_args = parser.parse_known_args()

    # Validate logging level
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if args.verbosity.upper() not in valid_levels:
        logger.error(
            f"Invalid verbosity level. Choose from: {valid_levels}")
        args.verbosity = "0"

    custom_params = __custom_params__(unknown_args)

    logging.basicConfig(level=args.verbosity.upper(),
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    passed_args = {"folder": args.folder,
                   **custom_params}
    log_args(vars(args))
    handlers.condition_sampling_partition(**passed_args)


def exec():
    pass

def main(**args):
    # waveleEC --setup "path/to/.eddypro"
    args, run, concat, partition, ep_setup = prepare_args(args)
    validate_args(args)
    log_args(args)

    if args['method'] == 'cov':
        concat = partition = False

    if run:
        handlers.run_from_eddypro(ep_setup, **args)
    # if concat:
    #     integrate_full_spectra_into_file(**args)
    if partition:
        handlers.condition_sampling_partition(**args)


if __name__ == '__main__':
    args = parse_args()
    main(**args)
