
# built-in modules
import re
import os
import warnings
from functools import reduce

# 3rd party modules
import numpy as np
import pandas as pd

# project modules
from .commons import get_all_sites
from . import corrections

def read_eddypro_metadata_file(filename):
    metadata = {}
    with open(filename, 'r') as file:
        section = None
        for line in file:
            line = line.strip()
            if line.startswith(';') or not line:
                continue  # Skip comments and empty lines
            if line.startswith('[') and line.endswith(']'):
                section = line[1:-1]  # Extract section name
                metadata[section] = {}
            else:
                key, value = line.split('=', 1)
                metadata[section][key.strip()] = value.strip()
    return metadata


def get_eddypro_output(sitename=None, **kwargs):
    if sitename is None:
        return get_all_sites(get_eddypro_output, **kwargs)
    mergeorconcat = kwargs.get('mergeorconcat', 'merge')
    folder        = kwargs.get('folder', 'data')
    
    ep_read_params = {'skiprows': [0,2], 'na_values': [-9999, 'NAN']}
    
    ep_path = os.path.join(folder, sitename, 'eddypro_output')

    files = {'FLUX': [], 'FLXNT': [], 'QCQA': [], 'META': []}
    for name in os.listdir(ep_path):
        if name.endswith('.csv'):
            if re.findall('_full_output_', name):
                if name.endswith('_adv.csv'):
                    files['FLUX'] += [pd.read_csv(os.path.join(ep_path, name), **ep_read_params)]
                else:
                    files['FLUX'] += [pd.read_csv(os.path.join(ep_path, name), na_values=[-9999, 'NAN'])]
            elif re.findall('_fluxnet_', name):
                flxnt_last = pd.read_csv(os.path.join(ep_path, name), na_values=[-9999, 'NAN'])
                flxnt_last["date"] = pd.to_datetime(flxnt_last["TIMESTAMP_START"], format='%Y%m%d%H%M').dt.strftime('%Y-%m-%d')
                flxnt_last["time"] = pd.to_datetime(flxnt_last["TIMESTAMP_START"], format='%Y%m%d%H%M').dt.strftime('%H:%M')
                files['FLXNT'] += [flxnt_last]
                del flxnt_last
            elif re.findall('_qc_details_', name):
                if name.endswith('_adv.csv'):
                    files['QCQA'] += [pd.read_csv(os.path.join(ep_path, name), **ep_read_params)]
                else:
                    files['QCQA'] += [pd.read_csv(os.path.join(ep_path, name), na_values=[-9999, 'NAN'])]
            elif re.findall('_metadata_', name):
                files['META'] += [pd.read_csv(os.path.join(ep_path, name), na_values=[-9999, 'NAN'])]
    
    for k in [k for k in list(files.keys()) if not files[k]]:
        del([files[k]])

    for k in files.keys():
        if len(files[k]) == 1:
            files[k] = files[k][0]
        elif mergeorconcat == 'concat':
            files[k] = pd.concat(files[k])
        else:
            files[k] = reduce(lambda left, right: pd.merge(
                left, right, on=["date", 'time'], how="outer", suffixes=('', '_DUP')), files[k])
    
    data = pd.DataFrame(files.pop('FLUX', {}))
    for name, dat in files.items():
        data = pd.merge(data, dat, on=["date", 'time'], how="outer", suffixes=('', f'_{name}'))
    
    data['TIMESTAMP'] = pd.to_datetime(data.date + ' ' + data.time)#.dt.tz_localize('UTC')
    
    # Despike because open path analysers can be noisy
    data['air_molar_volume_despiked'] = corrections.mauder2013(data.air_molar_volume, 5)
    data['Vd'] = (data.air_molar_volume_despiked * data.air_pressure /
                           (data.air_pressure - data.e))
    # make wet from dry
    data['Va'] = (data.air_molar_volume_despiked * data.air_pressure /
                           (data.air_pressure - data.e))
    return data


def get_eddypro_cospectra(sitename=None, x='natural_frequency', y='f_nat*cospec(w_co2)/cov(w_co2)', folder='data', subfolder='', help=False):
    assert (x is not None and y is not None) or (help)
    if sitename is None:
        return get_all_sites(get_eddypro_cospectra, x=x, y=y, folder=folder, subfolder=subfolder, help=help)
    
    ep_path = os.path.join(folder, sitename, 'output/eddypro_output', subfolder, 'eddypro_binned_cospectra')
    
    if not os.path.exists(ep_path): return None
    files = []
    for name in os.listdir(ep_path):
        if name.endswith('.csv'):
            if re.findall('_binned_cospectra_', name):
                binned_cosp = pd.read_csv(os.path.join(ep_path, name), skiprows=11, na_values=[-9999, 'NAN'])
                if help: 
                    print(binned_cosp.columns)
                    return
                if x not in binned_cosp.columns or y not in binned_cosp.columns: continue
                binned_cosp.dropna(subset=[x], inplace=True)
                binned_cosp = binned_cosp[[x, y]]
                binned_cosp['TIMESTAMP'] = name.split('_')[0]
                binned_cosp = binned_cosp.pivot(index='TIMESTAMP', columns=x, values=y).reset_index(drop=False)
                binned_cosp.columns = [c for c in binned_cosp.columns]
                files += [binned_cosp]
    
    if len(files) == 1:
        data = files[0]
    else:
        data = pd.concat(files)
    
    data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP'], format='%Y%m%d-%H%M')
    return data