#!/usr/bin/env python3

from http.client import GATEWAY_TIMEOUT
from re import A
import os
import sys
import numpy as np
import pandas as pd
import polars as pl
from itertools import islice
pd.options.mode.chained_assignment = None  # default='warn'

# Helpfull home-made modules
import modules.tools as tools

# Other constants
kb = 1.38064852e-23 # Boltzman's constant in m^2 kg s^-2 K^-1
NA = 6.02214086e23 # Avogadro's constant in mol^-1

def read_state_file(states_file="STATE"):
    ''' Read states file and modify data to fit into two dataframes. 
    1) states_data (dataframe containing the states data)
    2) states_info (containing the extra information, such as zed value/biasfactor etc.'''

    df = pd.read_csv(states_file, delim_whitespace=True, low_memory=False)

    # Extracting all states from the dataframe and set the time as the index value
    states_data = df[~df.iloc[:,1].isin(['SET', 'FIELDS'])]
    states_data.columns = states_data.columns[2:].tolist() + 2 * ['na']
    states_data = states_data.dropna(axis=1).astype({"time": int})
    
    # Getting additional information (all variables starting with #! SET)
    states_info = df[df.iloc[:,1] == 'SET'].iloc[:,2:4]
    states_info.columns = ['variable', 'value']
    g = states_info.groupby(['variable']).cumcount()

    # Pivot table. Now the unique variables are the column names
    states_info = (states_info.set_index([g, 'variable'])['value']
                   .unstack(fill_value=0)
                   .reset_index(drop=True)
                   .rename_axis(None, axis=1))

    # Add a column for the time values.
    states_info['time'] = states_data['time'].unique()

    # # Convert to polars dataframe
    # states_info = pl.from_pandas(states_info)
    # states_data = pl.from_pandas(states_data)

    return states_data, states_info

def setup_grid_settings(cvs, grid_min, grid_max, grid_bin):
    ''' Make small dataframe containing all grid_info for each cv.'''

    # For grid_min and grid_max if no value given, make an array of nan, if only one value is given, use it for all dimensions, otherwise use the user input.
    if grid_min == None:
        grid_min = np.empty(len(cvs))
        grid_min[:] = np.nan
    elif len(grid_min) == 1:
        grid_min = np.repeat(grid_min, len(cvs))
    elif len(grid_min) != len(cvs):
        sys.exit(f"ERROR: Number of grid_min values ({len(grid_min)}, i.e. {grid_min} ) not equal to number dimensions ({len(cvs)}).\nComma separated integers expected after --min\nIf 1 value is given, it is used for all dimensions.\nNow exiting")

    if grid_max == None:
        grid_max = np.empty(len(cvs))
        grid_max[:] = np.nan
    elif len(grid_max) == 1:
        grid_max = np.repeat(grid_max, len(cvs))
    elif len(grid_max) != len(cvs):
        sys.exit(f"ERROR: Number of grid_max values ({len(grid_max)}, i.e. {grid_max} ) not equal to number dimensions ({len(cvs)}).\nComma separated integers expected after --max\nIf 1 value is given, it is used for all dimensions.\nNow exiting")

    # For grid_bin, if only one value is given, use it for all dimensions. Otherwise, use the user input.
    if len(grid_bin) == 1:
        n_bins = np.repeat(grid_bin, len(cvs)) # repeat the bin value n times
    elif len(grid_bin) != len(cvs):
        sys.exit(f"ERROR: Number of grid_bin values ({len(grid_bin)}, i.e. {grid_bin} ) not equal to number dimensions ({len(cvs)}).\nComma separated integers expected after --bin\nIf 1 value is given, it is used for all dimensions.\nNow exiting")
    else:
        n_bins = grid_bin

    # The + 1 comes from the original invemichele script. He says its: #same as plumed sum_hills
    # I'm not sure why an extra bin is added.
    n_bins = [x + 1 for x in n_bins]

    # Make small dataframe containing all grid_info for each cv 
    return pd.DataFrame(np.vstack((grid_min, grid_max, n_bins)), columns=cvs, index=['grid_min', 'grid_max', 'n_bins'])

def find_sigmas(f, type):
    ''' See if you can find the sigma values in the first n lines of a file.'''
    
    # Look in KERNELS file
    if type == 'kernels':
        with open(f, 'rb') as f:

            # Get the first line for the keys
            first_line = f.readline().decode("utf-8")

            try:  # catch OSError in case of a one line file 
                f.seek(-2, os.SEEK_END)
                while f.read(1) != b'\n':
                    f.seek(-2, os.SEEK_CUR)
            except OSError:
                f.seek(0)
            
            # Find the last line for the values.
            last_line = f.readline().decode()
            
            # Zip into dictionary. [2:] is used to skip the #! and FIELDS values in the first line.
            # From CPython 3.6 onwards, and Python 3.7 onwards, regular dicts are sorted by insertion order, so you can use dict here instead of OrderedDict if you know your code will run under a suitable version.
            result = dict(zip(first_line.split()[2:], last_line.split()))
    
            # Return a dict containing the names and values of the cvs and their sigmas.
            sigmas = {k.split('_')[-1] : float(v) for k, v in result.items() if 'sigma' in k}
       
        return sigmas

    # Look in STATES file
    elif type == 'states':
        sigmas = {}
        with open(f) as fp:
            while True:
                # Read lines in chunks of 100. Otherwise with large STATES files, you get issues.
                next_n_lines = list(islice(fp, 100))
                if not next_n_lines:
                    break
                else:
                    for line in next_n_lines:
                        if not line.startswith('#!'):
                            break
                        elif 'sigma0' in line:
                            sigmas[line.split()[-2].split('_')[-1]] = float(line.split()[-1])
                    
                    return sigmas

    else:
        # Else result empty dict, which will result in an error.
        sys.exit(f"ERROR: Unknown type {type}. Can't find sigmas here.")

def calc_conv(fes_df, unitfactor, split_fes_at, calc_fes_from, fmt, write_output=True):
    ''' Calculate different convergence metrics given a free energy landscape.'''

    time_list = fes_df['time'].unique()

    # Reference distribution (p or v1)
    reference = fes_df[fes_df['time'] == fes_df['time'].unique()[-1]]
    
    # cvs
    cvs = [ x for x in reference.columns.values.tolist() if x not in ['time', 'fes']]
    print(f"\t\tcvs: {' and '.join(cvs)}")   

    kldiv_values, jsdiv_values, dalonso_values, dfe_values = [], [], [], []

    for index, time in enumerate(time_list):
        print(f"\t\tWorking on state {(index + 1)} of {len(time_list)}\t| {((index + 1)*100.0)/len(time_list):.1f}%", end='\r')

        # Current distribution (q or v2)
        current = fes_df[fes_df['time'] == time]

        # Free energy estimates
        ref_fe = reference['fes'].values
        cur_fe = current['fes'].values

        # Corresponding probability distributions
        ref = np.exp(-ref_fe / unitfactor)
        cur = np.exp(-cur_fe / unitfactor)

        # Normalized probability distributions
        ref_norm = ref / np.sum(ref)
        cur_norm = cur / np.sum(cur)

        # To adjust for large arear where q = 0, a Bayesian smoothing function is employed. Here a "simulation" is performed of N ideal steps, using the FE from sampling.
        # The new adjusted probability for each of the bins is then (1 + Pi * N) / (M + N), where M is the total number of bins.
        # N is chosen to be big enough to turn 0 values into very small values, without risking python not being able to handle the values.
        # This effect is similar as adding very small values to all gridpoints.
        N, M = 1e9, len(ref_norm) 
        ref_norm_smooth = (N * ref_norm + 1) / (N + M)
        cur_norm_smooth = (N * cur_norm + 1) / (N + M)
            
        # Kullback-Leibler divergence
        kldiv_values.append(tools.kldiv(ref_norm_smooth, cur_norm_smooth))

        # Jensen–Shannon divergence
        jsdiv_values.append(tools.jsdiv(ref_norm_smooth, cur_norm_smooth))

        # Alonso & Echenique metric
        dalonso_values.append(tools.dalonso(ref_norm_smooth, cur_norm_smooth))

        # DeltaFE
        # NB: summing is as accurate as trapz, and logaddexp avoids overflows
        cv = cvs[0]
        fesA = -unitfactor * np.logaddexp.reduce(-1/unitfactor * current[current[cv] < split_fes_at]['fes'].values)
        fesB = -unitfactor * np.logaddexp.reduce(-1/unitfactor * current[current[cv] > split_fes_at]['fes'].values)
        deltaFE = fesB - fesA

        dfe_values.append(deltaFE)
        
    print(f"\t\tWorking on state {(index + 1)} of {len(time_list)}\t| {((index + 1)*100.0)/len(time_list):.1f}%")

    # Dataframe from lists
    conv_df = pd.DataFrame({'time': time_list, 'KLdiv': kldiv_values, 'JSdiv': jsdiv_values, 'dA': dalonso_values, 'deltaFE': dfe_values})
    conv_df.to_csv(f"conv_{'_'.join(cvs)}_{calc_fes_from}.dat", index=False, sep='\t', float_format=fmt)

    print(f"\t\t--> Outputfile: conv_{'_'.join(cvs)}_{calc_fes_from}.dat")           

    return conv_df