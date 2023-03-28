#!/usr/bin/env python3

import os
import sys
import torch
import time
from datetime import timedelta
from tqdm.auto import tqdm
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

# Helpfull home-made modules
import opes_postprocessing.utils.tools as tools

# Other constants
kb = 1.38064852e-23 # Boltzman's constant in m^2 kg s^-2 K^-1
NA = 6.02214086e23 # Avogadro's constant in mol^-1


def conv_params(df, units='kT', temp=310, split_fes_at=None, verbose=True):
    """Calculate convergence parameters, i.e. KLdiv, JSdiv, dAlonso and deltaFE.
    
    :param df: Dataframe containing the free energy data for n cv's over time. Needed columns are at least: 'fes', 'time'
    :param units: Energy units. Must be 'kT', 'kJ/mol' or 'kcal/mol'.
    :param temp: Simulation temperature in degrees Kelving.
    :param split_fes_at: For delatFE.
    :param verbose: Allow printing.

    :returns: df containing columns 'kldiv', 'jsdiv', 'dalonso' and 'dfe'.
    """

    # Check that the dataframe contains at least time and fes columns.
    assert set(['time', 'fes']).issubset(df.columns),\
            f"Dataframe should contain at least 'time' and 'fes' columns.\nOnly found the following columns: {df.columns.values}"

    # Calculate unitfactor (units conversion factor)
    unitfactor = tools.get_unitfactor(units, temp)

    # Get list of all seperate times at which the fe was calculated.
    time_list = df['time'].unique()

    # Reference distribution (p or v1)
    reference = df[df['time'] == df['time'].unique()[-1]]
    
    # Find cvs
    cvs = [x for x in reference.columns.values.tolist() if x not in ['time', 'fes']]
    
    # You can assert list to confirm list is not empty
    assert list, f"No CVs found.\nOnly found the following column(s): {df.columns.values}"
    
    print(f"Found the following CVs: {' and '.join(cvs)}") if verbose else 0

    kldiv_values, jsdiv_values, dalonso_values, dfe_values = [], [], [], []

    for time in tqdm(time_list, total=len(time_list), desc='comparisons', position=0):

        # Current distribution (q or v2)
        current = df[df['time'] == time]

        # Free energy estimates
        ref_fe = reference['fes'].values
        cur_fe = current['fes'].values

        # Corresponding probability distributions
        ref = np.exp(-ref_fe / unitfactor)
        cur = np.exp(-cur_fe / unitfactor)

        # Normalized probability distributions
        ref_norm = ref / np.sum(ref)
        cur_norm = cur / np.sum(cur)

        # To adjust for large area where q = 0, a Bayesian smoothing function is employed. Here a "simulation" is performed of N ideal steps, using the FE from sampling.
        # The new adjusted probability for each of the bins is then (1 + Pi * N) / (M + N), where M is the total number of bins.
        # N is chosen to be big enough to turn 0 values into very small values, without risking python not being able to handle the values.
        # This effect is similar to adding very small values to all gridpoints.
        N, M = 1e9, len(ref_norm) 
        ref_norm_smooth = (N * ref_norm + 1) / (N + M)
        cur_norm_smooth = (N * cur_norm + 1) / (N + M)
            
        # Kullback-Leibler divergence
        kldiv_values.append(tools.kldiv(ref_norm_smooth, cur_norm_smooth))

        # Jensen–Shannon divergence
        jsdiv_values.append(tools.jsdiv(ref_norm_smooth, cur_norm_smooth))

        # Alonso & Echenique metric
        dalonso_values.append(tools.dalonso(ref_norm_smooth, cur_norm_smooth))

        # # DeltaFE
        # # NB: summing is as accurate as trapz, and logaddexp avoids overflows
        # cv = cvs[0]
        # fesA = -unitfactor * np.logaddexp.reduce(-1/unitfactor * current[current[cv] < split_fes_at]['fes'].values)
        # fesB = -unitfactor * np.logaddexp.reduce(-1/unitfactor * current[current[cv] > split_fes_at]['fes'].values)
        # deltaFE = fesB - fesA

        # dfe_values.append(deltaFE)

    # Turn lists into dataframes.
    # conv_params_df = pd.DataFrame({'time': time_list, 'KLdiv': kldiv_values, 'JSdiv': jsdiv_values, 'dA': dalonso_values, 'deltaFE': dfe_values})
    conv_params_df = pd.DataFrame({'time': time_list, 'kldiv': kldiv_values, 'jsdiv': jsdiv_values, 'dalonso': dalonso_values})

    return conv_params_df