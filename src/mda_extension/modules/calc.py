#!/usr/bin/env python 

''' Calculate order parameters '''

import numpy as np
import pandas as pd

from tqdm import tqdm
import checkarg.list as Guard
from MDAnalysis.analysis import rms, align

import time

# Included submodules
import mda_extension.modules.tools as tools
import mda_extension.modules.mp_functions as mpf

def rmsd(universes, labels=None, selection='protein and name CA'):
    ''' Calculate root-mean square deviation (rmsd) 
    
        Inputs:
        --> universes: list of universes.
        (--> labels: list of corresponding labels. If only one value is given, use it as a prefix. If empty just use numbers 0...n as labels.)
        (--> selection: proteinselection on which to align before running rmsd)'''

    # Preprocess universes and labels.
    # Turn into lists and make the labels fit the universes.
    universes, labels = tools.prepare_ul(universes, labels)

    # Check length of the list
    Guard.is_length_equals(labels, len(universes))

    df_list = []
    for index, universe in tqdm(enumerate(universes), total=len(universes), desc='Universes', position=0):
        R = rms.RMSD(universe, select="name CA").run().results.rmsd
        rmsd = pd.DataFrame(R, columns = ['frame','time','rmsd'])
        rmsd['origin'] = labels[index]

        # Drop the frame column, as it is not relevant.
        rmsd.drop('frame', axis=1, inplace=True)

        df_list.append(rmsd)

    df = pd.concat(df_list, ignore_index=True)

    # Convert from AA to nm
    df['rmsd'] /= 10

    return df

def rg(universes, labels=None):
    ''' Calculate radius of gyration (R_g)
    
        Inputs:
        --> universes: list of universes.
        (--> labels: list of corresponding labels. If only one value is given, use it as a prefix. If empty just use numbers 0...n as labels.)'''

    # Preprocess universes and labels.
    # Turn into lists and make the labels fit the universes.
    universes, labels = tools.prepare_ul(universes, labels)

    # Check length of the lists
    Guard.is_length_equals(labels, len(universes))

    df_list = []
    for index, universe in tqdm(enumerate(universes), total=len(universes), desc='Universes', position=0):
        protein = universe.select_atoms("protein")

        Rgyr = np.array([(universe.trajectory.time, protein.radius_of_gyration()) for ts in tqdm(universe.trajectory, total=len(universe.trajectory), desc="Rg", position=1)])

        rg = pd.DataFrame(np.array(Rgyr), columns = ['time', 'rg'])
        
        rg['origin'] = labels[index]

        df_list.append(rg)

    df = pd.concat(df_list, ignore_index=True)

    # Convert from AA to nm
    df['rg'] /= 10
    
    return df

def rmsf(universes, labels=None, selection='protein and name CA'):
    ''' Calculate root-mean square fluctuation per residue (rmsf) 
    
        Inputs:
        --> universes: list of universes.
        (--> labels: list of corresponding labels.)
        (--> selection: proteinselection on which to align before running rmsf)'''

    # Preprocess universes and labels.
    # Turn into lists and make the labels fit the universes.
    universes, labels = tools.prepare_ul(universes, labels)

    # Check length of the list
    Guard.is_length_equals(labels, len(universes))

    df_list = []
    for index, universe in tqdm(enumerate(universes), total=len(universes), desc='Universes', position=0):
        # rms.RMSF does not allow on-the-fly alignment to a reference, and presumes that you have already aligned the trajectory. 
        # Therefore we need to first align our trajectory to the average conformation.
        average = align.AverageStructure(universe, universe, select=selection, ref_frame=0).run()
        ref = average.results.universe

        align.AlignTraj(universe, ref, select=selection, in_memory=True).run()

        # Now we calculate the actual RMSF
        R = rms.RMSF(universe.select_atoms(selection)).run().results.rmsf
        res_ids = [*range(1, len(R) + 1)]
        rmsf = pd.DataFrame({'resid' : res_ids, 'rmsf' : R})
        rmsf['origin'] = labels[index]

        df_list.append(rmsf)

    df = pd.concat(df_list, ignore_index=True)

    # Convert from AA to nm
    df['rmsf'] /= 10

    return df

def weights(colvar, T):
    ''' Calculate the weights corresponding to each frame in the trajectory. 
        These weights can be used for reweighting other cvs. 
        
        Inputs:
        --> colvar: the dataframe created from the COLVAR file.
        --> T: temperature in K'''

    kb = 1.38064852e-23 # Boltzman's constant in m^2 kg s^-2 K^-1 or J K^-1
    NA = 6.02214086e23 # Avogadro's constant in mol^-1

    # Get the bias values as a unitless factor.
    # T is the temperature
    bias = colvar['opes.bias'].values / (kb * NA * T / 1000)

    # Calculate weights 
    weights = np.exp(bias - np.amax(bias))

    # Determine corresponding time, frame and walker in the trajectory
    time = colvar['time'].values
    origin = colvar['origin'].values

    # Calculate the effective sample size (not yet used for anything)
    effsize = np.sum(weights)**2 / np.sum(weights**2)

    df = pd.DataFrame({'time' : time, 'weights' : weights, 'origin' : origin})
        
    return df