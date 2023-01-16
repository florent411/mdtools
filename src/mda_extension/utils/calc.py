#!/usr/bin/env python 

''' Calculate order parameters '''

import numpy as np
import pandas as pd

from tqdm import tqdm
import checkarg.list as Guard
from MDAnalysis.analysis import rms, align

import time

# Included submodules
import mda_extension.utils.tools as tools
import mda_extension.utils.mp_functions as mpf

def rmsd(universes,
         labels=None,
         selection='protein and name CA'):
    """Calculate root-mean square deviation (rmsd)
    
    :param universes: list of the universes to analyse
    :param labels: list of corresponding labels. If only one value is given, use it as a prefix. If empty just use numbers 0...n as labels. (Default value = None)
    :param selection: proteinselection on which to align before running rmsd. (Default value = 'protein and name CA')

    :returns: df containing columns 'time', 'rmsd' and 'origin'.
    """

    # Preprocess universes and labels.
    # Turn into lists and make the labels fit the universes.
    universes, labels = tools.prepare_ul(universes, labels)

    # Check length of the list
    Guard.is_length_equals(labels, len(universes))

    df_list = []
    for index, universe in tqdm(enumerate(universes), total=len(universes), desc='Universes', position=0):
        R = rms.RMSD(universe, select=selection).run().results.rmsd
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
    """Calculate radius of gyration (R_g)
    
    :param universes: list of the universes to analyse
    :param labels: list of corresponding labels. If only one value is given, use it as a prefix. If empty just use numbers 0...n as labels. (Default value = None)

    :returns: df containing columns 'time', 'rg' and 'origin'.
    """

    # Preprocess universes and labels.
    # Turn into lists and make the labels fit the universes.
    universes, labels = tools.prepare_ul(universes, labels)

    # Check length of the lists
    Guard.is_length_equals(labels, len(universes))

    df_list = []
    for index, universe in tqdm(enumerate(universes), total=len(universes), desc='Universes', position=0):
        protein = universe.select_atoms("protein")

        Rgyr = np.array([(universe.trajectory.time, protein.radius_of_gyration()) for ts in universe.trajectory])

        rg = pd.DataFrame(np.array(Rgyr), columns = ['time', 'rg'])
        
        rg['origin'] = labels[index]

        df_list.append(rg)

    df = pd.concat(df_list, ignore_index=True)

    # Convert from AA to nm
    df['rg'] /= 10
    
    return df

def rmsf(universes, labels=None, selection='protein and name CA'):
    """Calculate root-mean square fluctuation per residue (rmsf)
    
    :param universes: list of the universes to analyse
    :param labels: list of corresponding labels. If only one value is given, use it as a prefix. If empty just use numbers 0...n as labels. (Default value = None)
    :param selection: proteinselection on which to align before running rmsd. (Default value = 'protein and name CA')

    :returns: df containing columns 'resid', 'rmsf' and 'origin'.
    """

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

