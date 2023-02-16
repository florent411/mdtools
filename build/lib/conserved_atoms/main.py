#!/usr/bin/env python3

from MDAnalysis.analysis import align as align
import numpy as np
import pandas as pd
from scipy import stats

def calc_density(u,
                 pocket_definition='point 0.0 0.0 0.0 15',
                 target='name OH2',
                 align_on='name CA',
                 verbose=True,
                 ):
    """
    Calculate the density or a certain atom type (atom_name) throughout a simulation.
    
    :param u: MDAnalysis universe, including a trajectory.
    :param pocket_definition: How to define the binding pocket or area in which to calculate density. (See par 3.1.4 on https://docs.mdanalysis.org/stable/documentation_pages/selections.html)
    :param atom_name: What is the name of the atom you want to calculate the density for in the trajectory. (OH2 is the oxygen atom of the water molecules)
    :param align: Align the trajectory on the atoms in the binding pocket.
    :param verbose: Allow printing

    :return: df containing the coordinates and density of all atoms in selection.
    """


    # Align trajectory 
    if align_on:
        align.AlignTraj(u,  # trajectory to align (mobile)
                        u,  # reference (ref)
                        select=f"({align_on}) and {pocket_definition}",  # selection of atoms to align
                        match_atoms=True,  # whether to match atoms based on mass
                        in_memory=True # Inplace
                    ).run()

        print(f":Warning! - Aligned trajectory on {pocket_definition}. You might have to realign if you want to calculate other variables that depend on relative distances.") if verbose else 0

    # Make selection. Updating is to make sure you update selection each frame.
    selection = u.select_atoms(f"({target}) and {pocket_definition}", updating=True)
    
    # Extract from each frame the positions (x, y, z) of the selected atoms, the id and the time.
    arr = np.concatenate([np.c_[selection.atoms.positions, selection.atoms.ids, selection.atoms.names, np.repeat(i.time / 1000, len(selection.atoms.positions))] for i in u.trajectory]) 

    # Turn into dataframe and determine dtypes
    df = pd.DataFrame(arr, columns = ['x','y','z', 'id', 'name', 'time'])
    df[['x', 'y', 'z', 'id', 'time']] = df[['x', 'y', 'z', 'id', 'time']].apply(pd.to_numeric)

    # Calculate density
    values = df[['x', 'y', 'z']].values.T
    kde = stats.gaussian_kde(values, bw_method='silverman')
    df['density'] = kde(values)

    return df

if __name__ == "__main__":
    
    # TODO
    pass
    