#!/usr/bin/env python3

import numpy as np
import pandas as pd

from scipy import stats
from scipy.stats import chi2
from sklearn.cluster import MeanShift

from kneed import KneeLocator

from MDAnalysis.analysis import align as align

from conserved_atoms.utils import tools

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

def cluster(df,
            n_frames,
            cutoff=None,
            outlier_treshold=0.01,
            verbose=True,
            ):
    """
    Calculate the density or a certain atom type (atom_name) throughout a simulation.
    
    :param df: Dataframe containing at least 'x', 'y', 'z', 'id', 'time' and 'density' columns. Output of calc_density.
    :param cutoff: Density cutoff. How to define the binding pocket or area in which to calculate density. (See par 3.1.4 on https://docs.mdanalysis.org/stable/documentation_pages/selections.html)
    :param atom_name: What is the name of the atom you want to calculate the density for in the trajectory. (OH2 is the oxygen atom of the water molecules)
    :param align: Align the trajectory on the atoms in the binding pocket.
    :param verbose: Allow printing

    :return: df containing the coordinates and density of all atoms in selection.
    """

    # Sort the dataframe and calculate (normalized) cumsum
    df_sorted = df.sort_values(by=['density'], ascending=False)
    df_sorted['density_cumsum'] = df_sorted['density'].cumsum()
    df_sorted['density_cumsum_norm'] = (df_sorted['density_cumsum'] - df_sorted['density_cumsum'].min()) / (df_sorted['density_cumsum'].max() - df_sorted['density_cumsum'].min())

    # If no cutoff is provided, then estimate it using the Mahalanobis distance and p-value
    # https://www.machinelearningplus.com/statistics/mahalanobis-distance/
    if not cutoff:

        print("No cutoff is given. Trying to estimate cutoff using the Mahalanobis distance and p-value...") if verbose else 0

        # Loop over different cutoff values. Here n_samples is the number of samples you try out.
        n_samples = 20
        bins = np.linspace(0.99, 0.1, n_samples + 1)

        # Create a list to store the dataframes, which you later concat.
        df_list = []

        for id, c_off in enumerate(bins):
            df_selection = df_sorted[df_sorted['density_cumsum_norm'] < c_off][['x', 'y', 'z', 'density']]
            
            # Add sample number.
            df_selection['sample'] = id

            # Run clustering
            # Bin seeding speeds up, but is less accurate
            ms = MeanShift(bandwidth=1.5, bin_seeding=True)
            # ms = MeanShift(bandwidth=bandwidth)
            ms.fit(df_selection[['x', 'y', 'z']].values)

            # Add cluster id values.
            df_selection['cluster_id'] = ms.labels_
    
            # Creating a new column in the dataframe that holds the Mahalanobis distance for each row
            df_selection['mahalanobis'] = np.concatenate([tools.calculate_malahanobis(y=df_selection[df_selection['cluster_id'] == i][['x', 'y', 'z']], data=df_selection[df_selection['cluster_id'] == i][['x', 'y', 'z']]) for i in df_selection['cluster_id'].unique()]) 

            # calculate p-value for each mahalanobis distance
            df_selection['p'] = 1 - chi2.cdf(df_selection['mahalanobis'], 2)

            # Append to full list
            df_list.append(df_selection)

        # Concatenate all dfs to one.
        df_all = pd.concat(df_list, axis=0)

        # Count number of outliers per sample.
        df_pcount = df_all[df_all['p'] < outlier_treshold][['p', 'sample']].groupby(["sample"]).count().rename(columns={'p': 'p_count'})

        kl = KneeLocator(df_pcount.index.values, df_pcount['p_count'], curve="convex", direction='decreasing')
        
        # Get the best df based on p_count knee. 
        df_clustered = df_all[(df_all['sample'] == kl.knee) & (df_all['p'] > outlier_treshold)].drop('sample', axis=1)

        print(f"\tDensity cutoff set to {df_clustered['density'].min()}") if verbose else 0

    else:

        # Get data subsection
        df_clustered = df_sorted[df_sorted['density'] < cutoff][['x', 'y', 'z', 'density']]

        # Run clustering
        # Bin seeding speeds up, but is less accurate
        ms = MeanShift(bandwidth=1.5, bin_seeding=True)
        # ms = MeanShift(bandwidth=bandwidth)
        ms.fit(df_clustered[['x', 'y', 'z']].values)

        # Add cluster id values.
        df_clustered['cluster_id'] = ms.labels_

        # Creating a new column in the dataframe that holds the Mahalanobis distance for each row
        df_clustered['mahalanobis'] = np.concatenate([tools.calculate_malahanobis(y=df_clustered[df_clustered['cluster_id'] == i][['x', 'y', 'z']], data=df_clustered[df_clustered['cluster_id'] == i][['x', 'y', 'z']]) for i in df_clustered['cluster_id'].unique()]) 

        # calculate p-value for each mahalanobis distance
        df_clustered['p'] = 1 - chi2.cdf(df_clustered['mahalanobis'], 2)

        df_clustered = df_clustered[df_clustered['p'] > outlier_treshold]


    # Make summary of the clusters. 
    size = df_clustered[['x', 'cluster_id']].groupby(["cluster_id"]).count()
    size.rename({'x': 'size'}, axis=1, inplace=True)

    means = df_clustered[['x', 'y', 'z', 'cluster_id']].groupby(["cluster_id"]).mean()
    std = df_clustered[['x', 'y', 'z', 'cluster_id']].groupby(["cluster_id"]).std()
    # ptp = df_clustered[['x', 'y', 'z', 'cluster_id']].groupby(["cluster_id"]).agg(np.ptp)

    df_summary = size.join(means).join(std, rsuffix='_std')#.join(ptp, rsuffix='_ptp')

    df_summary['occupation'] = df_summary['size'] / n_frames

    return df_clustered, df_summary

if __name__ == "__main__":
    
    # TODO
    pass
    