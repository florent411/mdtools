#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd

from tqdm import tqdm

from scipy import stats
from sklearn.cluster import MeanShift, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# from kneed import KneeLocator

from MDAnalysis.analysis import align as align

from conserved_atoms.utils import tools

# Radii of atoms in pm
# https://en.wikipedia.org/wiki/Atomic_radii_of_the_elements_(data_page)
ATOMIC_RADII = {'H' : 120, 'He' : 140, 'Li' : 182, 'Be' : 153, 'B' : 192, 'C' : 170, 'N' : 155, 'O' : 152, 'F' : 147, 'Ne' : 154, 'Na' : 227, 'Mg' : 173, 'Al' : 184, 'Si' : 210, 'P' : 180, 'S' : 180, 'Cl' : 175, 'Ar' : 188, 'K' : 275, 'Ca' : 231, 'Sc' : 211, 'Ni' : 163, 'Cu' : 140, 'Zn' : 139, 'Ga' : 187, 'Ge' : 211, 'As' : 185, 'Se' : 190, 'Br' : 185, 'Kr' : 202, 'Rb' : 303, 'Sr' : 249, 'Pd' : 163, 'Ag' : 172, 'Cd' : 158, 'In' : 193, 'Sn' : 217, 'Sb' : 206, 'Te' : 206, 'I' : 198, 'Xe' : 216, 'Cs' : 343, 'Ba' : 268, 'Pt' : 175, 'Au' : 166, 'Hg' : 155, 'Tl' : 196, 'Pb' : 202, 'Bi' : 207, 'Po' : 197, 'At' : 202, 'Rn' : 220, 'Fr' : 348, 'Ra' : 283, 'U' : 186} 

def calc_density(u,
                 pocket_definition='point 0.0 0.0 0.0 15',
                 target='OH2',
                 align_on='CA',
                 unwrap=False,
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
    if unwrap:
        protein = u.select_atoms('protein')

        for ts in tqdm(u.trajectory, desc='Unwrapping: ', unit='frames'):
            protein.unwrap(compound='fragments')

    if align_on:
        print("Aligning...", end="") if verbose else 0
        print(f":Warning! - Aligning trajectory on {pocket_definition} and name {target}. You might have to realign if you want to calculate other variables that depend on relative distances.") if verbose else 0

        align.AlignTraj(u,  # trajectory to align (mobile)
                        u,  # reference (ref)
                        select=f"protein and name {align_on} and ({pocket_definition})",  # selection of atoms to align
                        match_atoms=True,  # whether to match atoms based on mass
                        in_memory=True # Inplace
                    ).run()
        
        print("done") if verbose else 0

    # Make selection. Updating is to make sure you update selection each frame.
    selection = u.select_atoms(f"(name {target}) and {pocket_definition}", updating=True)
    
    # Extract from each frame the positions (x, y, z) of the selected atoms, the id and the time.
    arr = np.concatenate([np.c_[selection.atoms.positions, selection.atoms.ids, selection.atoms.names, np.repeat(i.time / 1000, len(selection.atoms.positions))] for i in tqdm(u.trajectory, desc='Extracting atoms: ', unit='frames')]) 

    # Turn into dataframe and determine dtypes
    df = pd.DataFrame(arr, columns = ['x','y','z', 'id', 'name', 'time'])
    df[['x', 'y', 'z', 'id', 'time']] = df[['x', 'y', 'z', 'id', 'time']].apply(pd.to_numeric)

    # Calculate density
    print("Calculating denstity...", flush=True, end="") if verbose else 0
    values = df[['x', 'y', 'z']].values.T
    kde = stats.gaussian_kde(values, bw_method='silverman')
    df['density'] = kde(values)
    print("done") if verbose else 0

    return df

def cluster(df,
            n_frames,
            clustering_algorithm='dbscan',
            epsilon=0.1,
            density_cutoff=None,
            atomic_radius=None,
            element='O',
            outlier_treshold=0.01,
            verbose=True,
            ):
    """
    Calculate the density or a certain atom type (atom_name) throughout a simulation.
    
    :param df: Dataframe containing at least 'x', 'y', 'z', 'id', 'time' and 'density' columns. Output of calc_density.
    :param n_frames: Number frames. Used to calculate the occupancy.
    :param clustering_algorithm: Perform clustering using 'dbscan' of 'meanshift' algorithm. (default='dbscan')
    :param epsilon: Resolution for dbscan. (default=0.1)
    :param density_cutoff: If you provide a density cutoff, only the density higher than this value will be taken into account and clustered. This is only for meanshift clustering. (default=None)
    :param atomic_radius: If no density cutoff is given the cutoff will be estimated using the given atomic radius. Only for meanshift clustering. (default=None)
    :param element: If atomic radius is given, the atomic radius of this element will be looked up in a list of known atomic radii. Only for meanshift clustering. (default='O')
    :param outlier_treshold: When cleaning up the clusters using the p-value, this will be the threshold for determining what is an outlier or not. (default=0.01)
    :param verbose: Print or not. (default=True)

    :return: df containing the coordinates and density of all atoms in selection.
    """

    # Perform DBSCAN
    if clustering_algorithm == 'dbscan':
        # Standardize the data so that each feature has mean=0 and variance=1
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(df[['x', 'y', 'z']])

        # Set the minimum cluster size to an occupancy of 10% (i.e. 10% of trajectory length)
        m_samples = round(0.08 * n_frames)

        # Create a DBSCAN object with the desired hyperparameters
        dbscan = DBSCAN(eps=epsilon, min_samples=m_samples)

        # Fit the model to your standardized data
        clusters = dbscan.fit_predict(data_scaled)

        # Add a new column to your original DataFrame with the cluster assignments
        df_clustered = df.copy()
        df_clustered['cluster_id'] = clusters

        # Remove all outliers
        # df_clustered = df_clustered[df_clustered['cluster_id'] != -1]

    # Use KMEANS clustering
    elif clustering_algorithm == 'meanshift':

        if atomic_radius == None:
            try:
                ATOMIC_RADII[element]
            except Exception as e:
                print(f"{e}\n\nElement {element} not found in the list of atomic radii. Please fill in the vdw radius in pm using the atomic_radius\= variable.")
                sys.exit(1)


        # Sort the dataframe and calculate (normalized) cumsum
        df_sorted = df.sort_values(by=['density'], ascending=False)
        df_sorted['density_cumsum'] = df_sorted['density'].cumsum()
        df_sorted['density_cumsum_norm'] = (df_sorted['density_cumsum'] - df_sorted['density_cumsum'].min()) / (df_sorted['density_cumsum'].max() - df_sorted['density_cumsum'].min())

        # If no cutoff is provided, then estimate it using the Mahalanobis distance and p-value
        # https://www.machinelearningplus.com/statistics/mahalanobis-distance/
        if not density_cutoff:

            if atomic_radius != None:
                print(f"No cutoff is given. Trying to estimate cutoff using atomic radius of {atomic_radius} pm.") if verbose else 0
            else:
                print(f"No cutoff is given. Trying to estimate cutoff using atomic radius of {element}, which is {ATOMIC_RADII[element]} pm.") if verbose else 0

            # Loop over different cutoff values. Here n_samples is the number of samples you try out.
            n_samples = 20
            bins = np.linspace(0.99, 0.1, n_samples + 1)

            # Create a list to store the dataframes, which you later concat.
            df_list = []

            for id, c_off in tqdm(enumerate(bins), desc="Sampling: ", unit=" samples"):
                df_selection = df_sorted[df_sorted['density_cumsum_norm'] < c_off].copy()
                
                # print(f"Sample {id} :", df_selection.shape)

                # Add sample number.
                df_selection['sample'] = id

                # Run clustering
                # Bin seeding speeds up, but is less accurate
                ms = MeanShift(bandwidth=1.5, bin_seeding=True)
                # ms = MeanShift(bandwidth=bandwidth)
                ms.fit(df_selection[['x', 'y', 'z']].values)

                # Add cluster id values.
                df_selection['cluster_id'] = ms.labels_
        
                # Count the number of points in each cluster
                counts = df_selection['cluster_id'].value_counts()

                # Filter out clusters with only more than 3 points
                keep_clusters = counts[counts > 3].index
                df_selection = df_selection[df_selection['cluster_id'].isin(keep_clusters)]
                    
                try:
                    # Creating a new column in the dataframe that holds the Mahalanobis distance for each row
                    df_selection['mahalanobis'] = np.concatenate([tools.calculate_mahalanobis(df_selection[df_selection['cluster_id'] == i][['x', 'y', 'z']]) for i in df_selection['cluster_id'].unique()]) 

                    # calculate p-value for each mahalanobis distance
                    df_selection['p'] = 1 - stats.chi2.cdf(df_selection['mahalanobis'], 2)
                except Exception as e:
                    print(e)
                    df_selection['mahalanobis'] = np.nan
                    df_selection['p'] = np.nan

                # Append to full list
                df_list.append(df_selection)

            # Concatenate all dfs to one.
            df_all = pd.concat(df_list, axis=0)

            # Count number of outliers per sample.
            # df_pcount = df_all[df_all['p'] < outlier_treshold][['p', 'sample']].groupby(["sample"]).count().rename(columns={'p': 'p_count'})
            # kl = KneeLocator(df_pcount.index.values, df_pcount['p_count'], curve="convex", direction='decreasing')


            # Calculate the average radius for each sample
            avg_radii = []
            for sample in df_all['sample'].unique():     
                try:
                    avg_radii.append(tools.get_cluster_radii(df_all.loc[df_all['sample'] == sample]).mean().to_numpy())
                except:
                    avg_radii.append(np.nan)

            avg_radii = np.array(avg_radii)

            # Find the index of the value closest half the atomic radius. (/100 to go from pm to Angstrom)
            radius_cutoff = (ATOMIC_RADII[element] / 100) / 2
            best_sample = np.abs(avg_radii.T[0] - radius_cutoff).argmin()

            # Get the best df based on p_count knee. 
            df_clustered = df_all[(df_all['sample'] == best_sample) & (df_all['p'] > outlier_treshold)].drop('sample', axis=1)

            print(f"-> Density cutoff set to {df_clustered['density'].min()}") if verbose else 0

        else:

            # Get data subsection
            df_clustered = df_sorted[df_sorted['density'] < density_cutoff][['x', 'y', 'z', 'density']]

            # Run clustering
            # Bin seeding speeds up, but is less accurate
            ms = MeanShift(bandwidth=1.5, bin_seeding=True)
            # ms = MeanShift(bandwidth=bandwidth)
            ms.fit(df_clustered[['x', 'y', 'z']].values)

            # Add cluster id values.
            df_clustered['cluster_id'] = ms.labels_

            # Count the number of points in each cluster
            counts = df_clustered['cluster_id'].value_counts()

            # Filter out clusters with only more than 3 points
            keep_clusters = counts[counts > 3].index
            df_clustered = df_clustered[df_clustered['cluster_id'].isin(keep_clusters)]
    
    else:
        print(f"ERROR: {clustering_algorithm} not recognized. Please use 'dbscan' or 'meanshift'.")
        sys.exit(1)


    # Creating a new column in the dataframe that holds the Mahalanobis distance for each row
    df_clustered['mahalanobis'] = np.concatenate([tools.calculate_mahalanobis(df_clustered[df_clustered['cluster_id'] == i][['x', 'y', 'z']]) for i in df_clustered['cluster_id'].unique()]) 

    # calculate p-value for each mahalanobis distance
    df_clustered['p'] = 1 - stats.chi2.cdf(df_clustered['mahalanobis'], 2)

    # Remove outliers from clusters.
    df_clustered = df_clustered[df_clustered['p'] > outlier_treshold]


    # Make summary of the clusters. 
    size = df_clustered[['x', 'cluster_id']].groupby(["cluster_id"]).count()
    size.rename({'x': 'size'}, axis=1, inplace=True)

    means = df_clustered[['x', 'y', 'z', 'cluster_id']].groupby(["cluster_id"]).mean()
    std = df_clustered[['x', 'y', 'z', 'cluster_id']].groupby(["cluster_id"]).std()
    # ptp = df_clustered[['x', 'y', 'z', 'cluster_id']].groupby(["cluster_id"]).agg(np.ptp)

    df_summary = size.join(means).join(std, rsuffix='_std')#.join(ptp, rsuffix='_ptp')

    # Calculate the avg cluster radius
    df_summary = df_summary.join(tools.get_cluster_radii(df_clustered))

    # Calculate cluster occupancy. 
    df_summary['occupancy'] = df_summary['size'] / n_frames


    return df_clustered, df_summary

def voxelize(df,
             resolution=0.03,
             verbose=True):
    """
    Voxelizes a point cloud by grouping nearby points into a single point.
    
    :param df: Dataframe contaning at at least ['x', 'y', 'z'] columns. From any other column the avg will be taken for all grouped points.
    :param resolution: float specifying the size of the voxel grid

    :return: df where the x, y and z coordinates are voxelized and the mean is taken for all remaining columns.
    """

    print("Voxelizing...", end="") if verbose else 0
    
    # Make copy to not be editing the original dataframe.
    df_voxelized = df.copy()

    # Scale the points to be between 0 and 1
    scaler = MinMaxScaler()
    points = scaler.fit_transform(df_voxelized[['x', 'y', 'z']].values)

    # Compute the voxel grid indices for each point
    indices = np.floor(points / resolution) + 0.5

    # Scale the voxelized points back to the original range and add back to dataframe
    voxelized_points = scaler.inverse_transform(indices * resolution)
    df_voxelized.loc[:, ['x', 'y', 'z']] = voxelized_points

    # Only keep the unique points
    df_voxelized = df_voxelized.groupby(['x', 'y', 'z'], as_index=False).mean()

    print("done") if verbose else 0
    print(f"Reduced number of points from {len(df)} to {len(df_voxelized)} points.") if verbose else 0
    print(f"That's a {(len(df) - len(df_voxelized))/len(df):.0%} reduction.") if verbose else 0

    return df_voxelized


if __name__ == "__main__":
    
    # TODO
    pass
    