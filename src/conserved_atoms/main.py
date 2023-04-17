#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd

import warnings
from tqdm import tqdm
from datetime import datetime
import argparse

from scipy import stats
from sklearn.cluster import MeanShift, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import MDAnalysis as mda
from MDAnalysis.analysis import align as align
from MDAnalysis.coordinates.memory import MemoryReader
from MDAnalysis.analysis.base import AnalysisFromFunction
import MDAnalysis.transformations as trans

from conserved_atoms.utils import tools

# Radii of atoms in pm
# https://en.wikipedia.org/wiki/Atomic_radii_of_the_elements_(data_page)
ATOMIC_RADII = {'H' : 120, 'He' : 140, 'Li' : 182, 'Be' : 153, 'B' : 192, 'C' : 170, 'N' : 155, 'O' : 152, 'F' : 147, 'Ne' : 154, 'Na' : 227, 'Mg' : 173, 'Al' : 184, 'Si' : 210, 'P' : 180, 'S' : 180, 'Cl' : 175, 'Ar' : 188, 'K' : 275, 'Ca' : 231, 'Sc' : 211, 'Ni' : 163, 'Cu' : 140, 'Zn' : 139, 'Ga' : 187, 'Ge' : 211, 'As' : 185, 'Se' : 190, 'Br' : 185, 'Kr' : 202, 'Rb' : 303, 'Sr' : 249, 'Pd' : 163, 'Ag' : 172, 'Cd' : 158, 'In' : 193, 'Sn' : 217, 'Sb' : 206, 'Te' : 206, 'I' : 198, 'Xe' : 216, 'Cs' : 343, 'Ba' : 268, 'Pt' : 175, 'Au' : 166, 'Hg' : 155, 'Tl' : 196, 'Pb' : 202, 'Bi' : 207, 'Po' : 197, 'At' : 202, 'Rn' : 220, 'Fr' : 348, 'Ra' : 283, 'U' : 186} 

def calc_density(u,
                 pocket_definition='point 0.0 0.0 0.0 15',
                 target='OH2',
                 align_on='CA',
                 unwrap=False,
                 write_traj=False,
                 verbose=True,
                 ):
    """
    Calculate the density or a certain atom type (atom_name) throughout a simulation.
    
    :param u: MDAnalysis universe, including a trajectory.
    :param pocket_definition: How to define the binding pocket or area in which to calculate density. (See par 3.1.4 on https://docs.mdanalysis.org/stable/documentation_pages/selections.html)
    :param align_on: Align the trajectory on the atoms in the binding pocket.
    :param target: What is the name of the atom you want to calculate the density for in the trajectory? (OH2 is the oxygen atom of the water molecules)
    :param unwrap: 
    :param write_traj: Write trajectory aligned to the defined pocket into pocket.xtc.
    :param verbose: Allow printing.

    :return: df containing the coordinates and density of all atoms in selection.
    """

    # Align trajectory
    if unwrap:
        print("Unwrapping protein...", end="") if verbose else 0
        # Unwrap protein
        protein = u.select_atoms('protein')
        not_protein = u.select_atoms('not protein')

        transforms = [trans.unwrap(u.atoms),
                    trans.center_in_box(protein, wrap=True),
                    trans.wrap(u.atoms, compound='fragments')]

        u.trajectory.add_transformations(*transforms)
        print("done") if verbose else 0

    if align_on:
        print(f"Aligning trajectory on {pocket_definition} and name {align_on}.\n(Note: You might have to realign if you want to calculate other variables that depend on relative distances.)") if verbose else 0

        print("Aligning...", end="") if verbose else 0
        align.AlignTraj(u,  # trajectory to align (mobile)
                        u,  # reference (ref)
                        select=f"protein and name {align_on} and ({pocket_definition})",  # selection of atoms to align
                        match_atoms=True,  # whether to match atoms based on mass
                        in_memory=True # Inplace
                    ).run()
        
        print("done") if verbose else 0

        if write_traj:
            print("Writing trajectory into pocket.xtc...", end="") if verbose else 0

            protein = u.select_atoms('protein')

            protein.write("pocket.gro")
            with mda.Writer("pocket.xtc", protein.n_atoms) as W:
                for ts in u.trajectory:
                    W.write(protein)
            print("done") if verbose else 0

    # Make selection. Updating is to make sure you update selection each frame.
    selection = u.select_atoms(f"(name {target}) and {pocket_definition}", updating=True)
    
    # Extract from each frame the positions (x, y, z) of the selected atoms, the id and the time.
    print("Extracting atoms...", flush=True, end="") if verbose else 0
    arr = np.concatenate([np.c_[selection.atoms.positions, selection.atoms.ids, selection.atoms.names, np.repeat(i.time / 1000, len(selection.atoms.positions))] for i in u.trajectory]) 
    print("done") if verbose else 0

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

    # Set values to nan for all outliers
    df_summary.loc[df_summary.index == -1, ['occupancy', 'x', 'y', 'z', 'x_std', 'y_std', 'z_std', 'radius', 'radius_std']] = np.nan

    # Index ['cluster_id'] to column
    df_summary.reset_index(names=['cluster_id'], inplace=True)

    # Add occupancy column to df_clustered.
    df_clustered = pd.merge(df_clustered, df_summary[['cluster_id', 'occupancy']], on='cluster_id')

    return df_clustered, df_summary

def voxelize(df,
             resolution=0.03,
             compact_by='mean',
             skip_outliers=True,
             
             verbose=True):
    """
    Voxelizes a point cloud by grouping nearby points into a single point.
    
    :param df: Dataframe contaning at at least ['x', 'y', 'z'] columns. From any other column the avg will be taken for all grouped points.
    :param resolution: float specifying the size of the voxel grid

    :return: df where the x, y and z coordinates are voxelized and the mean is taken for all remaining columns.
    """

    print("Voxelizing...", end="") if verbose else 0
    
    # Make copy to not be editing the original dataframe.
    # I only use the numeric data, otherwise you can get problems with the groupby function.
    # You can use the undocumented function _get_numeric_data() to filter only numeric columns:
    # And remove outliers if needed.
    if skip_outliers:
        try:
            df_voxelized = df[df['cluster_id'] != -1]._get_numeric_data().copy()
        except Exception as e:
            print(f"Warning: {e}")
            print("Continuing without removing outliers.")
            df_voxelized = df._get_numeric_data().copy()

    # Scale the points to be between 0 and 1
    scaler = MinMaxScaler()
    points = scaler.fit_transform(df_voxelized[['x', 'y', 'z']].values)

    # Compute the voxel grid indices for each point
    indices = np.floor(points / resolution) + 0.5

    # Scale the voxelized points back to the original range and add back to dataframe
    voxelized_points = scaler.inverse_transform(indices * resolution)
    df_voxelized.loc[:, ['x', 'y', 'z']] = voxelized_points

    # Only keep the unique points
    if compact_by == 'mean':
        df_voxelized = df_voxelized.groupby(['x', 'y', 'z'], as_index=False).mean()
    elif compact_by == 'max':
        df_voxelized = df_voxelized.groupby(['x', 'y', 'z'], as_index=False).max()
    elif compact_by == 'min':
        df_voxelized = df_voxelized.groupby(['x', 'y', 'z'], as_index=False).min()
    else:
        print(f"ERROR: compact_by type {compact_by} is not supported. Please select 'mean', 'max' or 'min'")
        sys.exit(1)

    print("done") if verbose else 0
    print(f"Reduced number of points from {len(df)} to {len(df_voxelized)} points. ({(len(df) - len(df_voxelized))/len(df):.0%} reduction)") if verbose else 0

    return df_voxelized

def write_pdb(df,
              output='conserved_atoms.pdb',
              skip_outliers=True,
              verbose=True):

    """
    Write a pdb file with the points given in the dataframe.
    
    :param df: Dataframe contaning at at least ['x', 'y', 'z'] columns. If an occupancy or density column is found, these are used as b-factor.
    :param skip_outliers: Skip all points that are not part of a cluster (cluster -1), i.e. the outliers.
    :param verbose: Allow printing.

    :return: True
    """

    print(f"Writing {output}...", end="") if verbose else 0

    # Check that the dataframe contains at least x, y and z columns.
    assert set(['x', 'y', 'z']).issubset(df.columns),\
            f"Dataframe should contain at least 'x', 'y', 'z' columns.\nOnly found the following columns: {df.columns.values}"

    if len(df) > 10000:
        # If dataframe is too large, raise a warning and suggest voxelization
        warnings.warn(f'''
WARNING: Your dataframe is very large ({len(df)} points).
We recommend to voxelize your data using the following function:

df_vox = conserved_atoms.voxelize(df_clustered)

and the rerunning this command:

conserved_atoms.write_pdb(df_vox)''')

        # Ask the user if they want to continue
        response = input("Do you want to continue? (y/N)")

        # If the user doesn't want to continue, raise an exception
        if response.lower() not in ['Y', 'y']:
            raise Exception("Now exiting.")

    # Make copy to not be editing the original dataframe and remove outliers if needed.
    if skip_outliers:
        try:
            df = df[df['cluster_id'] != -1].copy()
        except Exception as e:
            print(f"Warning: {e}")
            print("Continuing without removing outliers.")
    else:
        df = df.copy()

    # Reset index and turn it into a column to create the atomserial. (Just a list from 1-n)
    df.reset_index(drop=True, inplace=True)
    df['atomserial'] = df.index + 1

    # If the dataframe contains a density column, normalize it on a scale from 0 to 100
    # This will be put in the B-factor column of the pdb file.
    if 'density' in df.columns:
        scaler = MinMaxScaler((0, 100))
        df['density'] = scaler.fit_transform(df['density'].values.reshape(-1,1))

    # Write output in pdb files (for pymol etc.)
    with open(output, "w") as wf:

        # Print header with some information
        header = f'''# This .pdb file was created on {datetime.now().strftime("%d/%m/%Y %H:%M:%S")} using MDTools - Conserved Atoms.
#
# Note: The occupancy represents the fraction of time an atom of that cluster is found over the analysed simulation time. 0.0 - 1.0
#       The B-factor here represents the atomdensity normalized to a scale from 0 - 100. (Where 0 is lowest density and 100 is highest density.)
'''

        wf.write(header)

        # Write dataframe in .pdb format.
        for i, (_, row) in enumerate(df.iterrows()):
            
            # Make line
            printLine = '''\
{atomlabel:<4} {atomSerial:>6}  {atomName:<3} {resName:<3} {chainId} {resNum:>3}    {x:>8.3f}{y:>8.3f}{z:>8.3f}  {occupancy:>4.2f} {b_factor:>6.2f}         {element:>2}\
\n'''.format(atomlabel="ATOM",
            atomSerial=i,
            atomName=row['name'] if 'name' in df.columns else 'C',
            resName="CLS",
            chainId="Z",
            resNum=int(row['cluster_id']),
            x=row['x'],
            y=row['y'],
            z=row['z'],
            occupancy=min(1, row['occupancy']) if 'occupancy' in df.columns else 0.0,
            b_factor=row['density'] if 'density' in df.columns else 0.0,
            element='O')

            wf.write(printLine)

    print("done") if verbose else 0

    return

def write_dat(df,
              output='conserved_atoms.dat',
              verbose=True):

    """
    Write a pdb file with the points given in the dataframe.
    
    :param df: Dataframe contaning at at least ['x', 'y', 'z'] columns. If an occupancy or density column is found, these are used as b-factor.
    :param resolution: float specifying the size of the voxel grid
    :param verbose: Allow printing.

    :return: df where the x, y and z coordinates are voxelized and the mean is taken for all remaining columns.
    """

    print(f"Writing {output}...", end="") if verbose else 0

    # Format for all column types
    float_formats = {'x' : '%10.6f',
                    'y' : '%10.6f',
                    'z' : '%10.6f',
                    'x_std' : '%10.6f',
                    'y_std' : '%10.6f',
                    'z_std' : '%10.6f',
                    'radius' : '%8.4f',
                    'radius_std' : '%8.4f',
                    'id' : '%6d',
                    'name' : '%3s',
                    'mahalanobis' : '%12.4e',
                    'size' : '%5d',
                    'p' : '%12.4e',
                    'time' : '%6.2f',
                    'density' : '%12.4e',
                    'occupancy' : '%6.2f',
                    'cluster_id' : '%2d',
                    'index' : '%6d'}

    # Get the ones corresponding to the columns in the dataframes.
    fmt_list = [float_formats[c] if c in float_formats else '%s' for c in df.columns.values]

    # write the dataframe to a csv file with different alignment parameters for different columns
    np.savetxt(output, df.values, fmt=fmt_list,  delimiter='\t', header='\t'.join(df.columns), comments='')

    print("done") if verbose else 0

    return

def create_traj(u,
                df,
                write_struct_to='conserved_atoms.gro',
                write_traj_to='conserved_atoms.xtc',
                element='O',
                name='OH2',
                skip_outliers=True,
                verbose=True):

    """
    Create trajectory containing the conserved atoms merged with the trajectory used for analysis.
    
    :param u: MD Analysis universe used for the conserved atoms algorithm.
    :param df: Dataframe contaning at at least ['x', 'y', 'z', 'cluster_id', 'time'] columns. This is most likely the df_clustered which is outputted by conserved_atoms.cluster().
    :param write_to: Write output into this file.
    :param verbose: Allow printing.

    :return combined_u: Combined MDAnalysis universe.
    """

    print(f"Creating universe for conserved atom clusters...", end="") if verbose else 0

    # Make copy to not be editing the original dataframe and remove outliers if needed.
    if skip_outliers:
        df = df[df['cluster_id'] != -1][['cluster_id', 'time', 'x', 'y', 'z']].copy()
    else:
        df = df[['cluster_id', 'time', 'x', 'y', 'z']].copy()

    # Setup variables for new Universe.
    n_residues = len(df['cluster_id'].unique())
    n_atoms = len(df['cluster_id'].unique())
    resindices = np.repeat(range(n_residues), 1)
    assert len(resindices) == n_atoms
    segindices = [0] * n_residues

    # create the Universe
    clusters_u = mda.Universe.empty(n_atoms,
                                    n_residues=n_residues,
                                    atom_resindex=resindices,
                                    residue_segindex=segindices,
                                    trajectory=True) # necessary for adding coordinates

    # Add attributes
    clusters_u.add_TopologyAttr('name', [name] * n_residues)
    clusters_u.add_TopologyAttr('type', [element] * n_residues)
    clusters_u.add_TopologyAttr('resname', ['CLS'] * n_residues)
    clusters_u.add_TopologyAttr('resid', list(range(n_residues)))
    clusters_u.add_TopologyAttr('segid', ['conserved_atoms'])


    # Create an array containing the coordinates for the whole trajectory. All clusters that are not present are placed at [0., 0., 0.].
    # Get all possible combinations of time and cluster_id
    time_values = [u.trajectory.time / 1000 for ts in u.trajectory]
    cluster_id_values = df['cluster_id'].unique()
    idx = pd.MultiIndex.from_product([time_values, cluster_id_values], names=['time', 'cluster_id'])

    # Turn df into multi index df. 
    df_mi = df.set_index(['time', 'cluster_id'])
    
    # It could be that you have duplicates in your df_mi. 
    # This is probably caused that two water molecules from the same timestep appear in the same cluster.
    # In this case, warn the user and remove the duplicates.
    df_difference = len(df_mi) - len(df_mi[~df_mi.index.duplicated()])
    if df_difference > 0:
        warnings.warn(f"On {df_difference} occasions ({df_difference / len(df_mi):.1%}), two atoms from the same timestep appear in the same cluster. Duplicates are now removed to produce a trajectory.")

        df_mi = df_mi[~df_mi.index.duplicated()]

    # Reindex the dataframe to include all combinations and fill NaN value with zeros.
    df_mi = df_mi.reindex(idx).fillna(value=0.)
      
    # sort, convert to numpy array and reshape
    cluster_coords = df_mi.sort_values(by=['time', 'cluster_id']).to_numpy().reshape((len(time_values), len(cluster_id_values), 3))

    # Add coordinates to the trajectory
    clusters_u.load_new(cluster_coords, format=MemoryReader)

    print("done") if verbose else 0

    print(f"Merging with main universe...", end="") if verbose else 0

    # Getting coordinates for both universers and
    universe_coords = AnalysisFromFunction(lambda ag: ag.positions.copy(), u.atoms).run().results.timeseries
    cluster_coords = AnalysisFromFunction(lambda ag: ag.positions.copy(), clusters_u.atoms).run().results.timeseries

    merged_coords = np.hstack([universe_coords, cluster_coords])

    # Create combined universe
    combined_u = mda.Merge(u.atoms, clusters_u.atoms)
    combined_u.load_new(merged_coords, format=MemoryReader)

    print("done") if verbose else 0

    # Write structure file
    if write_struct_to:
        print(f"Writing structure file to {write_struct_to}...", end="") if verbose else 0
        pass
        print("done") if verbose else 0

    # Write trajectory
    if write_traj_to:
        print(f"Writing trajectory to {write_traj_to}...", end="") if verbose else 0
        pass
        print("done") if verbose else 0


    return combined_u

if __name__ == "__main__":
    
    # Setup argparse (allowing the use of flags)
    parser = argparse.ArgumentParser(description='Find conserved atoms.')

    # Adding flags
    # Optional flags (See help for explanation)
    parser.add_argument('-s',
                        '--structure',
                        nargs='?',
                        default='run.pdb',
                        type=str,
                        help='Structure file. (default: %(default)s)')

    parser.add_argument('-f',
                        '--trajectory',
                        nargs='?',
                        default='run.xtc',
                        type=str,
                        help='Trajectory file. (default: %(default)s)')

    parser.add_argument('-o',
                        '--output_dir',
                        nargs='?',
                        default='./conserved_atoms',
                        type=str,
                        help='Align trajectory on. (default: %(default)s)')

    parser.add_argument('-p',
                        '--pocket_definition',
                        nargs='?',
                        default='point 0.0 0.0 0.0 15',
                        type=str,
                        help='Definition of pocket to analyse. (default: %(default)s)')

    parser.add_argument('-a',
                        '--align_on',
                        nargs='?',
                        default='CA',
                        type=str,
                        help='Align trajectory on. (default: %(default)s)')
    
    parser.add_argument('-t',
                        '--target',
                        nargs='?',
                        default='OH2',
                        type=str,
                        help='Atom name of target atoms. (default: %(default)s)')

    parser.add_argument('-e',
                        '--element',
                        nargs='?',
                        default='O',
                        type=str,
                        help='Element. (default: %(default)s)')

    parser.add_argument('-w',
                        '--write',
                        nargs='?',
                        default='./conserved_atoms/conserved_atoms.{gro,xtc}',
                        type=str,
                        help='Write trajectorty Element. (default: %(default)s)')

    parser.add_argument('-v',
                        '--verbose',
                         action='store_true')


    # Rename variables from flags
    args = parser.parse_args()

    structure = args.structure
    trajectory = args.trajectory
    output_dir = args.output_dir
    pocket_definition = args.pocket_definition
    align_on = args.align_on
    target = args.target
    element = args.element
    verbose = args.verbose

    # Print start
    print("Starting conserved_atoms.py")

    # Reset directory
    tools.reset(output_dir, verbose=verbose)

    # Setup MDAnalysis Universe
    u = mda.Universe(structure, trajectory)

    # Define binding pocket
    df = calc_density(u,
                      pocket_definition=pocket_definition,
                      target=target,
                      align_on=align_on,
                      unwrap=False,
                      write_traj=False,
                      verbose=verbose)

    # Perform clustering
    df_clustered, df_summary = cluster(df,
                                       u.trajectory.n_frames,
                                       clustering_algorithm='dbscan',
                                       epsilon=0.1,
                                       density_cutoff=None,
                                       atomic_radius=None,
                                       element=element,
                                       outlier_treshold=0.01,
                                       verbose=verbose)

    print(f"Found {len(df_summary) - 1} clusters.")

    print(f"Writing output in {output_dir}")
    # Output the whole lot into files
    write_pdb(df_summary,
              output=f'{output_dir}/clusters_summary.pdb',
              skip_outliers=True,
              verbose=verbose)

    write_dat(df_summary,
              output=f'{output_dir}/clusters_summary.dat',
              verbose=verbose)

    if len(df_clustered) < 10000:   
        write_pdb(df_clustered,
                  output=f'{output_dir}/clusters_all.dat',
                  skip_outliers=True,
                  verbose=verbose)
    else:
        print(f"Not outputing clusters_all.pdb, as the number of points in is too large ({len(df_clustered)} > 10k)")

    write_dat(df_clustered,
            output=f'{output_dir}/clusters_all.dat',
            verbose=verbose)

    # Voxelize dataframe to reduce number of datapoints.
    df_vox = voxelize(df_clustered)

    write_pdb(df_vox,
              output=f'{output_dir}/clusters_vox.pdb',
              skip_outliers=True,
              verbose=verbose)

    write_dat(df_vox,
              output=f'{output_dir}/clusters_vox.dat',
              verbose=verbose)

    # Create output files
    create_traj(u,
                df_clustered,
                element=element,
                write_struct_to=f'{output_dir}/conserved_atoms.gro',
                write_traj_to=f'{output_dir}/conserved_atoms.xtc',
                name=target,
                skip_outliers=True,
                verbose=verbose)

    print("Finished.")



