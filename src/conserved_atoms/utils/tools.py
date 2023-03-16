#!/usr/bin/env python 

''' General tools for conserved_atoms protocol '''

import os
import shutil
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

def calculate_mahalanobis(data):
    """
    Function to calculate the Mahalanobis distance
    https://www.machinelearningplus.com/statistics/mahalanobis-distance/
    """

    # Calculate covariance matrix and mean of the data
    cov = np.cov(data, rowvar=False)
    mean = np.mean(data, axis=0)

    # Calculate the Mahalanobis distance for each point in the point cloud using einsum (faster than other methods)
    diff = data - mean
    distances = np.sqrt(np.einsum('ij,ji->i', np.dot(diff, np.linalg.inv(cov)), diff.T))

    return distances

def get_cluster_radii(df):
    """
    Function to calculate the average radius of each cluster. 
    This is done by taking the average distance of the cluster mean to each of the points that make up the convex hull of the cluster. 
    """

    radii = []
    for cluster_id in df['cluster_id'].unique():
        cluster_points = df.loc[df['cluster_id'] == cluster_id, ['x', 'y', 'z']].to_numpy()
        
        # Calculate the convex hull and extract the points that form the hull
        hull = ConvexHull(cluster_points)
        hull_points = cluster_points[hull.vertices]
        
        # Get the mean of the full point cloud (hull_points is the points on the hull, hull.points is all points)
        center_point = np.mean(hull.points, axis=0)
        
        # Einsum is supposed to be faster, but is also harder to read, so I use linalg.norm instead
        # distances = np.sqrt(np.einsum('ij,ij->i', hull.points - mean, hull.points - mean))
        distances = np.linalg.norm(hull_points - center_point, axis=1)

        # Distances give an idea of the radius of the point cloud.
        radius = np.mean(distances)
        radius_std = np.std(distances)
        
        # Add to list
        radii.append([cluster_id, radius, radius_std])

    # Turn into dataframe, turn cluster_id column to int type and then set cluster_id column as index.
    df_rad = pd.DataFrame(np.array(radii), columns=['cluster_id', 'radius', 'radius_std'])
    df_rad['cluster_id'] = df_rad['cluster_id'].astype(int)
    df_rad.set_index('cluster_id', inplace=True)

    return df_rad

def reset(path, verbose):
    """
    Check if folder exists. If so, delete old one and make new one.
    """

    if not os.path.exists(path):
        print(f"Making {path}...", flush=True, end="") if verbose else 0
        os.makedirs(path)
        print("done") if verbose else 0
    else:
        print(f"Deleting {path}*...", flush=True, end="") if verbose else 0
        shutil.rmtree(path)
        print("done") if verbose else 0

        print(f"Making {path}...", flush=True, end="") if verbose else 0
        os.makedirs(path)
        print("done") if verbose else 0

    return 0