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

def weights(colvar, temp=310):
    """
    Calculate the weights corresponding to each frame in the trajectory. 
    These weights can be used for reweighting other cvs. 
        
    :param colvar: the dataframe created from the COLVAR file.
    :param temp: temperature in K
    
    :returns: df containing columns 'weights' and 'time'
    """

    kb = 1.38064852e-23 # Boltzman's constant in m^2 kg s^-2 K^-1 or J K^-1
    NA = 6.02214086e23 # Avogadro's constant in mol^-1

    # Get the bias values as a unitless factor.
    # temp is the temperature
    bias = colvar['opes.bias'].values / (kb * NA * temp / 1000)

    # Calculate weights 
    weights = np.exp(bias - np.amax(bias))

    # Determine corresponding time, frame and walker in the trajectory
    time = colvar['time'].values
    origin = colvar['origin'].values

    # Calculate the effective sample size (not yet used for anything)
    effsize = np.sum(weights)**2 / np.sum(weights**2)

    df = pd.DataFrame({'time' : time, 'weights' : weights, 'origin' : origin})
        
    return df

def from_state(state_data,
               state_info,
               cvs=None,
               process='last',
               mintozero=True,
               temp=310,
               units='kT',
               bins=100,
               device='mt',
               verbose=True):
    
    """
    Calculate the free energy surface from the dumped state file (STATE_WFILE)
        
    :param state_data: the dataframe created from the STATE file, containing the state data.
    :param state_info: the dataframe created from the STATE file, containing the header info.
    :param cvs: CV(s) to calculate the FES for, determines dimensionality. If None, find them in the COLVAR df.
    :param process: Options: -'all' process all frames
                             -'last' process only last frame 
                             - n (int) process n frames evenly spread out over the trajectory.')
    :param mintozero: Shift the minimum to zero
    :param temp: temperature in K
    :param units: Output units. 
                  Options: -'kT'
                           -'kJ/mol'
                           -'kcal/mol'.
    :param bins: Number of bins per each cv: int, tuple, list or comma separated string.
    :param device: How to run calculations. 
                   Options: -'mt' multithreaded looped code structure (fastest for smaller sets). 
                            -'np' performs all calculations using numpy arrays. 
                            -'torch' offloads to GPU (CUDA or MLS, the MacOS Intel GPU). If no GPU is available it runs on the CPUs.
    :param verbose: Print output when running.
  
    :returns: df containing columns '[cv]' and 'time'
    """

    # Calculate unitfactor (units conversion factor)
    unitfactor = tools.get_unitfactor(units, temp)

    # Define cvs, either from user input, or, if cvs are not defined, take the cvs from file
    cvs = tools.define_cvs(cvs, state_data, type='state')

    # Get dimensionality
    dimensions = len(cvs)
    print(f"Calculating FES of {' and '.join(cvs)} ({dimensions} dimension(s))") if verbose else 0
    
    # Get list of all different times, which corresponds to unique state and scrub to only keep the states you want to analyse.
    times_list = tools.scrub_times_list(state_data['time'].unique(), process, verbose)

    # Dict comprehension to find the relevant states in the dataframe, i.e. the one to calculate the FES from.
    rel_states = {time : state for time, state in state_data.groupby('time') if time in times_list}

    # Loop over all state in times_list
    for time, state in tqdm(rel_states.items(), desc=f"Working on state", total=len(rel_states.items()), unit=""):
        
        # Get the kernels (centers) and width (sigmas) and store as np_array.
        # The arrays have the shape n_cvs x n_points, except the height, which is 1D (list).
        # I use the cv_order_key to make sure the order of the columns is the same as in the dataframe (or plumed output file).
        centers = np.array(state[[cv for cv in state.columns if cv in cvs]].astype(float).values)

        sigmas = np.array(state[[f"sigma_{cv}" for cv in state.columns if cv in cvs]].astype(float).values)

        # Get Kernel Hights as a list.
        height = np.array(state['height'].astype(float).values)

        # Get other variables
        sum_weights = state_info[state_info['time'] == time]["sum_weights"].astype(float).iloc[0]
        zed = state_info[state_info['time'] == time]["zed"].astype(float).iloc[0] * sum_weights
        epsilon = state_info[state_info['time'] == time]["epsilon"].astype(float).iloc[0]
        cutoff = state_info[state_info['time'] == time]['kernel_cutoff'].astype(float).iloc[0]
        val_at_cutoff = np.exp(-0.5 * cutoff**2)
        
        # Prepare the grid
        # Define the bounds (min, max) and number of bins for each state
        # Where axis=0 refers to columns
        grid_min = centers.min(axis=0)
        grid_max = centers.max(axis=0)

        # Setup the grid
        n_bins, mgrid = tools.setup_grid(grid_min, grid_max, bins, dimensions)
        grid_combos = np.array(mgrid.T.reshape(-1, dimensions))

        # List used to save all probablities.
        norm_probabilities = []

        # Standard looped algorithm
        if device in ['mt']:
            
            def state_to_norm_prob(combo):
                ''' Nested function. This function has direct access to variables and names defined in the enclosing function.
                This function is used to calculate the FES from the bias. Is called below by the multiprocessing pool.'''

                # delta_s = s - s0 / sigma
                delta_s = (combo[:, np.newaxis] - centers.T)/sigmas.T
                
                # "the Gaussian function is defined as G(s, s0) = h exp −0.5 SUM(s − s0)/sigma)^2 (eq. 2.64)
                gauss_function = height * (np.maximum(np.exp(-0.5 * np.sum(delta_s**2, axis=0)) - val_at_cutoff, 0))

                # "However, P˜(s) should be normalized not with respect to the full CV space but only over the CV space actually explored up to step n, which we call Wn. Thus, we introduce the normalization factor Z (zed). Finally, we can explicitly write the bias at the nth step as eq (2.63) where epsilon can be seen as a regularization term that ensures that the argument of the logarithm is always greater than zero."
                return np.sum(gauss_function)/zed + epsilon

            # Calculate normalized probabilities using multithreading by calling the nested function above
            with ThreadPool() as pool:
                norm_probabilities = list(tqdm(pool.imap(state_to_norm_prob, grid_combos), total=len(grid_combos), desc=f"Gridpoint", unit="", leave=False))

            # Reshape back into n-dimensional array
            norm_probabilities = np.array(norm_probabilities).reshape(n_bins)               

            # Set minimum value to zero
            if not mintozero:
                max_prob = 1
            else:
                max_prob = np.max(norm_probabilities)

            # Calculate FES
            # F(s) = - 1/beta * log (P/zed + epsilon)
            fes = -unitfactor * np.log(norm_probabilities / max_prob)

        # Using Torch (GPU or CPU)
        elif device in ['cuda', 'mps', 'cpu']:
            # It is now possible to run the calculations using Pytorch (cuda, mps, cpu) 
            # Convert to Pytorch tensors
            grid_combos = torch.Tensor(grid_combos).to(device)
            centers = torch.Tensor(centers).to(device)
            sigmas = torch.Tensor(sigmas).to(device)
            height = torch.Tensor(height).to(device)

            # When running torch the size of the tensors can get very big, overloading the memory [ERROR: Invalid buffer size: 30.19 GB].
            # If this is the case, then running the script in batches. The code does at most 6 attempts to cut the data in batches and try to do the analysis.
            # The number of batches goes by 2^n. So 2 batches, 4, 8, 16, 32, 64. Running this in batches on the GPU appears to still be quicker than the looped structure.
            try:
                # delta_s = s - s0 / sigma
                delta_s = (grid_combos.unsqueeze(-1) - centers.T) / sigmas.T
                
                # "the Gaussian function is defined as G(s, s0) = h exp −0.5 SUM(s − s0)/sigma)^2 (eq. 2.64)
                gauss_function = height * (torch.maximum(torch.exp(-0.5 * torch.sum(delta_s**2, axis=1)) - val_at_cutoff, torch.Tensor([0]).to(device)))
            
                # "However, P˜(s) should be normalized not with respect to the full CV space but only over the CV space actually explored up to step n, which we call Wn. Thus, we introduce the normalization factor Z (zed). Finally, we can explicitly write the bias at the nth step as eq (2.63) where epsilon can be seen as a regularization term that ensures that the argument of the logarithm is always greater than zero."
                norm_probabilities = torch.sum(gauss_function, axis=1)/zed + epsilon
            
            except Exception as e:
                # print("\nWARNING: ", e) if verbose else 0
                size = grid_combos.size()[0]
                # print(f"This probably means that the size of the tensor ({size}) is too large for memory.\n\nTrying to calculate in batches.") if verbose else 0

                attempt = 1
                while attempt <= 6:
                    # Split into batches.
                    batches = torch.split(grid_combos, round(size / (2 ** attempt)))

                    # print(f"--> Attempt {attempt}: {len(batches)} batches...", end="", flush=True) if verbose else 0

                    batch_np = []
                    try:  
                        for batch in tqdm(batches, total=len(batches), leave=False):
                            # Do the same steps as described above, but do it in batches.
                            batch_ds = (batch.unsqueeze(-1) - centers.T) / sigmas.T
                        
                            batch_gf = height * (torch.maximum(torch.exp(-0.5 * torch.sum(batch_ds**2, axis=1)) - val_at_cutoff, torch.Tensor([0]).to(device)))
                        
                            batch_np.append(torch.sum(batch_gf, axis=1)/zed + epsilon)

                        norm_probabilities = torch.cat(batch_np)

                        break
                    except Exception as e:
                        # print(f"ERROR: ({e})") if verbose else 0
                        
                        if attempt >= 6:
                            print(f"ERROR: Doing calculation in batches didn't work. Try to solve issue or use other device.") if verbose else 0
                            exit(1)
                        else:
                            attempt += 1
                                    
            # Reshape back into n-dimensional tensor
            norm_probabilities = torch.reshape(norm_probabilities, n_bins)

            # Set minimum value to zero
            if not mintozero:
                max_prob = 1
            else:
                max_prob = torch.max(norm_probabilities)

            # Calculate FES 
            # F(s) = - 1/beta * log (P/zed + epsilon)
            fes = -unitfactor * torch.log(norm_probabilities / max_prob)

            # Transform back to numpy array
            fes = fes.cpu().detach().numpy()
            grid_combos = grid_combos.cpu().detach().numpy()

        # Use direct numpy array calculation
        elif device in ['numpy']:

            # delta_s = s - s0 / sigma
            delta_s = (np.expand_dims(grid_combos, -1) - centers.T) / sigmas.T

            # "the Gaussian function is defined as G(s, s0) = h exp −0.5 SUM(s − s0)/sigma)^2 (eq. 2.64)
            gauss_function = height * (np.maximum(np.exp(-0.5 * np.sum(delta_s**2, axis=1)) - val_at_cutoff, 0))

            # "However, P˜(s) should be normalized not with respect to the full CV space but only over the CV space actually explored up to step n, which we call Wn. Thus, we introduce the normalization factor Z (zed). Finally, we can explicitly write the bias at the nth step as eq (2.63) where epsilon can be seen as a regularization term that ensures that the argument of the logarithm is always greater than zero."
            norm_probabilities = np.sum(gauss_function, axis=1)/zed + epsilon
            
            # Reshape back into n-dimensional tensor
            norm_probabilities = np.reshape(norm_probabilities, n_bins)

            # Set minimum value to zero
            if not mintozero:
                max_prob = 1
            else:
                max_prob = np.max(norm_probabilities)

            # Calculate FES 
            # F(s) = - 1/beta * log (P/zed + epsilon)
            fes = -unitfactor * np.log(norm_probabilities / max_prob)

        # Otherwise error
        else:
            sys.exit(f"ERROR: Unknown device: {device}")
            
        # Add all fes data to a np array. Later used to make a single FES file.          
        fes_data_array = np.column_stack((grid_combos, fes.flatten(), [time] * len(grid_combos)))

        # Try to append, if the list does not yet exist, create it.
        try:
            fes_complete = np.append(fes_complete, fes_data_array, axis=0)
        except:
            fes_complete = fes_data_array

    fes_df = pd.DataFrame(fes_complete, columns = list(cvs) + ['fes', 'time'], dtype = float)

    return fes_df

def from_colvar(colvar,
               cvs=None,
               sigmas=None,
               process='last',
               mintozero=True,
               temp=310,
               units='kT',
               bins=100,
               device='mt',
               verbose=True):
    """
    Calculate the free energy through reweighting of the COLVAR file. 
        
    :param colvar: the dataframe created from the COLVAR file.
    :param cvs: CV(s) to calculate the FES for, determines dimensionality. If None, find them in the COLVAR df.
    :param sigmas: Corresponding sigmas. If None, find them in other files.
    :param process: Options: -'all' process all frames
                             -'last' process only last frame 
                             - n (int) process n frames evenly spread out over the trajectory.')
    :param mintozero: Shift the minimum to zero
    :param temp: Temperature in Kelvin
    :param units: Output units. 
                  Options: -'kT'
                           -'kJ/mol'
                           -'kcal/mol'.
    :param bins: Number of bins per each cv: int, tuple, list or comma separated string.
    :param device: How to run calculations. 
                   Options: -'mt' multithreaded looped code structure (fastest for smaller sets). 
                            -'np' performs all calculations using numpy arrays. 
                            -'torch' offloads to GPU (CUDA or MLS, the MacOS Intel GPU). If no GPU is available it runs on the CPUs.
    :param verbose: Print output when running.
  
    :returns: df containing columns '[cv]' and 'time'
    """

    # Calculate unitfactor (units conversion factor)
    unitfactor = tools.get_unitfactor(units=units, temp=temp)
    
    # Define cvs, either from user input, or, if cvs are not defined, take the cvs from file
    cvs = tools.define_cvs(cvs, colvar, type='colvar')
    sigmas_dict = tools.define_sigmas(sigmas, cvs)

    # Get dimensionality
    dimensions = len(cvs)
    print(f"Calculating FES of {' and '.join(cvs)} ({dimensions} dimension(s))") if verbose else 0

    # Get list of all different times, which corresponds to unique state and scrub to only keep the states you want to analyse.
    times_list = tools.scrub_times_list(colvar['time'].unique(), process, verbose)

    # Get only relevant data from dataframe (time, cvs and bias)
    subset_data = colvar[["time"] + list(cvs) + ["opes.bias"]]
        
    # Loop over all states in times_list
    for index, colvar_time in tqdm(enumerate(times_list), desc=f"Working on state", total=len(times_list), unit=""):
        # Extract the data up until that state (with time colvar_time).
        relevant_data = subset_data[subset_data['time'].between(0, colvar_time)]

        # Get kernels. They are stored in lists where the index of the list is the CV number.
        centers = np.array(relevant_data[[cv for cv in relevant_data.columns if cv in cvs]].astype(float).values)

        # Sigmas are given. Make a vector as long as the dataframe.
        sigmas = np.array([np.repeat(sigmas_dict[k], relevant_data.shape[0], axis=0) for k in cvs]).T
        
        # Get the bias values in unitless factor.
        # Given is in kJ/mol, so unitfactor is not used but the kbt for kJ/mol.
        bias = relevant_data['opes.bias'].values / (kb * NA * temp / 1000)

        # Prepare the grid
        # Define the bounds (min, max) and number of bins for each state
        # Where axis=0 refers to columns
        grid_min = centers.min(axis=0)
        grid_max = centers.max(axis=0)

        # Setup the grid
        n_bins, mgrid = tools.setup_grid(grid_min, grid_max, bins, dimensions)
        grid_combos = np.array(mgrid.T.reshape(-1, dimensions))

        # Standard looped algorithm
        if device in ['mt']:
            
            def bias_to_fes(combo):
                ''' Nested function. This function has direct access to variables and names defined in the enclosing function.
                This function is used to calculate the FES from the bias. Is called below by the multiprocessing pool.'''

                delta_s = (combo[:, np.newaxis] - centers.T)/sigmas.T

                args = bias - 0.5 * np.sum(delta_s**2, axis=0)

                return -unitfactor * np.logaddexp.reduce(args)

            # Calculate FES using multithreading by calling the nested function above
            with ThreadPool() as pool:
                fes = list(tqdm(pool.imap(bias_to_fes, grid_combos), total=len(grid_combos), desc=f"Gridpoint", unit="", leave=False))

        # Using Torch (GPU or CPU)
        elif device in ['cuda', 'mps', 'cpu']:            
            # It is now possible to run the calculations using Pytorch (cuda, mps, cpu) 
            # Convert to Pytorch tensors
            grid_combos = torch.Tensor(grid_combos).to(device)
            centers = torch.Tensor(centers).to(device)
            sigmas = torch.Tensor(sigmas).to(device)
            bias = torch.Tensor(bias).to(device)
            
            # When running torch the size of the tensors can get very big, overloading the memory [ERROR: Invalid buffer size: 30.19 GB].
            # If this is the case, then running the script in batches. The code does at most 6 attempts to cut the data in batches and try to do the analysis.
            # The number of batches goes by 2^n. So 2 batches, 4, 8, 16, 32, 64. Running this on the GPU appears to still be quicker than the looped structure. (Below) 
            try:
                # delta_s = s - s0 / sigma
                delta_s = (grid_combos.unsqueeze(-1) - centers.T) / sigmas.T

                args = bias - 0.5 * torch.sum(delta_s**2, axis=1)

                # Calculate fes and turn back into numpy array.
                fes_tensor = -unitfactor * torch.logsumexp(args, dim=-1)
                fes = fes_tensor.cpu().detach().numpy()
                grid_combos = grid_combos.cpu().detach().numpy()
                
            except Exception as e:
                # print("\nWARNING: ", e) if verbose else 0
                size = grid_combos.size()[0]
                # print(f"This probably means that the size of the tensor ({size}) is too large for memory.\n\nTrying to calculate in batches.") if verbose else 0

                attempt = 1
                while attempt <= 6:
                    # Split into batches.
                    batches = torch.split(grid_combos, round(size / (2 ** attempt)))

                    # print(f"--> Attempt {attempt}: {len(batches)} batches...", end="", flush=True) if verbose else 0
                    batch_fes = []
                    try:  
                        for batch in tqdm(batches, total=len(batches), leave=False):

                            # Do the same steps as described above, but do it in batches.
                            batch_ds = (batch.unsqueeze(-1) - centers.T) / sigmas.T
                        
                            batch_args = bias - 0.5 * torch.sum(batch_ds**2, axis=1)

                            # Calculate fes.
                            batch_fes.append(-unitfactor * torch.logsumexp(batch_args, dim=-1))

                        # Turn back into numpy array.
                        fes = torch.cat(batch_fes).cpu().detach().numpy()                         
                        grid_combos = grid_combos.cpu().detach().numpy()

                        break
                    except Exception as e:
                        # print(f"ERROR: ({e})") if verbose else 0
                        
                        if attempt >= 6:
                            print(f"ERROR: Doing calculation in batches didn't work. Try to solve issue or use other device.") if verbose else 0
                            exit(1)
                        else:
                            attempt += 1

        # Use direct numpy array calculation
        elif device in ['numpy']:
            print(f"Working on state {(index + 1)} of {len(times_list)}\t   ", end='\r') if verbose else 0

            # delta_s = s - s0 / sigma
            delta_s = (np.expand_dims(grid_combos, -1) - centers.T) / sigmas.T

            args = bias - 0.5 * np.sum(delta_s**2, axis=1)

            # Calculate fes and turn back into numpy array.
            fes = -unitfactor * np.logaddexp.reduce(args, axis=-1)
                
        # Otherwise error
        else:
            sys.exit(f"ERROR: Unknown device: {device}")

        # Set minimum value to zero
        if mintozero:
            fes += abs(np.min(fes))

        # Add all fes data to a np array. Later used to make a single FES file.
        fes_data_array = np.column_stack((grid_combos, fes, [colvar_time]*len(grid_combos)))

        # Try to append, if the list does not yet exist, create it.
        try:
            fes_complete = np.append(fes_complete, fes_data_array, axis=0)
        except:
            fes_complete = fes_data_array

    fes_df = pd.DataFrame(fes_complete, columns = list(cvs) + ['fes', 'time'], dtype = float)
    
    return fes_df

# def from_kernels(kernels, cvs, grid_info, process, mintozero, unitfactor, T, fes_prefix, device, fmt):
#     ''' Get free energy surface estimation from the collective variables file (COLVAR).
#         These weights can be used for reweighting other cvs. 
        
#         Inputs:
#         --> colvar: the dataframe created from the COLVAR file.
#         --> T: temperature in K'''


#     # The powerset of a set S is the set of all subsets of S, including the empty set and S itself
#     # First set is the empty set, so it's removed with [1:]
#     cvs_powerset = list(tools.powerset(cvs))[1:] 

#     # Get list of all different times, which corresponds to unique states (time is the index of the df)
#     kernels_time_list = kernels['time'].unique()
   
#     # Check if there are points
#     if len(kernels_time_list) == 0:
#         sys.exit(f"ERROR: No time points found.\nNow exiting")

#     # See what kernels to analyse
#     if process == 'last':
#         # Process only last frame
#         kernels_time_list = [kernels_time_list[-1]]
#         print(f"\t\tKeeping last state time points") if verbose else 0
#     elif process == 'all' or len(kernels_time_list) <= int(process):
#         # If you have less frames than you want to keep or want to keep all frames  
#         print(f"\t\tKeeping all ({len(kernels_time_list)}) time points") if verbose else 0
#     elif len(kernels_time_list) >= int(process):
#         # Striding the list of times to analyse.
#         last = kernels_time_list[-1]
#         total = len(kernels_time_list)
#         stride = int(np.ceil(len(kernels_time_list) / float(process)))
#         kernels_time_list = kernels_time_list[::stride]
    
#         # Note: I've decided to always add the last frame which is the "final" state, this might give a small discontinuity in the timesteps between the last two frames.
#         print(f"\t\tKeeping {len(kernels_time_list)} of {total} states (except last state)") if verbose else 0
#         if kernels_time_list[-1] != last:
#             kernels_time_list = np.concatenate((kernels_time_list, [last]), axis=None)
#             print(f"\t\tNOTE: last frame was added to the end. This might give a small discontinuity in the timesteps between the last two frames.\n") if verbose else 0
#     else:
#         sys.exit(f"ERROR: Something went wrong when striding.")

#     # Make a list in which you keep all FES dataframes to return to main
#     fes_all = []

#     # Loop over all combination of the powerset
#     for cvs in cvs_powerset:
        
#         dimensions = len(cvs)
#         print(f"\t\tstarting with FES of {' and '.join(cvs)} ({dimensions} dimension(s))") if verbose else 0

#         # Get only relevant data from dataframe (time, cvs, sigma_cvs and height)
#         subset_data = kernels[["time"] + list(cvs) + ["sigma_" + s for s in cvs] + ["logweight"]]

#         # Loop over all states in kernels_time_list
#         for index, kernels_time in enumerate(kernels_time_list):

#             # Extract the data up until that state (with time kernels_time).
#             relevant_data = subset_data[subset_data['time'].between(0, kernels_time)]

#             # The arrays have the shape n_cvs x n_points, except the hight, which is 1D (list).
#             # I use the cv_order_key to make sure the order of the columns is the same as in the dataframe (or plumed output file).
#             centers = np.array(relevant_data[[cv for cv in relevant_data.columns if cv in cvs]].astype(float).values)

#             sigmas = np.array(relevant_data[[f"sigma_{cv}" for cv in relevant_data.columns if cv in cvs]].astype(float).values)

#             # Get the bias values in unitless factor.
#             # Given is in kJ/mol, so unitfactor is not used but the kbt for kJ/mol.
#             bias = relevant_data['logweight'].values / (kb * NA * T / 1000)

#             # Prepare the grid
#             grid_min = grid_info.loc['grid_min', list(cvs)].tolist()
#             grid_max = grid_info.loc['grid_max', list(cvs)].tolist()
#             n_bins = grid_info.loc['n_bins', list(cvs)].astype(int).tolist()

#             # If needed, redefine the bounds (min, max) for each state
#             if np.isnan(grid_min).any():
#                 # Where axis=0 refers to columns
#                 grid_min = centers.min(axis=0)
#             if np.isnan(grid_max).any():
#                 # Where axis=0 refers to columns
#                 grid_max = centers.max(axis=0)

#             # Define a list with the bounds for each cv
#             # [[cv1-min cv1-max] [cv2-min cv2-max] ... [cvn-min cvn-max]]]
#             bounds = list(zip(grid_min, grid_max))

#             # Make n dimensional meshgrid, where the dimension represents the cvs.
#             # Then make all possible combinations of the n dimensions (n cvs)
#             mgrid = np.mgrid[[slice(row[0], row[1], n*1j) for row, n in zip(bounds, n_bins)]]

#             grid_combos = np.array(mgrid.T.reshape(-1, dimensions))
#             total_len = len(grid_combos)

#             # It is now possible to run the calculations using Pytorch (cuda, mps, cpu) 
#             # Convert to Pytorch tensors
#             grid_combos = torch.Tensor(grid_combos).to(device)
#             centers = torch.Tensor(centers).to(device)
#             sigmas = torch.Tensor(sigmas).to(device)
#             bias = torch.Tensor(bias).to(device)
                
#             # When running torch the size of the tensors can get very big, overloading the memory [ERROR: Invalid buffer size: 30.19 GB].
#             # If this is the case, then running the script in batches. The code does at most 6 attempts to cut the data in batches and try to do the analysis.
#             # The number of batches goes by 2^n. So 2 batches, 4, 8, 16, 32, 64. Running this on the GPU appears to still be quicker than the looped structure. (Below) 
#             try:
#                 print(f"\t\tWorking on state {(index + 1)} of {len(kernels_time_list)}\t   ", end='\r') if verbose else 0

#                 # delta_s = s - s0 / sigma
#                 delta_s = (grid_combos.unsqueeze(-1) - centers.T) / sigmas.T

#                 args = bias - 0.5 * torch.sum(delta_s**2, axis=1)

#                 # Calculate fes and turn back into numpy array.
#                 fes_tensor = -unitfactor * torch.logsumexp(args, dim=-1)
#                 fes = fes_tensor.cpu().detach().numpy()
#                 grid_combos = grid_combos.cpu().detach().numpy()
                
#             except Exception as e:
#                 print("\nWARNING: ", e) if verbose else 0
#                 size = grid_combos.size()[0]
#                 print(f"This probably means that the size of the tensor ({size}) is too large for memory.\n\nTrying to calculate in batches.") if verbose else 0

#                 attempt = 1
#                 while attempt <= 6:
#                     # Split into batches.
#                     batches = torch.split(grid_combos, round(size / (2 ** attempt)))

#                     # print(f"--> Attempt {attempt}: {len(batches)} batches...", end="", flush=True) if verbose else 0
#                     batch_fes = []
#                     try:  
#                         for b_index, batch in enumerate(batches):
#                             print(f"\t\tWorking on state {(index + 1)} of {len(kernels_time_list)}\t | {(b_index/len(batches)):.0%}                        ", end="\r") if verbose else 0

#                             # Do the same steps as described above, but do it in batches.
#                             batch_ds = (batch.unsqueeze(-1) - centers) / sigmas
                        
#                             batch_args = bias - 0.5 * torch.sum(batch_ds**2, axis=1)

#                             # Calculate fes.
#                             batch_fes.append(-unitfactor * torch.logsumexp(batch_args, dim=-1))

#                         # Turn back into numpy array.
#                         fes = torch.cat(batch_fes).cpu().detach().numpy()                         
#                         grid_combos = grid_combos.cpu().detach().numpy()

#                         print(f"\t\tWorking on state {(index + 1)} of {len(kernels_time_list)}\t | {(len(batches)/len(batches)):.0%}                        ", end="") if verbose else 0
#                         break
#                     except Exception as e:
#                         # print(f"ERROR: ({e})") if verbose else 0
                        
#                         if attempt >= 6:
#                             print(f"ERROR: Doing calculation in batches didn't work. Try to solve issue or use other device.") if verbose else 0
#                             exit(1)
#                         else:
#                             attempt += 1

#         # Set minimum value to zero
#         if mintozero:
#             fes += abs(np.min(fes))

#         # Add all fes data to a np array. Later used to make a single FES file.
#         fes_data_array = np.column_stack((grid_combos, fes, [kernels_time]*len(grid_combos)))
        
#         if index == 0:
#             fes_complete = fes_data_array
#         else:
#             fes_complete = np.append(fes_complete, fes_data_array, axis=0)
        

#         fes_data = pd.DataFrame(fes_complete, columns = list(cvs) + ['fes', 'time'], dtype = float)
#         fes_data.to_csv(f"{fes_prefix}_{'_'.join(cvs)}_kernels", index=False, sep='\t', float_format=fmt)
        
#         # Erase and go to beginning of line
#         sys.stdout.write('\033[2K\033[1G')
#         print(f"\t\t\t\t--> outputfile: {fes_prefix}_{'_'.join(cvs)}_kernels\n") if verbose else 0

#         # Add to the list of dataframes to return
#         fes_all.append(fes_data)

#     return fes_all

def from_weights(df,
                 weights_df=None,
                 colvar=None,
                 bins=101,
                 units='kT',
                 temp=310,
                 mintozero=True,
                 time=0):
                 
    """
    Calculate a FES estimation for any cv/timeseries using the weights determine. 
    colvar
    :param df: Dataframe containing the cv(s) timeseries. Needs columns 'time', '[cvs]', 'origin'.
    :param weights: Dataframe containg the weights for each cv. Column headers must correspond. At least needed. 'time', 'weights', 'origin'. If not found, it will be calculated from provided colvar file.
    :param colvar: the dataframe created from the COLVAR file. Used to call weights function.
    :param units: Output units. 
                  Options: -'kT'
                           -'kJ/mol'
                           -'kcal/mol'.
    :param temp: temperature in K
    :param mintozero: Shift the minimum to zero
    :param time: Give the time corresponding to this fes.

    
    :returns: df containing '[cv(s)]', 'dist_unweighted', 'dist_reweighted', 'fes'
    """

    # If weights are not given, try getting them from colvar file.
    try:
        # Make a copy to make sure you're not editing the original df.
        weights_df = weights_df.copy()

    except:
        try:
            print("No weights given. Trying to fetch from colvar.")
            weights_df = weights(colvar)
            # print(weights_df)

        except Exception as e:
            sys.exit(f"ERROR: {e}\n\nWeights ({weights_df}) and/or colvar ({colvar}) invalid. Provide at least one of the two.")
    
    # Make a copy to make sure you're not editing the original df.
    df = df.copy()

    # Merge weights and dataframe
    df = df.merge(weights_df, how="inner", on=["time", "origin"])

    # First find the cvs in the input dataframe, this determines dimensionality
    column_names = df.columns.to_list()
    cvs = [cv for cv in column_names if cv not in ['time', 'fes', 'origin', 'weights']]
    dimensions = len(cvs)

    # Calculate unitfactor (units conversion factor)
    unitfactor = tools.get_unitfactor(units=units, temp=temp)

    # Prepare the grid
    # Define the bounds (min, max) and number of bins for each state
    grid_min = df[cvs].min().values
    grid_max = df[cvs].max().values

    # Setup the grid
    n_bins, mgrid = tools.setup_grid(grid_min, grid_max, bins, dimensions)

    # 1D
    if dimensions == 1:
       
        # Calculate unweighted histogram (dist_unweighted)
        hist_u, bins_u = np.histogram(df[cvs[0]], bins=n_bins[0])
        hist_u = hist_u / hist_u.sum() # Normalize (sum is 1)

        # Calculate weighted histogram (dist_reweighted)
        hist_w, bins_w = np.histogram(df[cvs[0]], bins=n_bins[0], weights=df['weights'])
        hist_w = hist_w / hist_w.sum() # Normalize (sum is 1)

        # Get bin centers
        bin_centers = (bins_w[:-1] + bins_w[1:]) / 2

        # Calculate FES (and ignore divide by zero error.)
        with np.errstate(divide='ignore'):
            fes = -unitfactor * np.log(hist_w)

        # If needed, set minimum to zero
        if mintozero:
            fes = fes - np.min(fes)
  
        fes_df = pd.DataFrame(
                    {cvs[0] : bin_centers,
                     'dist_unweighted' : hist_u,
                     'dist_reweighted' : hist_w,
                     'fes' : fes})
        
        fes_df['time'] = time

        return fes_df

    elif len(cvs) == 2:
        
        # Setup bins
        x_bins = np.linspace(grid_min[0], grid_max[0], n_bins[0])
        y_bins = np.linspace(grid_min[1], grid_max[1], n_bins[1])
        
        # Calculate unweighted histogram (dist_unweighted)
        hist_u, x_bins, y_bins = np.histogram2d(df[cvs[0]], df[cvs[1]], bins=(x_bins, y_bins))
        # Histogram does not follow Cartesian convention,
        # therefore transpose H for visualization purposes.
        hist_u = hist_u.T
        hist_u = hist_u / hist_u.sum() # Normalize (sum is 1)

        # Calculate weighted histogram (dist_weighted)
        hist_w, x_bins, y_bins = np.histogram2d(df[cvs[0]], df[cvs[1]], bins=(x_bins, y_bins), weights=df['weights'])
        
        # Histogram does not follow Cartesian convention,
        # therefore transpose H for visualization purposes.
        hist_w = hist_w.T
        hist_w = hist_w / hist_w.sum() # Normalize (sum is 1)

        # Get the center values of the bins
        X, Y = np.meshgrid(x_bins, y_bins)
        X_c = (x_bins[:-1] + x_bins[1:]) / 2
        Y_c = (y_bins[:-1] + y_bins[1:]) / 2

        # Calculate FES (and ignore divide by zero error.)
        with np.errstate(divide='ignore'):
            fes = -unitfactor * np.log(hist_w)

        # If needed, set minimum to zero
        if mintozero:
            fes = fes - np.min(fes)

        # Turn into dataframe
        fes_df = pd.DataFrame(fes.T, columns=Y_c, index=X_c)
        
        # Move index to column and melt df
        fes_df = fes_df.reset_index(names=cvs[0])
        fes_df = fes_df.melt(id_vars=cvs[0], var_name=cvs[1], value_name='fes')

        fes_df['time'] = time

        return fes_df

    # Other dimensions are not yet implemented
    else:
        raise Exception("More than 2 dimensions not (yet) supported for this plot type.")
