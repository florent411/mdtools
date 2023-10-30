#!/usr/bin/env python 

''' General tools for opes_postprocessing '''

import os
import sys
import torch
import numpy as np
from itertools import islice
from curses.ascii import isdigit
from itertools import chain, combinations

def powerset(iterable):
    '''Determine the powerset. The powerset of a set S is the set of all subsets of S, including the empty set and S itself.'''
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)

    # Powerset as a list of tuples
    ps_lop = chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    
    # Powerset as a list of lists
    powerset = [list(tup) for tup in ps_lop]

    return powerset

def find_device():
    ''' Check where to run the calculations '''
    if torch.cuda.is_available():
        device = 'cuda'
        device_name = torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        device = 'mps'
        device_name = 'MacOS GPU'
    else:
        device = 'cpu'
        device_name = 'CPU (PyTorch)'
    return device, device_name

def get_unitfactor(units, temp=310):
    ''' Calculate the conversion factor to switch between units.'''
    
    # Constants
    kb = 1.38064852e-23 # Boltzman's constant in m^2 kg s^-2 K^-1
    NA = 6.02214086e23 # Avogadro's constant in mol^-1
    
    # Set
    if units == 'kJ/mol':
        # 1 kT = 0.008314459 kJ/mol
        unitfactor = kb * NA * temp / 1000
    elif units == 'kcal/mol':
        # 1 kJ/mol = 0.238846 kcal/mol
        unitfactor = 0.238846 * kb * NA * temp / 1000
    elif units == 'kT':
        # 1 kT = 1 kT
        unitfactor = 1

    return unitfactor

def scrub_times_list(times_list, process_max, verbose):
    ''' See what needs to processed and return a times_list containing only the states you want to analyse. 
        process_max = 'last' - Keep only the last states
        process_max = 'all' - Keep all states
        process_max = int - Keep maximum this number of states. Stride to fit. Note: I've decided to always add the last frame which is the "final" state, this might give a small discontinuity in the timesteps between the last two frames.
    '''

    # Check process_max value
    if isinstance(process_max, int):
        pass
    elif process_max.isdigit() and int(process_max) > 0:
        process_max = int(process_max)
    elif process_max not in ['all', 'last']:
        sys.exit(f"ERROR: Illegal value found for process_max ({process_max}). Please choose 'all', 'last' or a positive integer\nNow exiting")

    # See what state to analyse
    if process_max == 'last':
        # Process only last frame
        times_list = [times_list[-1]]
        print(f"Keeping last state only") if verbose else 0 
    elif process_max == 'all' or len(times_list) <= int(process_max):
        # If you have less frames than you want to keep or want to keep all frames  
        print(f"Keeping all ({len(times_list)}) state") if verbose else 0
    elif len(times_list) >= int(process_max):
        # Striding the list of times to analyse.
        last = times_list[-1]
        total = len(times_list)
        stride = int(np.ceil(len(times_list) / float(process_max)))
        times_list = times_list[::stride]
    
        # Note: I've decided to always add the last frame which is the "final" state, this might give a small discontinuity in the timesteps between the last two frames.
        print(f"Keeping {len(times_list)} of {total} state (+ last state)") if verbose else 0
        if times_list[-1] != last:
            times_list = np.concatenate((times_list, [last]), axis=None)
            print(f"!NOTE: last frame was added to the end. This might give a small discontinuity in the timesteps between the last two frames.\n") if verbose else 0
    else:
        sys.exit(f"ERROR: Something went wrong when striding.")

    return times_list

def setup_grid(grid_min, grid_max, bins, dimensions):
    ''' Setup the grid '''

    # Turn n_bins into list
    if isinstance(bins, int):
        n_bins = [bins]
    elif isinstance(bins, tuple):
        n_bins = list(bins)
    elif isinstance(bins, str):
        n_bins = list(map(int, bins.split(','))) # list of ints

    if len(n_bins) != dimensions:
        if len(n_bins) == 1 and len(n_bins) != dimensions:
            n_bins *= dimensions            
        else:
            sys.exit(f"ERROR: Given number of bins ({bins}) does not fit dimensionality ({dimensions}) of cvs.")

    # Define a list with the bounds for each cv
    # [[cv1-min cv1-max] [cv2-min cv2-max] ... [cvn-min cvn-max]]]
    bounds = list(zip(grid_min, grid_max))

    # Make n dimensional meshgrid, where the dimension represents the cvs.
    # Then make all possible combinations of the n dimensions (n cvs)
    mgrid = np.mgrid[[slice(row[0], row[1], n*1j) for row, n in zip(bounds, n_bins)]]

    return n_bins, mgrid

def define_cvs(cvs, input_df, type):
    ''' Define cvs, either from user input, or, if cvs are not defined, take the cvs from file (the input dataframe) '''

    # First find the cvs in the input dataframe (which comes from a plumed output file (e.g. STATE, COLVAR, KERNELS))
    if type == 'state':
        # Filter out any column name starting with sigma and the height and time columns 
        column_names = input_df.columns.to_list()
        cvs_from_file = [x for x in column_names if not x.startswith("sigma") and x not in ['height', 'time']]
    elif type == 'colvar':
        # I'm taking the column name between the columns 'time' and 'opes.bias', which is where I always find my cvs as defined in plumed.dat
        # PRINT STRIDE=500 FILE=COLVAR ARG=cv_1,(...),cv_n,opes.*,other_variables
        # If you do this differently the results can differ.
        cvs_from_file = input_df.loc[:, 'time':'opes.bias'].columns.values[1:-1]
    else:
        sys.exit(f"ERROR: type ({type}) not (yet) supported.")

    # If no input for cvs (cvs == None), just use cvs as determined above, otherwise use the user input (making sure its a subset of the cvs in the input)
    if cvs == None:
        cvs = cvs_from_file
    # Otherwise, to keep the right order (as in the state/colvar file), I find the cvs in the state file and compare them with the user input.
    else:
        # Turn into a list
        if isinstance(cvs, list):
            pass 
        elif isinstance(cvs, tuple):
            cvs_from_user = list(cvs)
        elif isinstance(cvs, str):
            cvs_from_user = list(map(str, cvs.split(','))) # list of strings
        else:
            sys.exit(f"ERROR: inputtype ({type(cvs)}) not (yet) supported. Use a list, tuple or comma separated string")
        
        # Check if user input is a subset of the cvs found in the state file
        if not all(elem in cvs_from_file for elem in cvs_from_user):
            sys.exit(f"ERROR: CV input ({cvs_from_user}) should be subset of cvs in the state file ({cvs_from_file}).")

        # Take the intersection, with the order as in cvs_from_file.
        cvs = [x for x in cvs_from_file if x in cvs_from_user]

    return cvs

def define_sigmas(sigmas, cvs):
    ''' Setup sigmas or see if you can find the sigma values in the first n lines of a file.'''

    if sigmas != None:
        # Turn into a list
        if isinstance(sigmas, list):
            sigmas = list(map(float, sigmas))
        elif isinstance(sigmas, tuple):
            sigmas = list(map(float, sigmas))
        elif isinstance(sigmas, str):
            sigmas = map(float, sigmas.split(',')) # list of floats
        else:
            sys.exit(f"ERROR: inputtype ({type(sigmas)}) not (yet) supported. Use a list, tuple or comma separated string")

        # Return a dict containing the names and values of the cvs and their sigmas.
        try:
            sigmas_dict = dict(zip(cvs, sigmas))
            return sigmas_dict
    
        except Exception as e:
            sys.exit(f"ERROR: {e}\n Probably, the number of sigmas give ({sigmas}) does not fit the number of cvs ({cvs}).")
    
    # If the sigmas are not defined, look for them in files (KERNELS and STATE file)
    else:
        # Look in Kernels file
        try:
            with open('KERNELS', 'rb') as f:

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
                sigmas_dict = {k.split('_')[-1] : float(v) for k, v in result.items() if 'sigma' in k}
                
                # Check if the found cvs are at least a subset of the found cvs
                return sigmas_dict if all(elem in list(sigmas_dict.keys()) for elem in cvs) else 0

        except:
            try:
                sigmas = {}
                with open('STATE', 'rb') as f:
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
                                    sigmas_dict[line.split()[-2].split('_')[-1]] = float(line.split()[-1])
                            
                        # Check if the found cvs are at least a subset of the found cvs
                        return sigmas_dict if all(elem in list(sigmas_dict.keys()) for elem in cvs) else 0

            except:
                sys.exit(f"ERROR: No sigmas given and can't find sigmas in ./KERNELS or ./STATE.")    


def get_column_names(filename):
    with open(filename, "r") as file:
        first_line = file.readline()            
    column_names = first_line.strip().split(" ")[2:]

    return column_names

def kldiv(p, q):
    """Kullback-Leibler divergence D(P || Q) for discrete distributions
    
    Parameters
    ----------
    p, q : array-like, dtype=float, shape=n
    Discrete probability distributions.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def jsdiv(p, q):
    """Jensen–Shannon divergence for measuring the similarity between two probability distributions. 
    It is based on the Kullback–Leibler divergence, with differences such as symmetry and it always has a finite value.

    Parameters
    ----------
    p, q : array-like, dtype=float, shape=n
    Discrete probability distributions.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    m = (p + q) / 2

    # Calculate the Kullback-Leibler divergences: D(P||M) and D(Q||M)
    Dpm = np.sum(np.where(p != 0, p * np.log(p / m), 0))
    Dqm = np.sum(np.where(q != 0, q * np.log(q / m), 0))

    return 0.5 * Dpm + 0.5 * Dqm

def dalonso(v1, v2):
    """Alonso divergence (dA) - A Physically Meaningful Method for the Comparison of Potential Energy Functions
    
    Parameters
    ----------
    v1, v2 : array-like, dtype=float, shape=n
    Discrete probability distributions.
    """
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)

    v1_var, v2_var = v1.var(), v2.var()
    r = np.corrcoef(v1, v2)[0,1]

    return np.sqrt((v1_var + v2_var) * (1 - r**2))
