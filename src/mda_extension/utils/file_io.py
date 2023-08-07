#!/usr/bin/env python 

''' File in/out '''

import sys

import numpy as np
import pandas as pd

# Included submodules
import mda_extension.utils.tools as tools

def read_colvar(root, walker_paths, colvar='COLVAR', labels=None, verbose=False):
    """Read COLVAR file(s) into dataframe.

    :param root: Main folder.
    :param walker_paths: List of path to all walkers. (/. if only 1 walker.)
    :param colvar: Name of the COLVAR file. (Default value = 'COLVAR')
    :param labels: Corresponding labels. (Default value = None)
    :param verbose: Print what's happening. (Default value = False)

    :returns: dataframe of the COLVAR file
    """

    n_walkers = len(walker_paths)

    # Catch None values
    if labels == None:
        labels = [*range(n_walkers)]
    elif type(labels) is not list:
        labels = [labels]

    # If you have one label for multiple universes, use it as a prefix.
    if len(labels) == 1 and n_walkers != 1:
        labels = [f"{labels[0]}{i}" for i in range(n_walkers)]

    if n_walkers == 1:
        # Get the names of the columns
        with open(f"{root}/{colvar}", "r") as file:
            first_line = file.readline()            
        column_names = first_line.strip().split(" ")[2:]

        print(f"\t-> {root}/{colvar}...", end="") if verbose else 0
        df = pd.read_csv(f"{root}/{colvar}", names=column_names, delim_whitespace=True, comment="#")
        
        # Add label and frame number
        df['origin'] = labels[0]
        df['frame'] = df.index

        print(f"done") if verbose else 0

    elif n_walkers > 1:      
        print(f"\t-> ./walker{','.join([str(i) for i in range(n_walkers)])}/COLVAR (combining {n_walkers} walkers)...", end="") if verbose else 0
        
        # Load the dataframe for each walker
        colvar_list = []
        for i, path in enumerate(walker_paths):

            # Get the names of the columns
            with open(f"{root}/{path}/{colvar}.{i}", "r") as file:
                first_line = file.readline()            
            column_names = first_line.strip().split(" ")[2:]

            # Add walker number
            walker_df = pd.read_csv(f"{root}/{path}/{colvar}.{i}", names=column_names, delim_whitespace=True, comment="#")
            walker_df['origin'] = labels[i]
    
            # Add frame number
            walker_df['frame'] = walker_df.index

            colvar_list.append(walker_df)

        # Add all walkers to the main dataframe and sort by walker and time
        # This corresponds with `sort -gs walker*/COLVAR.* > walker0/COLVAR` in bash.
        # df = pd.concat(colvar_list).sort_values(by=['time', 'walker'])
        
        # Unsorted to match concatenated trajectory
        df = pd.concat(colvar_list)
        print(f"done") if verbose else 0

    else:
        raise ValueError(f"Invalid number of walkers: ({n_walkers})")

    return df

def read_state(filename, verbose=False):
    """
    Read states file and modify data to fit into two dataframes.

    :param filename: path of STATE file. 
    :param verbose: Print what's happening. (Default value = False)

    :returns: states_data: dataframe containing the STATE data.
    :returns: states_info: containing the extra information, such as zed value/biasfactor etc.
    """

    print(f"\t-> {filename}...", end="") if verbose else 0
    df = pd.read_csv(filename, delim_whitespace=True, low_memory=False)

    # Extracting all states from the dataframe and set the time as the index value
    states_data = df[~df.iloc[:,1].isin(['SET', 'FIELDS'])]
    states_data.columns = states_data.columns[2:].tolist() + 2 * ['na']
    states_data = states_data.dropna(axis=1).astype({"time": int})
    
    # Getting additional information (all variables starting with #! SET)
    states_info = df[df.iloc[:,1] == 'SET'].iloc[:,2:4]
    states_info.columns = ['variable', 'value']
    g = states_info.groupby(['variable']).cumcount()

    # Pivot table. Now the unique variables are the column names
    states_info = (states_info.set_index([g, 'variable'])['value']
        .unstack(fill_value=0)
        .reset_index(drop=True)
        .rename_axis(None, axis=1))

    # Add a column for the time values.
    states_info['time'] = states_data['time'].unique()
    print(f"done") if verbose else 0

    return states_data, states_info

def read_kernels(filename, verbose=False):
    """Read KERNELS file into dataframe.

    :param filename: path of KERNELS file. 
    :param verbose: Print what's happening. (Default value = False)

    :returns: dataframe containing the KERNELS data.

    """

    # Get the names of the columns
    with open(filename, "r") as file:
        first_line = file.readline()            
    column_names = first_line.strip().split(" ")[2:]

    print(f"\t-> {filename}...", end="") if verbose else 0
    df = pd.read_csv(filename, names=column_names, delim_whitespace=True, comment="#")
    print(f"done") if verbose else 0

    return df


def read_dssp(universes, labels=None, numerical=False, verbose=False):
    """Read DSSP file into dataframe.

    :param filename: system.
    :param numerical: Convert the structural components to a numerical value. (Default value = False)
    :param verbose: Print what's happening. (Default value = False)

    :returns: dataframe containing the DSSP data.

    """

    # Preprocess universes and labels.
    # Turn into lists and make the labels fit the universes.
    universes, labels = tools.prepare_ul(universes, labels)

    filenames = [f"{universe.root}/dssp.dat" for universe in universes]

    for index, file in enumerate(filenames):
        
        dssp = []
    
        print(f"\t-> {file}...", end="") if verbose else 0
    
        try:
            with open(file, "r") as rf:
                next(rf)
                for line in rf:
                    dssp.append([*line])
        except FileNotFoundError:
            print(f"ERROR: {file} not found. Create it using: gmx do_dssp -f trajectory.xtc -s topology.tpr -tu ns -ssdump dssp.dat -sc scount.xvg")
            sys.exit(1)

        dssp = np.delete(np.array(dssp), -1, 1)

        if numerical:
            dssp = tools.convert_dssp_to_index(dssp)

        df = pd.DataFrame(dssp)
        df.columns += 1
        df = df.add_prefix('res_')
        df = df.reset_index().rename(columns={'index':'time'})
        
        # Add origin information
        df['origin'] = labels[index]

    print(f"done") if verbose else 0

    return df
