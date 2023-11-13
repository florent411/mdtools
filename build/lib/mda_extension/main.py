#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
from dotmap import DotMap
from datetime import date

# MD Analysis
import MDAnalysis as mda
from MDAnalysis import transformations
from MDAnalysis.analysis import align, rms

# Guard module
import checkarg.list as Guard

# Included submodules from the mdtools package
from backpack.main import Backpack
from opes_postprocessing.utils import calc_fes , calc_conv
from mda_extension.utils import tools, file_io, calc

# Other constants
kb = 1.38064852e-23 # Boltzman's constant in m^2 kg s^-2 K^-1
NA = 6.02214086e23 # Avogadro's constant in mol^-1

class MD(mda.Universe):
    ''' Instance for an (unbiased) MD simulations. '''

    def __init__(self, 
                 root=os.getcwd(),
                 topology='run.pdb',
                 trajectory=None,
                 transform=False,
                 backpack='backpack.pkl',
                 verbose=False):
        ''' Initialize run '''

        # Set own variables
        self.root = root
        self.verbose = verbose
        self.topology = topology

        # Take over variables from the mda.Universe superclass.
        if trajectory:
            if type(trajectory) is not list:
                trajectory = [trajectory]

            self.trajectory_path = [f"{root}/{t}" for t in trajectory]
            super().__init__(f"{root}/{topology}", self.trajectory_path)
        else:
            super().__init__(f"{root}/{topology}")

        print(f"----------[Creating universe for {os.path.relpath(self.root, start='./..')}]----------\n") if self.verbose else 0

        if transform:   
            # Remove PBC, align and center protein
            protein = self.select_atoms('protein')
            not_protein = self.select_atoms('not protein')

            # we define the transformation
            transforms = (transformations.unwrap(protein),
                          transformations.center_in_box(protein, wrap=True),
                          transformations.fit.fit_rot_trans(protein, self),
                          transformations.wrap(not_protein, compound='fragments'))
            self.trajectory.add_transformations(*transforms)

        # Create backpack which will contain all information
        self.backpack = Backpack(location=f"{root}/{backpack}", verbose=self.verbose)

        # Integrate all items from backpack database
        for key in self.backpack.content.keys():
            setattr(self, key, self.backpack.get(key))


class OPES(MD):
    ''' Instance for an OPES simulation. Building on/expanding on the MD class. '''

    def __init__(self,
                 root=os.getcwd(),
                 topology='run.pdb',
                 trajectory=None,
                 walker_prefix='walker',
                 colvar='COLVAR',
                 state='STATE',
                 kernels='KERNELS',
                 verbose=False):

        # Take over variables from the MD superclass.
        super().__init__(root, topology, trajectory, verbose=verbose)

        # Turn into list if needed.
        if trajectory:
            if type(trajectory) is not list:
                trajectory = [trajectory]
            
            # Get the number of walkers based on the number of trajectories and save the paths, names, ids and origins in a list.
            self.n_walkers = len(trajectory)
            self.walker_paths = [f"{os.path.dirname(traj)}" for traj in trajectory]
            self.walker_names = [f"{os.path.basename(os.path.dirname(traj))}" for traj in trajectory]
            self.walker_ids = [*range(self.n_walkers)]
            self.walker_origins = [f"{walker_prefix}{id}" for id in self.walker_ids]

        else:
            self.n_walkers = 1
            self.walker_paths = ["."]
            self.walker_names = ["."]
            self.walker_ids = ["."]
            self.walker_origins = ["."]

        # Check if the lengths are right.
        Guard.is_length_equals(self.walker_paths, self.n_walkers)
            
        # Load OPES specific output files into dataframes
        print("\nReading plumed files:") if self.verbose else 0

        # Load files if they were not in the backpack and add them.
        if not hasattr(self, 'colvar'):
            self.colvar = file_io.read_colvar(self.root, self.walker_paths, colvar, labels=walker_prefix, verbose=self.verbose)
            self.backpack.set('colvar', self.colvar)

        if not (hasattr(self, 'state_data') and hasattr(self, 'state_info')):
            self.state_data, self.state_info = file_io.read_state(f"{self.root}/{self.walker_paths[0]}/{state}", verbose=self.verbose)
            self.backpack.set('state_data', self.state_data)
            self.backpack.set('state_info', self.state_info)

        if not hasattr(self, 'kernels'):
            self.kernels = file_io.read_kernels(f"{self.root}/{self.walker_paths[0]}/{kernels}", verbose=self.verbose)
            self.backpack.set('kernels', self.kernels)

        # If multiple walkers, make a list containing the walkers as a new MD object
        if self.n_walkers > 1:
            print(f"\nReading in walkers...") if self.verbose else 0
            
            # If multiple walkers, create a list of walkers, each accessible by self.walker[walker_id].
            self.walkers = []
            for i, path in enumerate(self.walker_paths):
                print(f"\t-> Walker {i}...", end="") if self.verbose else 0
                self.walkers.append(Walker(root=f"{self.root}/{path}",
                                           topology=f"../run_prot.pdb",
                                           trajectory=os.path.basename(trajectory[i]),
                                           prefix=walker_prefix,
                                           id=i,
                                           colvar=f"COLVAR.{i}",
                                           verbose=False))
                print(f"done") if self.verbose else 0

    def calc(self,
             order_param,
             *args,
             key_string=None,
             **kwargs):
        ''' Calculate anything for OPES class '''

        # Available function in the calc module
        dispatcher = {'weights' : calc_fes.weights,
                      'fes_state' : calc_fes.from_state,
                      'fes_colvar' : calc_fes.from_colvar,
                      'fes_weights' : calc_fes.from_weights,
                      'rg': calc.rg,
                      'rmsd': calc.rmsd,
                      'rmsf': calc.rmsf,
                      'mindist': calc.mindist,
                      'conv_params' : calc_conv.conv_params,
        }
                    #   'fes_kernels' : calc_fes.from_kernels,
                    #   'kldiv' : calc_conv.kldiv,
                    #   'jsdiv' : calc_conv.jsdiv,
                    #   'dalonso' : calc_conv.dalonso,
        
        # Run the requested function, with the given arguments
        value = dispatcher[order_param](*args, **kwargs)
        
        # Add to  as variable within the function and (if needed) save in the backpack
        if key_string == 'timeseries':
            if not 'timeseries' in self.backpack.content:
                self.backpack.set('timeseries', value)
                setattr(self, 'timeseries', value)
            else:
                merged_df = pd.merge(self.backpack.get('timeseries'),
                                     value,
                                     on=['time', 'origin'],
                                     suffixes=['', f'_{date.today().strftime("%Y%m%d")}'], how='outer')
                                     
                self.backpack.set('timeseries', merged_df)
                setattr(self, 'timeseries', merged_df)
        elif len(key_string.split('.')) > 1:
            p_type = key_string.split('.')[0]
            if not p_type in self.backpack.content:
                dm = DotMap()
            else:
                dm = self.backpack.get(p_type)

            # Add the value to the dotmap
            keys = key_string.split('.')
            
            # Get to the right dotmap depth
            current_level = dm
            for key in keys[1:-1]:
                current_level = current_level[key]

            current_level[keys[-1]] = value

            self.backpack.set(p_type, dm)
            setattr(self, p_type, dm)
        else:
            self.backpack.set(key_string, value)
            setattr(self, key_string, value)

    def read(self,
             key,
             *args,
             alias=None,
             save=True,
             **kwargs):
        ''' Calculate anything for OPES class '''

        # Available function in the calc module
        dispatcher = {'dssp' : file_io.read_dssp,
                      'xvg' : file_io.read_xvg,
        }
                    #   'fes_kernels' : calc_fes.from_kernels,
                    #   'kldiv' : calc_conv.kldiv,
                    #   'jsdiv' : calc_conv.jsdiv,
                    #   'dalonso' : calc_conv.dalonso,
        
        # Run the requested function, with the given arguments
        value = dispatcher[key](*args, **kwargs)
        
        # Set as variable within the function and (if needed) save in the backpack
        if alias == None: 
            setattr(self, key, value)
            self.backpack.set(key, value) if save else 0
        else:
            setattr(self, alias, value)
            self.backpack.set(alias, value) if save else 0

class Walker(MD):
    ''' Instance for an OPES walker simulation. Building on/expanding on the MD class. '''

    def __init__(self,
                 root,
                 topology='run.pdb',
                 trajectory=None,
                 id=0,
                 prefix=None,
                 colvar="COLVAR.0",
                 verbose=False):
        # Take over variables from the MD superclass.
        super().__init__(root, topology, trajectory, verbose=verbose)

        # Initialize own variables
        self.id = id

        if prefix is None:
            self.origin = self.id
        else:
            self.origin = prefix + str(self.id)

        self.walker_paths = ["."]
        self.walker_names = ["."]


        # Load files
        self.colvar = file_io.read_colvar(root, self.walker_paths, colvar, labels=self.origin, verbose=self.verbose)
