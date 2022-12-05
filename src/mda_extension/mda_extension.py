#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd

# MD Analysis
import MDAnalysis as mda
from MDAnalysis import transformations
from MDAnalysis.analysis import align, rms

# Guard module
import checkarg.list as Guard

# Included submodules
from backpack.backpack import Backpack

import mda_extension.modules.tools as tools
import mda_extension.modules.file_io as fio
import mda_extension.modules.calc as calc

# Other constants
kb = 1.38064852e-23 # Boltzman's constant in m^2 kg s^-2 K^-1
NA = 6.02214086e23 # Avogadro's constant in mol^-1

class MD(mda.Universe):
    ''' Instance for an (unbiased) MD simulations. '''

    def __init__(self, root=os.getcwd(),
                 topology='run.pdb',
                 trajectory=None,
                 align_on=None,
                 transform=False,
                 backpack='backpack.pkl',
                 verbose=False):
        ''' Initialize run '''

        # Set own variables
        self.root = root
        self.verbose = verbose
        self.align_on = align_on

        # Take over variables from the mda.Universe superclass.
        if trajectory:
            if type(trajectory) is not list:
                trajectory = [trajectory]

            trajectory = [f"{root}/{t}" for t in trajectory]
            super().__init__(f"{root}/{topology}", trajectory)
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
                        transformations.wrap(not_protein))
            self.trajectory.add_transformations(*transforms)

        # Create backpack which will contain all information
        self.backpack = Backpack(location=f"{root}/{backpack}", verbose=self.verbose)

        # Integrate all items from backpack database (db)
        for key in self.backpack.db.keys():
            setattr(self, key, self.backpack.get(key))


    def calc(self,
             key,
             *args,
             save=True,
             **kwargs):
        ''' Calculate anything for MD class '''

        # Available function in the calc module
        dispatcher = {'rmsd': calc.rmsd,
                      'rg': calc.rg,
                      'rmsf': calc.rmsf}
                      # TODO  'mindist': calc.mindist}
        
        # Run the requested function, with the given arguments
        value = dispatcher[key](*args, **kwargs)
        
        # Set as variable within the function and (if needed) save in the backpack
        setattr(self, key, value)
        self.backpack.set(key, value) if save else 0


class OPES(MD):
    ''' Instance for an OPES simulation. Building on/expanding on the MD class. '''

    def __init__(self,
                 root=os.getcwd(),
                 topology='run.pdb',
                 trajectory=None,
                 walker_labels=None,
                 align_on=None,
                 colvar='COLVAR',
                 states='STATES',
                 kernels='KERNELS',
                 fes=None,
                 verbose=False):

        # Take over variables from the MD superclass.
        super().__init__(root, topology, trajectory, align_on, verbose=verbose)

        # Turn into list if needed.
        if trajectory:
            if type(trajectory) is not list:
                trajectory = [trajectory]
            
            # Get the number of walkers based on the number of trajectories and save the paths in a list.
            self.n_walkers = len(trajectory)
            self.walker_paths = [f"{os.path.dirname(traj)}" for traj in trajectory]
            self.walker_names = [f"{os.path.basename(os.path.dirname(traj))}" for traj in trajectory]

        else:
            self.n_walkers = 1
            self.walker_paths = ["."]
            self.walker_names = ["."]

        # Check if the lengths are right.
        Guard.is_length_equals(self.walker_paths, self.n_walkers)
            
        # Load OPES specific output files into dataframes
        print("\nReading plumed files:") if self.verbose else 0
        self.colvar = fio.read_colvar(self.root, self.walker_paths, colvar, labels=walker_labels, verbose=self.verbose)
        self.states, self.states_info = fio.read_states(f"{self.root}/{self.walker_paths[0]}/{states}", verbose=self.verbose)
        self.kernels = fio.read_kernels(f"{self.root}/{self.walker_paths[0]}/{kernels}", verbose=self.verbose)

        # Read Free Energy information
        if fes:
            print(f"\nReading free energy:") if self.verbose else 0
            self.fes = fio.read_fes(fes, verbose=self.verbose)
        else:
            print(f"\nNo free energy will be read.") if self.verbose else 0

        # If multiple walkers, make a list containing the walkers as a new MD object
        if self.n_walkers > 1:
            print(f"\nReading in walkers...") if self.verbose else 0
            
            # If multiple walkers, create a list of walkers, each accessible by self.walker[walker_id].
            self.walkers = []
            for i, path in enumerate(self.walker_paths):
                print(f"\t-> Walker {i}...", end="") if self.verbose else 0
                self.walkers.append(Walker(root=f"{self.root}/{path}",
                                           topology=f"../run_prot.tpr",
                                           trajectory=f"run.xtc",
                                           align_on=self.align_on,
                                           labels=walker_labels,
                                           id=i,
                                           colvar=f"COLVAR.{i}",
                                           verbose=False))
                print(f"done") if self.verbose else 0

    def calc(self,
             key,
             *args,
             save=True,
             **kwargs):
        ''' Calculate anything for OPES class '''

        # Available function in the calc module
        dispatcher = {'rmsd': calc.rmsd,
                      'rg': calc.rg,
                      'rmsf': calc.rmsf,
                      'weights' : calc.weights}
                    #   'mindist': calc.mindist}
        
        # Run the requested function, with the given arguments
        value = dispatcher[key](*args, **kwargs)
        
        # Set as variable within the function and (if needed) save in the backpack
        setattr(self, key, value)
        self.backpack.set(key, value) if save else 0



class Walker(MD):
    ''' Instance for an OPES walker simulation. Building on/expanding on the MD class. '''

    def __init__(self,
                 root,
                 topology='run.pdb',
                 trajectory=None,
                 align_on=None,
                 id=0,
                 labels=None,
                 colvar="COLVAR.0",
                 verbose=False):
        # Take over variables from the MD superclass.
        super().__init__(root, topology, trajectory, align_on, verbose=verbose)

        # Initialize own variables
        self.id = id

        if labels is None:
            self.labels = [self.id]
        elif type(labels) is not list:
            self.labels = [labels + str(self.id)]

        self.walker_paths = ["."]
        self.walker_names = ["."]

        # Load files
        self.colvar = fio.read_colvar(root, self.walker_paths, colvar, labels=self.labels, verbose=self.verbose)
