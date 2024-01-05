# mda_extension Package Documentation

The `mda_extension` package is built upon the foundation of the MDAnalysis package, a powerful Python library for analyzing molecular dynamics trajectories. It leverages the `MDAnalysis.Universe` class as a base, extending its capabilities with additional features tailored for unbiased MD and On-the-Fly Probability Enhanced Sampling (OPES) simulations.

The `mda_extension` package provides a set of classes and utilities for working with
- molecular dynamics (MD) simulations, specifically designed for unbiased MD
- On-the-Fly Probability Enhanced Sampling (OPES) simulations


## Classes

### `MD` Class Overview

The `MD` class is an extension of the MDAnalysis `Universe` class, providing additional functionality for MD simulations. It includes features such as trajectory transformation, backpack creation, and integration of information from the backpack database.

#### Initialization Parameters

- `root` (str): The root directory for the simulation (default: current working directory).
- `topology` (str): The topology file (default: 'run.pdb').
- `trajectory` (list or str): List of trajectory file paths or a single trajectory file path (default: None).
- `transform` (bool): Boolean flag indicating whether to perform trajectory transformation (default: False).
- `backpack` (str): The filename for the backpack file (default: 'backpack.pkl'). See backpack README for more information.
- `verbose` (bool): Boolean flag for verbose output (default: False).

#### Methods
Both classes contain the following methods.
- `calc(order_param, *args, key_string=None, **kwargs)`: Calculate specific properties based on the provided order parameter. The order_params that can be calculated for the system are now:
    - rmsd
    - rg
    - rmsf
    - mindist
    - dssp

- `read(key, *args, alias=None, save=True, **kwargs)`: Read specific file types (e.g. .xvg) and store the results in the backpack. This can be usefull when using third party applications to calculate order params.

#### Usage
See demo_md.ipynb.

### `OPES` Class Overview

The `OPES` class extends the `MD` class, specializing in OPES simulations. It includes functionality for reading OPES-specific output files (e.g., COLVAR, STATE, KERNELS) and handling multiple walkers.

#### Initialization Parameters

- Parameters inherited from the `MD` class.
- `walker_prefix` (str): Prefix for walker-specific information (default: 'walker').
- `colvar` (str): File name for the collective variable file (default: 'COLVAR').
- `state` (str): File name for the state file (default: 'STATE').
- `kernels` (str): File name for the kernels file (default: 'KERNELS').

#### Methods
Both classes contain the following methods.
- `calc(order_param, *args, key_string=None, **kwargs)`: Calculate specific properties based on the provided order parameter. The order_params that can be calculated for the system are now:
    - rmsd
    - rg
    - rmsf
    - mindist
    - dssp

    In case of the OPES modules, it is also possible to calculate:
    - weights; weights for each frame of the simulation
    - from_state; FES based on the STATE file
    - from_colvar; FES based on the COLVAR file
    - from_weights; FES based on reweighting

- `read(key, *args, alias=None, save=True, **kwargs)`: Read specific file types (e.g. .xvg) and store the results in the backpack. This can be usefull when using third party applications to calculate order params.

#### Usage
See demo_opes.ipynb.


#### `Walker` Class Overview

The `Walker` class represents an individual walker in an OPES simulation. It inherits functionality from the `MD` class and includes additional features for handling walker-specific information.

#### Initialization Parameters

- Parameters inherited from the `MD` class.
- `id` (int): Walker ID.
- `prefix` (str): Prefix for the walker's origin (default: None).
- `colvar` (str): File name for the collective variable file specific to the walker (default: 'COLVAR.0').





