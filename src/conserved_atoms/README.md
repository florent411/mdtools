# Conserved Atoms Analysis

## Overview
The MDTools - Conserved Atoms script is a Python tool designed for analyzing Molecular Dynamics (MD) simulations to identify conserved atoms within a specified binding pocket or area. It leverages MDAnalysis, a powerful library for MD trajectory analysis.

## Usage

### Execution

The code can be loaded as a package:

```python
import conserved_atoms.main as ca
```

or executed from the command lines as follows:

```bash
python conserved_atoms.py -s <structure_file> \ 
                          -f <trajectory_file> \ 
                          -o <output_directory> \ 
                          -p <pocket_definition> \ 
                          -a <align_on> \ 
                          -t <target_atom_name> \ 
                          -e <element> \ 
                          -w <output_trajectory_file> \ 
                          -v
```

**Flags:**
- `-s`, `--structure`: Structure file (default: `run.pdb`).
- `-f`, `--trajectory`: Trajectory file (default: `run.xtc`).
- `-o`, `--output_dir`: Output directory (default: `./conserved_atoms`).
- `-p`, `--pocket_definition`: Definition of the pocket to analyze (default: `point 0.0 0.0 0.0 15`).
- `-a`, `--align_on`: Align trajectory on (default: `CA`).
- `-t`, `--target`: Atom name of target atoms (default: `OH2`).
- `-e`, `--element`: Element (default: `O`).
- `-w`, `--write`: Write trajectory Element (default: `./conserved_atoms/conserved_atoms.{gro,xtc}`).
- `-v`, `--verbose`: Enable verbose mode.

**Example:**
```bash
python conserved_atoms.py -s my_structure.pdb\
                          -f simulation_data.xtc\
                          -o results\
                          -p "sphere 1.0 2.0 3.0 10"\
                          -a C_alpha\
                          -t CA\
                          -e C\
                          -w ./output/conserved_atoms.gro\
                          -v
```

This command will run the `conserved_atoms.py` script with the specified parameters. Make sure to replace the placeholder values (`<structure_file>`, `<trajectory_file>`, etc.) with your actual file names or values.

### Functions

#### `calc_density`

- **Description:** Calculate the density of a specified atom type throughout a simulation. This is usefull to get an idea of where certain atomtypes reside throughout a simulation. 
- **Parameters:**
  - `u`: MDAnalysis universe, including a trajectory.
  - `pocket_definition`: Definition of the binding pocket or area for density calculation.
  - `target`: Name of the atom to calculate the density for in the trajectory.
  - `align_on`: Atom used for aligning the trajectory.
  - `unwrap`: Boolean indicating whether to unwrap the protein.
  - `write_traj`: Boolean indicating whether to write the trajectory aligned to the defined pocket.
  - `verbose`: Boolean allowing printing.

#### `cluster`

- **Description:** Perform clustering on the calculated atom density. This way one can find all the hotspots for a certain atom type. Think sodium binding spots of conserverd water molecules.
- **Parameters:**
  - `df`: DataFrame containing density information.
  - `n_frames`: Number of frames in the simulation.
  - `clustering_algorithm`: Clustering algorithm to use ('dbscan' or 'meanshift').
  - `epsilon`: Resolution for DBSCAN clustering.
  - `density_cutoff`: Cutoff for density (only for meanshift clustering).
  - `atomic_radius`: Atomic radius used for estimating density cutoff.
  - `element`: Element for which atomic radius is looked up.
  - `outlier_threshold`: Threshold for determining outliers during cleanup.
  - `verbose`: Boolean allowing printing.

#### `voxelize`

- **Description:** Sometimes a pointcloud, containing all the atoms of interest, can be very large. This function can be used to reduce the number of points still showing the same area. This is done by voxelization, i.e. by grouping nearby points into a single point.
- **Parameters:**
  - `df`: DataFrame containing coordinates.
  - `resolution`: Size of the voxel grid.
  - `compact_by`: Method for compacting data in a voxel ('mean', 'max', or 'min').
  - `skip_outliers`: Boolean indicating whether to skip outliers.
  - `verbose`: Boolean allowing printing.

#### `write_pdb`

- **Description:** Write a PDB file with the found atom clusters. This can then be used for visualization purposes.
- **Parameters:**
  - `df`: DataFrame containing coordinates and additional information.
  - `output`: Output PDB file name.
  - `skip_outliers`: Boolean indicating whether to skip outliers.
  - `verbose`: Boolean allowing printing.

#### `write_dat`

- **Description:** Write a data file with information about the found atom clusters.
- **Parameters:**
  - `df`: DataFrame containing coordinates and additional information.
  - `output`: Output data file name.
  - `verbose`: Boolean allowing printing.

#### `create_traj`

- **Description:** Create a trajectory containing the conserved atoms merged with the original trajectory. This can be used to visualize the trajectory using eg ngview.

- **Parameters:**
  - `u`: MDAnalysis universe used for the conserved atoms algorithm.
  - `df`: DataFrame containing cluster information.
  - `write_struct_to`: File name to write the structure.
  - `write_traj_to`: File name to write the trajectory.
  - `element`: Element for the conserved atoms.
  - `name`: Atom name for the conserved atoms.
  - `skip_outliers`: Boolean indicating whether to skip outliers.
  - `verbose`: Boolean allowing printing.

## Example usage
See demo.ipynb


## Acknowledgments

This script relies on the MDAnalysis library for handling molecular dynamics data. Refer to the MDAnalysis documentation for further details.

---

Feel free to customize the documentation further based on your specific preferences and additional information about the script.