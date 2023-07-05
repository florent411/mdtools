#!/usr/bin/env python3

'''
theoretical_helicity.py

Calculate theoretical helicity based on:
A helix propensity scale based on experimental studies of peptides and proteins
CN Pace, JM Scholtz - Biophysical journal, 1998 - Elsevier
'''

# Import modules from modules folder
import os
import sys

import numpy as np
import pandas as pd #much faster reading from file
import argparse

import seaborn as sns
from matplotlib import pyplot as plt

#
try:
    from plotter.utils import tools
    standard_layout = True
except:
    print("\n\n!Warning: plotter.tools not found, using standard seaborn layout.")
    standard_layout = False

### Parser stuff ###
parser = argparse.ArgumentParser(description='Calculate and plot theoretical helicity of a peptide based on CN Pace & JM Scholts paper.')

# files
parser.add_argument('--sequence',
                    '-s',
                    dest='input_sequence',
                    type=str,
                    default=None,
                    help='Sequence of peptide. (default: %(default)s)')

parser.add_argument('--out',
                    '-o',
                    dest='output_file',
                    default='theoretical_helicity.dat',
                    type=str,
                    help='Name of the output file. (default: %(default)s)')

parser.add_argument('--plot_out',
                    '-p',
                    dest='plot_output_file',
                    default='theoretical_helicity.pdf',
                    type=str,
                    help='Name of the plot output file. (default: %(default)s)')

# Parsed variables
args = parser.parse_args()

sequence = args.input_sequence
output_file = args.output_file
plot_output_file = args.plot_output_file 

# Constants 
hp_dict = {
    "A" : 0.00,
    "L" : 0.21,
    "R" : 0.21,
    "M" : 0.24,
    "K" : 0.26,
    "Q" : 0.39,
    "E" : 0.40,
    "I" : 0.41,
    "W" : 0.49,
    "S" : 0.50,
    "Y" : 0.53,
    "F" : 0.54,
    "H" : 0.61,
    "V" : 0.61,
    "N" : 0.65,
    "T" : 0.66,
    "C" : 0.68,
    "D" : 0.69,
    "G" : 1.00,
    "P" : 1.00
}

# If no sequence is given, ask to user.
if sequence == None:
    sequence = input("Please insert one-letter code of the peptide:\n")

# Check length and if everything is alpha.
if len(sequence) > 10000:
    print("ERROR: Sequence exceeds maximum of 10000 characters")
    exit(1)
elif not sequence.isalpha():
    print("ERROR: Sequence appears to contain non-letter values")
    exit(1)
else:    
    # Make uppercase
    sequence = sequence.upper()
    
    valid_amino_acids = list(hp_dict.keys())

    # Turning sequence into dataframe.
    hps = [1.0 - hp_dict[residue] if residue in valid_amino_acids else np.nan for residue in list(sequence)]

    df = pd.DataFrame(
            {'residue': list(sequence),
            'helical propensity': hps,
            })
    df.index += 1

    # Print warnings for all invalid amino acids
    invalid_amino_acids_in_sequence = list(df[df['helical propensity'].isnull()]['residue'].unique())

    if len(invalid_amino_acids_in_sequence) == 1:
        print(f"!WARNING: {invalid_amino_acids_in_sequence[0]} is not a valid amino acid.\n")
    elif len(invalid_amino_acids_in_sequence) > 1:
        print(f"!WARNING: {' ,'.join(invalid_amino_acids_in_sequence[:-1])} and {invalid_amino_acids_in_sequence[-1]} are not a valid amino acid.\n")
    else:
        pass

    print(f"Now writing {output_file}... ", end="")
    
    df.to_csv(f"{output_file}",
              sep='\t',
              float_format="%.2f",
              header=True,
              index=True,
              index_label="id")
           
    print("done")

    print(f"Now plotting {plot_output_file}... ", end="")

    if standard_layout:
        # Setup default matplotlib values for layout
        tools.setup_format()


    # Plot image
    g = sns.pointplot(data=df,
                      x=df.index,
                      y="helical propensity")
    
    g.set_xticklabels(df['residue'].to_list())
    g.set(ylim=(-0.1, 1.1))

    plt.savefig(plot_output_file, transparent=True)

    print("done\n")
