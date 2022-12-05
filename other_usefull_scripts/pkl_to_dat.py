#! /usr/bin/env python3

import sys
import numpy as np
import pandas as pd #much faster reading from file
import pickle
import argparse

### Parser stuff ###
parser = argparse.ArgumentParser(description='Fetch a variable from a pickled dictionary.')

# files
parser.add_argument('--in',
                    '-f',
                    dest='input_file',
                    type=str,
                    default='backpack.pkl',
                    help='The file name of the pickled database. (default: %(default)s)')
parser.add_argument('--out',
                    '-o',
                    dest='output_file',
                    type=str,
                    default=None,
                    help='Name of the output file. (default: [variablename].dat])')
parser.add_argument('--key',
                    '-k',
                    dest='key',
                    type=str,
                    help='Key of the variable.')

# Parsed variables
args = parser.parse_args()

input_file = args.input_file
output_file = args.output_file
key = args.key

# Check keep_max value
if output_file is None:
    output_file = f"{key}=".split('=')[0] + ".dat"

# Load database from file.
with open(input_file, 'rb') as handle:
    db = pickle.load(handle)

# Fetch 
try:
    var = db[key]
except Exception as e:
    print(f"ERROR: Could not find {key} in database {input_file}.\n\n{e}")
    sys.exit(1)


if isinstance(var, pd.DataFrame):
    # If dataframe use pd.to_csv.
    var.to_csv(output_file, sep='\t', index=False)
elif isinstance(var, (np.ndarray, np.generic) ):
    # If numpy array use np.save_txt.
    np.savetxt(output_file, var, delimiter='\t')
else:
    # Otherwise just dump into file
    with  open(output_file, "w") as f:
        f.print(var)

print(f"Done: Got {key} from {input_file} and printed it in {output_file}.")