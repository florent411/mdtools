#! /usr/bin/env python3

import os
import re
import sys
import math
import argparse
from pprint import pprint
import numpy as np
import pandas as pd #much faster reading from file
from pathlib import Path

### Parser stuff ###
parser = argparse.ArgumentParser(description='Split the FES file and optionally remove every nth frame.')
# files
parser.add_argument('--fes',
                    '-f',
                    dest='fes_file',
                    type=argparse.FileType('r'),
                    default='FES',
                    help='The fes file name. (default: %(default)s)')
parser.add_argument('--out_prefix',
                    '-o',
                    dest='output_prefix',
                    type=str,
                    default='fes',
                    help='Prefix for all output files. (default: %(default)s)')
parser.add_argument('--out_dir',
                    '-d',
                    dest='output_dir',
                    type=str,
                    default='fes_out',
                    help='Name of the output directory. (default: %(default)s)')
parser.add_argument('--keep_max',
                    '-m',
                    dest='keep_max',
                    type=str,
                    default='all',
                    help='Process all frames (all), only last frame (last) or n frames (int). (default: %(default)s)')

# Parsed variables
args = parser.parse_args()
fes_file = args.fes_file
output_prefix = args.output_prefix
output_dir = args.output_dir
keep_max = args.keep_max

# Check keep_max value
if keep_max.isdigit() and int(keep_max) > 0:
    keep_max = int(keep_max)
elif keep_max not in ['all', 'last']:
    sys.exit(f"ERROR: Illegal value found for keep_max ({keep_max}). Please choose 'all', 'last' or a positive integer\nNow exiting")

# Safely check if the output directory already exists.
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Do splitting.
fes_from_file = pd.read_csv(fes_file, delim_whitespace=True)

times_list = fes_from_file['time'].unique()

# See what colvar to analyse
if keep_max == 'last':
    # Process only last frame
    times_list = [times_list[-1]]
elif keep_max == 'all' or len(times_list) <= int(keep_max):
    # If you have more frames than want to process all, or if the max 
    pass
elif len(times_list) >= int(keep_max):
    # Striding the list of times to analyse.
    last = times_list[-1]
    stride = int(np.ceil(len(times_list) / float(keep_max)))
    print(f"Number of time points ({len(times_list)}) is more than given keep_max ({keep_max}).\nSetting stride to {stride}", end="")
    times_list = times_list[::stride]
    print(f" --> keeping {len(times_list)} time points")
    # Note: I've decided to always add the last frame which is the "final" state, this might give a small discontinuity in the timesteps between the last two frames.
    if times_list[-1] != last:
        times_list = np.concatenate((times_list, [last]), axis=None)
        print(f"Note: last frame was added to the end. There are now {len(times_list)} time points. This might give a small discontinuity in the timesteps between the last two frames.")
else:
    sys.exit(f"ERROR: Something went wrong when striding.")


for i, time in enumerate(times_list):
    fes = fes_from_file[fes_from_file['time'] == time]

    fes.to_csv(f'{output_dir}/{output_prefix}_{i + 1}.dat', index=False, sep='\t', float_format='% 12.6f')

print(f"Done: outputted {len(times_list)} fes files in ./{output_dir}")