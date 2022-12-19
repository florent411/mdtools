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
parser = argparse.ArgumentParser(description='Split the STATES file and remove every nth frame.')
# files
parser.add_argument('--states',
                    '-f',
                    dest='states_file',
                    type=argparse.FileType('r'),
                    default='STATE',
                    help='the state file name, with the compressed kernels')
parser.add_argument('--out_name',
                    '-o',
                    dest='output_name',
                    type=str,
                    default='STATES_SCRUBBED',
                    help='Name of the output file(s).')
parser.add_argument('--out_dir',
                    '-d',
                    dest='output_dir',
                    type=str,
                    default='separate_states',
                    help='Name of the output directory (only if split is set to True).')

# Compulsory
parser.add_argument('--keep_max',
                    '-m',
                    dest='keep_max',
                    type=str,
                    default='all',
                    help='Keep a maximum of m state. (integer or \'all\')')
parser.add_argument('--stride',
                    '-s',
                    dest='stride',
                    type=int,
                    default=None,
                    help='Keep only every nth state.')

# Other
parser.add_argument('--split',
                    action='store_true',
                    default=False,
                    help='Save each state in a separate file.')
parser.add_argument('--save_last',
                    action='store_true',
                    default=False,
                    help='Save last state.')

# Parsed variables
args = parser.parse_args()
states_file = args.states_file
output_name = args.output_name
output_dir = args.output_dir
keep_max = args.keep_max
stride = args.stride
save_last = args.save_last
split = args.split

# Safely check if the output directory already exists.
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Do splitting.
reader = states_file.read()
read_states = reader.split("#! FIELDS")
# Check how to stride the states.
if stride == None:
    if keep_max == 'all' or len(read_states) <= int(keep_max):
        stride = 1
    else:
        stride = math.ceil(len(read_states) / float(keep_max))
        print(f"Number of states ({len(read_states)}) is more than maximum states ({keep_max}).\nSetting stride to {stride}")
else:
    pass

if split:    
    # Do the stripping
    out_n = 1
    for i, part in enumerate(read_states):
        if i == 0:
            pass # first one is always empty
        elif ((i - 1) % stride == 0):
            with open(f"./{output_dir}/{output_name}.{out_n}", mode="w") as newfile:
                newfile.write("#! FIELDS" + part)
                out_n += 1

    # Summarize what is done
    print(f"Splitted {states_file.name} into {out_n - 1} separate state files.\nOutputdir: ./{output_dir}")

else:
    # Do the stripping
    out_n = 1
    
    with open(f"./{output_name}", mode="w") as newfile:
        for i, part in enumerate(read_states):
            if i == 0:
                pass # first one is always empty
            elif ((i - 1) % stride == 0):
                newfile.write("#! FIELDS" + part)
                out_n += 1
    
    # Summarize what is done
    print(f"Splitted {states_file.name} into {out_n - 1} state files.\nFile: ./{output_name}")

# Save last 
if save_last:
    with open(f"./{output_name}.last.dat", mode="w") as newfile:
            newfile.write("#! FIELDS" + read_states[-1])
