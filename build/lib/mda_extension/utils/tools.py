#!/usr/bin/env python 

''' General tools '''

import numpy as np
# import torch

# def set_device(device, verbose):
#     if device == 'auto':
#         if torch.cuda.is_available():
#             device = 'cuda'
#             print(f"Found compatible CUDA device. Device set to {device} ({torch.cuda.get_device_name(0)})") if verbose else 0
#         elif torch.backends.mps.is_available():
#             device = 'mps'
#             print(f"Found compatible GPU. Device set to {device} (Apple GPU)") if verbose else 0
#         else:
#             print(f"Did not find any compatible GPU. Device set to {device}. (CPU)") if verbose else 0
#             device = 'cpu'
#     elif device in ['cpu', 'mpl', 'cuda']:
#         print(f"Device is set to: {device}")
#     else:
#         raise ValueError(f"Unknown device option: {device}") if verbose else 0
    
#     return device

def moving_average(a, n=3) :
    """

    :param a: Timeseries
    :param n: Calculate moving average over n points. (Default value = 3)

    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def prepare_ul(universes, labels):
    """
    Do preprocessing of the universes and labels.

    :param universes:
    :param labels: 

    """

    # Catch non-list input
    if type(universes) is not list:
        universes = [universes]

    # Catch None values
    if labels == None:
        try: 
            labels = [u.origin for u in universes]
        except:
            labels = [*range(len(universes))]
            
    elif type(labels) is not list:
        labels = [labels]

    # If you have one label for multiple universes, use it as a prefix.
    if len(labels) == 1 and len(universes) != 1:
        labels = [f"{labels[0]}{i}" for i in range(len(universes))]

    return universes, labels