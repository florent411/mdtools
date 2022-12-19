#!/usr/bin/env python 

''' General tools '''

import os
import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt

def setup_format():
    ''' Settings for default format'''
    font_style = {'family' : 'Helvetica', 'weight' : 'normal', 'size' : 14}
    # font_style = {'family' : 'Courier New', 'weight' : 'bold'}
    axes_style = {'labelsize' : 20, 'labelweight' : 'normal', 'linewidth' : 2}
    ticks_style = {'labelsize' : 16, 'minor.visible' : False}
    legend_style = {'fontsize' : 14, 'frameon' : False}

    rc('font', **font_style)
    rc('axes', **axes_style)
    rc('xtick', **ticks_style)
    rc('ytick', **ticks_style)
    rc('legend', **legend_style)

def save_img(filename):
    ''' Save image if needed ''' 

    # Create img folder if it does not already exists.
    os.makedirs("./out", exist_ok=True) 
    
    plt.savefig(f"./img/{filename}", bbox_inches="tight")

    return 0 

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
