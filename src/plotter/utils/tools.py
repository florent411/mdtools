#!/usr/bin/env python 

''' General tools for the plotter module'''

import os
import sys
import numpy as np
import seaborn as sns
from matplotlib import rc
import matplotlib.pyplot as plt

def moving_average(a, n=3):
    """
    Calculate moving average (https://en.wikipedia.org/wiki/Moving_average).
        
    :param palette: Can be the name of an existing palette (https://www.geeksforgeeks.org/seaborn-color-palette/)
                    or a list of colors (python list of comma seperated string), from which a palette is generated.
    :param n_hues: The number of colors the palette has to exist of.
    
    :return: palette
    """
    
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

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

def setup_palette(palette, n_hues=10):
    """
    Create a seaborn palette based on a palette name or a list of colors.
        
    :param palette: Can be the name of an existing palette (https://www.geeksforgeeks.org/seaborn-color-palette/)
                    or a list of colors (python list of comma seperated string), from which a palette is generated.
    :param n_hues: The number of colors the palette has to exist of.
    
    :return: palette
    """

    # Setup colorpalette if a list of colors is inputted
    if type(palette) is list:
        # If only 1 color, double it to create a single color spectrum/palette in sns.blend_palette (which takes a minimum of 2 colors)
        if len(palette) == 1:
            palette += palette

        # Create a palette based on the inputted colors
        palette = sns.blend_palette(palette, n_hues)

        return palette

    elif type(palette) is str:
        # Test if palette exists
        try:
            sns.color_palette(palette, as_cmap=True)
            return palette
    
        except:
            palette = list(map(str, palette.split(','))) # list of strings

            # If only 1 color, double it to create a single color spectrum/palette in sns.blend_palette (which takes a minimum of 2 colors)
            if len(palette) == 1:
                palette += palette

            # Create a palette based on the inputted colors
            palette = sns.blend_palette(palette, n_hues)

            return palette

    else:
            raise TypeError(f"ERROR: {palette} is not a valid palette input.")

def setup_variables(df, variables=None):
    """
    Define variables, either from user input, or, if variables are not defined, take them from the input dataframe '''
        
    :param df: df from which to extract what variables there are.
    :param variables: User input. List of chosen variables. This function then checks if they actually exist in the df.
    
    :return: list of variables
    """

    # First find the variables in the input dataframe
    column_names = df.columns.to_list()
    variables_from_df = [x for x in column_names if x not in ['time', 'fes', 'origin']]
    
    # If no input for variables (variables == None), just use variables as determined above, otherwise use the user input (making sure its a subset of the cvs in the input)
    if variables == None:
        variables = variables_from_df

    # Otherwise, to keep the right order (as in the state/colvar file), I find the variables in the state file and compare them with the user input.
    else:
        # Turn into a list
        if isinstance(variables, list):
            pass 
        elif isinstance(variables, tuple):
            variables = list(variables)
        elif isinstance(variables, str):
            variables = list(map(str, variables.split(','))) # list of strings
        else:
            sys.exit(f"ERROR: inputtype ({type(variables)}) not (yet) supported. Use a list, tuple or comma separated string")
        
        # Check if user input is a subset of the variables found in the state file
        if not all(elem in variables_from_df for elem in variables):
            sys.exit(f"ERROR: CV input ({variables}) should be subset of variables in the state file ({variables_from_df}).")

    if len(variables) > 2:
        sys.exit(f"ERROR: More than 2 dimensions is not yet supported. Please select a subset of {variables}.")

    return variables


def save_img(filename):
    """
    Save the plot as an image in the ./img/ directory.
        
    :param filename: Name of the file to save to.

    :return: 0
    """

    # Create img folder if it does not already exists.
    os.makedirs("./img", exist_ok=True) 
    
    plt.savefig(f"./img/{filename}", bbox_inches="tight")

    return 0 

