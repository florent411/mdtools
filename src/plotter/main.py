#!/usr/bin/env python 

''' Plot images '''

import sys
import numpy as np
import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.neighbors import KernelDensity

# Included submodules
from plotter.utils import tools

def timeseries(df,
               variables,
               hue='origin',
               density=True,
               stride=10,
               rolling_avg=None,
               label_dict=None,
               palette="flare_r",
               save=True,
               **kwargs):
    """
    timeseries plots cv(s) vs time.
    
    :param df: Input dataframe. Needs at least the columns 'time', '[cv]', 'origin'
    :param variables: Variable(s) to plot, has to correspond to the columns header in the df.
    :param hue: What column to color the different lines on
    :param density: Plot marginal density distribution of the y axis
    :param stride: Stride the number of datapoints
    :param rolling_avg: Plot rolling average over a faded full  plot
    :param label_dict: Linking codes/column names with a more elaborate title to plot on the axes
    :param palette: Colormap
    :param save: Save image in the ./img folder.

    :return: fig and axes
    """

    # Make a copy to make sure you're not editing the original df.
    df = df.copy()

    # Setup default matplotlib values for layout
    tools.setup_format()

    # Setup color palette
    # Check how many different lines/inputs there are.
    n_hues = len(df[hue].unique())
    palette = tools.setup_palette(palette, n_hues)

    # Setup variable list
    variables = tools.setup_variables(df, variables)

    # Stride, otherwise plotting gets very slow. (And reset index, otherwise Searborn complains)
    df = df[df.index % stride == 0].reset_index()

    # Extra settings per option
    alpha = 0.3 if rolling_avg else 0.8
    legend_loc = 1 if density else 0
    h_ratio = [1, 7] if len(variables) > 1 else [1]
    w_ratio = [7, 1]

    # Create one or two subplots
    if density:
        # gridspec_kw makes sure that the ratio between the first and second plot on each line is 6:1
        # The small one is the distribution
        fig, axes = plt.subplots(len(variables), 2, figsize=(8, 4 * len(variables)), gridspec_kw={'height_ratios': h_ratio, 'width_ratios': w_ratio}, squeeze=False)
        fig.subplots_adjust(wspace=0, hspace=0)
    else:
        # Otherwise, just plot in one axis. (Squeeze makes sure you can still axis it with two indices i.e. axes[0][0])
        fig, axes = plt.subplots(1, 1, figsize=(7, 4), squeeze=False)
        fig.subplots_adjust(hspace=0.3)

    # Other dimensions are not yet implemented
    if len(variables) == 1:

        # Plot the timeseries
        sns.lineplot(ax = axes[0][0],
                     data=df,
                     x='time',
                     y=variables[0],
                     hue=hue,
                     palette=palette,
                     linewidth=0.2,
                     alpha=alpha)
                     
        # Add the proper labels for each axes
        try:
            axes[0][0].set(xlabel="Time (ns)", ylabel=label_dict[variables[0]])
        except:
            print("Warning: couldn't find the labels in the label dict.")
            axes[0][0].set(xlabel="Time (ns)", ylabel=variables[0])

        # Add the rolling average for each walker
        if rolling_avg:
            # Calculate moving average and corresponding time values
            # This way the average is centered in the middle of the values it averages over (.shift()).
            df[f'rolling_avg'] = df.groupby(hue)[variables[0]].rolling(rolling_avg).mean().shift(-round(rolling_avg/2)).droplevel(0)

            # Plot the rolling average
            sns.lineplot(ax=axes[0][0], data=df, x='time', y='rolling_avg',
                        hue=hue, legend=False,
                        palette=palette, linewidth=2)

        # Add the densityplots on the side.
        if density:
            # Plot the stacked density for each walker
            # axes[0][1] is the right plot for each row. (The small one)
            sns.kdeplot(ax=axes[0][1],
                        data=df,
                        y=variables[0],
                        hue=hue,
                        multiple="stack",
                        legend=True,
                        fill=True,
                        palette=palette,
                        alpha=.5,
                        linewidth=0)

            # Remove labels
            axes[0][1].get_xaxis().set_visible(False)
            axes[0][1].get_yaxis().set_visible(False)

            # Plot a line saying dist (distribution)
            plt.text(0.5, 0.04, 'dist', horizontalalignment='center', verticalalignment='center', transform=axes[0][1].transAxes)

        # Set legend outside of box, or, if only 1 walker, remove legend
        if n_hues > 1:
            axes[0][0].legend([],[], frameon=False) if density else 0
            sns.move_legend(axes[0][legend_loc], "upper left", bbox_to_anchor=(1, 1))
        else:
            for ax in axes.reshape(-1):
                ax.legend([],[], frameon=False)

        # Set x-range
        if "xrange" in kwargs:
            axes[0][0].set_xlim(kwargs['xrange'])

        # Set y-range
        if "yrange" in kwargs:
            axes[0][0].set_ylim(kwargs['yrange'])
            axes[0][1].set_ylim(kwargs['yrange']) if density else 0

    # Make 2D scatterplot with the time in the colorbar. 
    elif len(variables) == 2:

        if density:
            main_ax = axes[1][0]
        else:
            main_ax = axes[0][0]

        # Plot the timeseries
        # sns.kdeplot(ax = axes[1][0], data=df, x=variables[0], y=variables[1], hue=hue, palette=palette, alpha=0.1)
        sns.lineplot(ax=main_ax,
                     data=df,
                     x=variables[0],
                     y=variables[1],
                     hue=hue,
                     sort=False,
                     palette=palette,
                     linewidth=0.1,
                     alpha=alpha)

        # Add the proper labels for each axes
        if label_dict:
            main_ax.set(xlabel=label_dict[variables[0]], ylabel=label_dict[variables[1]])
        else:
            main_ax.set(xlabel=variables[0], ylabel=variables[1])

        # Add the rolling average for each walker
        if rolling_avg:
            # Calculate moving average and corresponding time values
            # This way the average is centered in the middle of the values it averages over (.shift()).
            df[f'rolling_avg_x'] = df.groupby(hue)[variables[0]].rolling(rolling_avg).mean().shift(-round(rolling_avg/2)).droplevel(0)
            df[f'rolling_avg_y'] = df.groupby(hue)[variables[1]].rolling(rolling_avg).mean().shift(-round(rolling_avg/2)).droplevel(0)

            # Plot the rolling average
            sns.lineplot(ax=main_ax,
                         data=df,
                         x='rolling_avg_x',
                         y='rolling_avg_y',
                         hue=hue,
                         legend=False,
                         palette=palette,
                         linewidth=1)

        # Add the densityplots on the side.
        if density:

            # Plot distribution of X in the thin upper left plot.
            sns.kdeplot(ax=axes[0][0],
                        data=df,
                        x=variables[0],
                        hue=hue,
                        multiple="stack",
                        legend=False,
                        fill=True,
                        palette=palette,
                        alpha=.3,
                        linewidth=0)

            # Remove axes and add the word dist.       
            axes[0][0].get_xaxis().set_visible(False)
            axes[0][0].get_yaxis().set_visible(False)
            axes[0][0].text(0.02, 0.5, 'dist', horizontalalignment='center', verticalalignment='center', rotation=270, transform=axes[0][0].transAxes)
            
            # Plot distribution of Y in the thin lower right plot.
            sns.kdeplot(ax=axes[1][1],
                        data=df,
                        y=variables[0],
                        hue=hue,
                        multiple="stack",
                        legend=False,
                        fill=True,
                        palette=palette,
                        alpha=.3,
                        linewidth=0)

            # Remove axes and add the word dist.       
            axes[1][1].get_xaxis().set_visible(False)
            axes[1][1].get_yaxis().set_visible(False)
            axes[1][1].text(0.5, 0.02, 'dist', horizontalalignment='center', verticalalignment='center', transform=axes[1][1].transAxes)

            # Plot 2D distribution in the small upper right plot.
            try:
                sns.kdeplot(ax=axes[0][1],
                            data=df,
                            x=variables[0],
                            y=variables[1],
                            legend=False,
                            shade=True,
                            cmap=palette,
                            alpha=.3)

            except Exception as e:
                print(f"Warning: {e} --> not showing plot in upper right corner.")

            # Remove axes       
            axes[0][1].get_xaxis().set_visible(False)
            axes[0][1].get_yaxis().set_visible(False)

        # Some more layout options
        # Add the proper labels for each axes
        if label_dict:
            axes[0][0].set(xlabel=label_dict[variables[0]], ylabel=label_dict[variables[1]])
        else:
            axes[0][0].set(xlabel=variables[0], ylabel=variables[1])

        # Set x-range
        if "xrange" in kwargs:
            axes[0][0].set_xlim(kwargs['xrange'])
            axes[1][0].set_xlim(kwargs['xrange']) if density else 0

        # Set y-range
        if "yrange" in kwargs:
            axes[1][0].set_ylim(kwargs['yrange'])
            axes[1][1].set_ylim(kwargs['yrange']) if density else 0

        # If only 1 walker, remove legend
        if not n_hues > 1:
            for ax in axes.reshape(-1):
                ax.legend([],[], frameon=False)
    else:
        raise Exception("More than 2 dimensions not (yet) supported for this plot type.")

    # Save image if needed.
    tools.save_img(f"{'_'.join(variables)}_time.pdf") if save else 0

    return fig, axes

def rmsf(df, # Input dataframe
         hue='origin', # What column to color the different lines on
         palette="flare_r", # Colormap
         label_dict=None, # Linking codes/column names with a more elaborate title to plot on the axes
         save=True): # Save image in the ./img folder.
    """
    rmsf plots rmsf vs residue number.
    
    :df: Input dataframe. Needs at least the columns 'rmsf', 'resid', 'origin'
    :hue: What column to color the different lines on
    :palette:  Colormap
    :label_dict: Linking codes/column names with a more elaborate title to plot on the axes
    :save:  Save image in the ./img folder.

    :return: fig and axes
    """

    # Setup default matplotlib values for layout
    tools.setup_format()

    # Setup color palette
    # Check how many different lines/inputs there are.
    n_hues = len(df[hue].unique())
    palette = tools.setup_palette(palette, n_hues)

    # Plot figure
    fig, axes = plt.subplots(figsize=(8,4))
    sns.lineplot(ax=axes,
                 data=df,
                 x='resid',
                 y='rmsf',
                 hue=hue,
                 legend=True,
                 palette=palette,
                 linewidth=2,
                 marker='.')

    if len(df['origin'].unique()) >1:
        sns.move_legend(axes, "upper left", bbox_to_anchor=(1, 1))
    else:
        axes.legend([],[], frameon=False)

    plt.xticks(np.arange(min(df["resid"]), max(df["resid"]) + 1, 1.0))

    # Plot stdev if it is there
    if 'rmsf_std' in df.columns:
        axes.fill_between(df['resid'], df['rmsf'] - df['rmsf_std'], df['rmsf'] + df['rmsf_std'], alpha=0.2, label='stdev')

    # Add the proper labels for each axes
    if label_dict:
        try:
            axes.set(xlabel=label_dict["resid"], ylabel=label_dict["rmsf"])
        except:
            print("Warning: couldn't find the labels in the label dict.")
            axes.set(xlabel="Residue ID", ylabel="RMSF (nm)")
    else:
        axes.set(xlabel="Residue ID", ylabel="RMSF (nm)")

    # Save image if needed.
    tools.save_img("rmsf.pdf") if save else 0

    return fig, axes

# MetaD/OPES specific 
def fes(df,
        variables=None,
        fe_max=None,
        last_x_procent=20,
        cbar='continuous',
        palette='Spectral_r',
        kind='contourf',
        n_levels=None,
        label_dict=None,
        save=True,
        **kwargs):
    """
    fes plots fes vs cv (over time).
    
    :param df: Input dataframe. Needs at least the columns 'fes', '[cv]', 'time'
    :param variables: Variable(s) to plot, has to correspond to the columns header in the df.
    :param fe_max: Cap the Free Energy to a certain value.
    :param last_x_procent: Plot the evolution of FES over last x procent.
    :param cbar: How to show colorbar: 'continuous', 'discrete' or 'off'
    :param n_levels: Number of levels for the colormap
    :param palette: Colormap
    :param kind: Type of 2D plot. Options: "hist", "contour", "contourf" or "dc"
    :param n_levels: Number of levels for the colormap
    :param label_dict: Linking codes/column names with a more elaborate title to plot on the axes
    :param save:  Save image in the ./img folder.

    :return: fig and axes
    """

    # Make a copy to make sure you're not editing the original df.
    df = df.copy()

    # Setup default matplotlib values for layout
    tools.setup_format()

    # Setup color palette
    # Check how many different lines/inputs there are.
    palette = tools.setup_palette(palette, None, as_cmap=True)

    # Setup variable list
    variables = tools.setup_variables(df, variables)

    # Save variables as list
    if type(variables) is not list:
        variables = [variables]

    # fes of cv1 over time
    if len(variables) == 1:

        # If you want to cap the FE
        if fe_max:
            df['fes'].where(df['fes'] < fe_max, fe_max, inplace=True)
        
        # Only plot last x procent
        time_cutoff = (df['time'].max() - df['time'].min()) / 100 * (100 - last_x_procent)
        df = df[df['time'] > time_cutoff]

        print(time_cutoff)
        print(df)

        fig, axes = plt.subplots(1, 1, figsize=(8,4))

        g = sns.lineplot(ax=axes,
                         data=df,
                         x=variables[0],
                         y='fes',
                         hue='time',
                         linewidth=1.2,
                         palette=palette)

        if cbar == 'continuous':
            # Create colorbar
            norm = plt.Normalize(df['time'].min(), df['time'].max())
            sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
            sm.set_array([])

            # Replace legend with new colorbar
            g.get_legend().remove()
            g.figure.colorbar(sm, label="Time (ns)")

        elif cbar == 'discrete':
            # Put legend outside of plot
            g.legend(loc='upper left', bbox_to_anchor=(1, 1))
        elif cbar == 'off':
            # Remove legend
            g.get_legend().remove()
        else:
            sys.exit("ERROR: cbar must be one of 'continuous', 'discrete' or 'off'")

        # Add the proper labels for each axes
        try:
            axes.set(xlabel=label_dict[variables[0]], ylabel=f"Free Energy ($k_BT$)")
        except:
            print("Warning: couldn't find the labels in the label dict.")
            axes.set(xlabel=variables[0], ylabel=f"Free Energy ($k_BT$)")
        
        # Set x-range
        try:
            axes.set_xlim(kwargs['xrange'])
        except:
            pass

        # Set y-range
        try:
            axes.set_ylim(kwargs['yrange'])
        except:
            axes.set_ylim(bottom=0)

    elif len(variables) == 2:
        # Setup plot
        fig, axes = plt.subplots(figsize=(8,4))
        pal = plt.get_cmap(palette).copy()
        pal.set_bad('white', 1.0)

        # Turn df into 2d grid.
        fes = df[df['time'] == df['time'].unique()[-1]].pivot(index=variables[1], columns=variables[0], values='fes')
        
        # If you want to cap the FE
        if fe_max:
            fes[fes > fe_max] = np.nan
        else:
            fe_max = round(fes.values.max())

        # Make levels each 1 kBT (or each 0.1 kBT if fe_max < 1)
        if not n_levels:
            n_levels = fe_max if fe_max > 1 else round(fe_max * 10)

        levels = np.linspace(0.0, fe_max, n_levels + 1)

        # Define data            
        X = fes.columns.values
        Y = fes.index.values
        Z = fes.values

        # Plot   
        if kind == "hist":
            img = axes.pcolormesh(X, Y, Z, cmap=palette, shading='auto')   
        elif kind == "contour":         
            img = axes.contour(X, Y, Z, cmap=palette, levels=levels, corner_mask=True, extend='max')
        elif kind == "contourf":
            img = axes.contourf(X, Y, Z, cmap=palette, levels=levels, corner_mask=True, extend='max')
        elif kind =="dc":
            img = axes.contour(X, Y, Z, colors="#000000", levels=levels, corner_mask=True, extend='max', linewidths=0.4)

            c_levels = np.linspace(0.0, fe_max, 50*(n_levels + 1))
            img = axes.contourf(X, Y, Z, cmap=palette, levels=c_levels, corner_mask=True, extend='max')
        else:
            raise ValueError(f"kind ({kind}) not supported (in this dimension). Use 'hist', 'contour', 'contourf.")

        plt.colorbar(img, ax=axes, label="Free energy ($k_BT$)")
        
        axes.set(xlabel=variables[0])
        axes.set(ylabel=variables[1])

        # Some more layout options

        # Add the proper labels for each axes
        try:
            axes.set(xlabel=label_dict[variables[0]], ylabel=label_dict[variables[1]])
        except:
            print("Warning: couldn't find the labels in the label dict.")
            axes.set(xlabel=variables[0], ylabel=variables[1])

        # Set x-range
        try:
            axes.set_xlim(kwargs['xrange'])
        except:
            pass

        # Set y-range
        try:
            axes.set_ylim(kwargs['yrange'])
        except:
            pass

    # Other dimensions are not yet implemented
    else:
        raise Exception("More than 2 dimensions not (yet) supported for this plot type.")


    # Save image if needed.
    tools.save_img(f"fes_rew_{'_'.join(variables)}.pdf") if save else 0

    return None

def fes_rew(df,
            weights,
            variables=None,
            kind='hist',
            mintozero=True,
            n_bins=100,
            fe_max=None,
            n_levels=None,
            palette='Spectral_r',
            label_dict=None,
            save=True,
            **kwargs):
    """
    fes plots a reweighted FES using a time series (ts) and weights for reweighting. 1D and 2D. '''
    
    :param df: Input dataframe. Needs at least the columns 'fes', '[cv]', 'time', 'origin'
    :param weights: Dataframe containg the weights for each cv
    :param variables: Variable(s) to plot, has to correspond to the columns header in the df.
    :param kind: Type of plot. Options for 1D: "fes", "hist", "kde" and for 2D: "hist", "contour", "contourf" or "dc"
    :param min_to_zero: Set minimum FE value to zero.
    :param n_bins: Number of bins for each cv.
    :param fe_max: Cap the Free Energy to a certain value.
    :param n_levels:, Number of levels for the colormap
    :param palette: Colormap
    :param label_dict: Linking codes/column names with a more elaborate title to plot on the axes
    :param save:  Save image in the ./img folder.

    :return: fig and axes
    """

    # Make a copy to make sure you're not editing the original df.
    df = df.copy()

    # Setup default matplotlib values for layout
    tools.setup_format()

    # Setup color palette
    palette = tools.setup_palette(palette)

    # Setup variable list
    variables = tools.setup_variables(df, variables)

    # Merge weights and dataframe
    df = df.merge(weights, how="inner", on=["time", "origin"])

    if len(variables) == 1:
        # Setup plot
        fig, axes = plt.subplots(figsize=(8,4))
        axes_dist = axes.twinx()
        
        # Calc and plot probability distribution
        # Print only FES
        if kind == "fes":
            hist, bins = np.histogram(df[variables[0]], bins=n_bins, weights=df['weights']) # Make histogram
            hist = hist / hist.sum() # Normalize (sum is 1)
            bin_width = bins[1] - bins[0] # Get bar width
            bin_centers = (bins[:-1] + bins[1:]) / 2 # Get bin centers

        # Print FES with histograms
        elif kind == "hist":
            for index, w in enumerate(['simulation', 'reweighted']):
                # Calculate histogram
                if w == 'simulation':
                    hist, bins = np.histogram(df[variables[0]], bins=n_bins) # Make histogram
                else:
                    hist, bins = np.histogram(df[variables[0]], bins=n_bins, weights=df['weights']) # Make histogram
                hist = hist / hist.sum() # Normalize (sum is 1)
                bin_width = bins[1] - bins[0] # Get bar width
                bin_centers = (bins[:-1] + bins[1:]) / 2 # Get bin centers

                # Plot
                axes_dist.bar(bin_centers, hist, width=bin_width, color=["#808080", "#A43820"][index], alpha=0.1, label=w)
                axes_dist.yaxis.set_ticklabels([])

        # Print FES with KDEs
        elif kind =='kde':
            bin_centers = np.linspace(df[variables[0]].min(), df[variables[0]].max(), 100)
            bin_width = bin_centers[1] - bin_centers[0]
            
            for index, w in enumerate(['simulation', 'reweighted']):
                kde = KernelDensity(bandwidth=bin_width, kernel='gaussian')
                if w == 'simulation':
                    kde.fit(df[variables[0]].values.reshape([-1,1]))
                else:
                    kde.fit(df[variables[0]].values.reshape([-1,1]), sample_weight=df['weights'].values)
                hist = np.exp(kde.score_samples(bin_centers.reshape(-1,1)))
            
                # Plot kernel density
                axes_dist.fill_between(bin_centers, hist, color=["#808080", "#A43820"][index], linewidth=0, alpha=0.1, label=w)

        else:
            raise ValueError(f"kind ({kind}) not supported (in this dimensionality). Use 'fes', 'hist' of 'kde'.")

        # Calculate FES (and ignore divide by zero error.)
        with np.errstate(divide='ignore'):
            fes = -np.log(hist) # in kBT

        # If needed, set minimum to zero
        if mintozero:
            fes = fes - np.min(fes)

        # Plot FES
        axes.plot(bin_centers, fes, color=["#808080", "#A43820"][1], linewidth=2)

        # Some more layout options
        
        # Add the proper labels for each axes
        try:
            axes.set(xlabel=label_dict[variables[0]], ylabel=f"Free Energy ($k_BT$)")
        except:
            print("Warning: couldn't find the labels in the label dict.")
            axes.set(xlabel=variables[0], ylabel=f"Free Energy ($k_BT$)")
        
        # Set x-range
        try:
            axes.set_xlim(kwargs['xrange'])
            axes_dist.set_xlim(kwargs['xrange'])
        except:
            pass

        # Set y-range
        try:
            axes.set_ylim(kwargs['yrange'])
        except:
            axes.set_ylim(bottom=0)


        # set y-range of distribution
        axes_dist.set_ylim(bottom=0)
        axes_dist.yaxis.set_ticklabels([])
        axes_dist.legend(title='Sampling distribution') if kind != 'fes' else 0

    elif len(variables) == 2:
        # Setup plot
        fig, axes = plt.subplots(figsize=(8,4))
        pal = plt.get_cmap(palette).copy()
        pal.set_bad('white', 1.0)

        x_bins = np.linspace(df[variables[0]].min(), df[variables[0]].max(), n_bins)
        y_bins = np.linspace(df[variables[1]].min(), df[variables[1]].max(), n_bins)
        
        hist, x_bins, y_bins = np.histogram2d(df[variables[0]], df[variables[1]], bins=(x_bins, y_bins), weights=df['weights'])
        
        # Histogram does not follow Cartesian convention,
        # therefore transpose H for visualization purposes.
        hist = hist.T
        hist = hist / hist.sum() # Normalize (sum is 1)
        
        X, Y = np.meshgrid(x_bins, y_bins)
        
        # Get the center values of the bins
        X_c = (x_bins[:-1] + x_bins[1:]) / 2
        Y_c = (y_bins[:-1] + y_bins[1:]) / 2

        # Calculate FES (and ignore divide by zero error.)
        with np.errstate(divide='ignore'):
            fes = -np.log(hist) # in kBT

        # If needed, set minimum to zero
        if mintozero:
            fes = fes - np.min(fes)
        
        # If you want 
        if fe_max:
            fes[fes > fe_max] = np.nan

        # Make levels each 1 kBT (or each 0.1 kBT if fe_max < 1)
        if not n_levels:
            n_levels = fe_max if fe_max > 1 else round(fe_max * 10)

        levels = np.linspace(0.0, fe_max, n_levels + 1)
            
        # Plot   
        if kind == "hist":
            img = axes.pcolormesh(X, Y, fes, cmap=palette, shading='auto')   
        elif kind == "contour":         
            img = axes.contour(X_c, Y_c, fes, cmap=palette, levels=levels, corner_mask=True, extend='max')
        elif kind == "contourf":
            img = axes.contourf(X_c, Y_c, fes, cmap=palette, levels=levels, corner_mask=True, extend='max')
        elif kind =="dc":
            img = axes.contour(X_c, Y_c, fes, colors="#000000", levels=levels, corner_mask=True, extend='max', linewidths=0.4)

            c_levels = np.linspace(0.0, fe_max, 50*(n_levels + 1))
            img = axes.contourf(X_c, Y_c, fes, cmap=palette, levels=c_levels, corner_mask=True, extend='max')
        else:
            raise ValueError(f"kind ({kind}) not supported (in this dimension). Use 'hist', 'contour', 'contourf.")

        plt.colorbar(img, ax=axes, label="Free energy ($k_BT$)")
        
        axes.set(xlabel=variables[0])
        axes.set(ylabel=variables[1])

        # Some more layout options

        # Add the proper labels for each axes
        try:
            axes.set(xlabel=label_dict[variables[0]], ylabel=label_dict[variables[1]])
        except:
            print("Warning: couldn't find the labels in the label dict.")
            axes.set(xlabel=variables[0], ylabel=variables[1])

        # Set x-range
        try:
            axes.set_xlim(kwargs['xrange'])
        except:
            pass

        # Set y-range
        try:
            axes.set_ylim(kwargs['yrange'])
        except:
            axes.set_ylim(bottom=0)

    # Other dimensions are not yet implemented
    else:
        raise Exception("More than 2 dimensions not (yet) supported for this plot type.")


    # Save image if needed.
    tools.save_img(f"fes_rew_{'_'.join(variables)}.pdf") if save else 0

    return fig, axes

def kernels_time(kernels, save=True):
    ''' Plot kernels over time '''

    # Make a copy to make sure you're not editing the original df.
    df = df.copy()

    # Setup default matplotlib values for layout
    tools.setup_format()

    # Find cvs
    # cvs = list(kernels.loc[:, 'time': kernels.columns[kernels.columns.str.startswith('sigma')].tolist()[0]].columns.values[1:-1])
    cvs = [cv.split("_")[-1] for cv in kernels.columns[kernels.columns.str.startswith('sigma')].tolist()]

    if len(cvs) == 2:
        # Plot image
        fig, axes = plt.subplots(figsize=(6,4))
        p = axes.scatter(kernels[cvs[0]], kernels[cvs[1]], 
                    c=kernels['time']/1000,
                    cmap=plt.cm.get_cmap('Spectral'),
                    s=0.01, label='Kernel')
        axes.set(xlabel=cvs[0], ylabel=cvs[1])
        plt.legend()

        cbar = plt.colorbar(p)
        cbar.ax.set_label('Time (ns)')

    # Other dimensions are not yet implemented
    else:
        raise Exception("Other than 2 dimensions not (yet) supported for this plot type.")


    # Save image if needed.
    tools.save_img("kernels_time.pdf") if save else 0

    return 0