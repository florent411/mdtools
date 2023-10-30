#!/usr/bin/env python 

''' Plot images '''

import sys
import numpy as np
import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

# Included submodules
from plotter.utils import tools

def timeseries(df,
               variables,
               hue='origin',
               density=True,
               stride=10,
               moving_avg=None,
               label_dict=None,
               palette="flare_r",
               p_type='line',
               s_size=5,
               save=True,
               **kwargs):
    """
    timeseries plots cv(s) vs time.
    
    :param df: Input dataframe. Needs at least the columns 'time', '[cv]', 'origin'
    :param variables: Variable(s) to plot, has to correspond to the columns header in the df.
    :param hue: What column to color the different lines on
    :param density: Plot marginal density distribution of the y axis
    :param stride: Stride the number of datapoints
    :param moving_avg: Plot moving average over a faded version of the full dataset plot
    :param label_dict: Linking codes/column names with a more elaborate title to plot on the axes
    :param palette: Colormap
    :param ptype: Type of plot. Now supported: 'line', 'scatter' and 'kde' (kde is only for 2d).
    :param s_size: Size of scatter points.
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
    alpha = 0.5 if moving_avg else 0.8
    alpha -= 0.2 if n_hues > 1 else 0

    legend_loc = 1 if density else 0
    h_ratio = [1, 7] if len(variables) == 2 else [1]
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

        if p_type == 'line':  
            # Plot the timeseries
            sns.lineplot(ax = axes[0][0],
                        data=df,
                        x='time',
                        y=variables[0],
                        hue=hue,
                        palette=palette,
                        linewidth=0.2,
                        alpha=alpha)
        elif p_type == 'scatter':
            # Plot the timeseries
            sns.scatterplot(ax = axes[0][0],
                            data=df,
                            x='time',
                            y=variables[0],
                            hue=hue,
                            palette=palette,
                            s=s_size,
                            alpha=alpha)
        else:
            sys.exit("ERROR: Unsupported p_type: {p_type}. Please choose 'scatter' or 'line'.")


        # Add the proper labels for each axes
        try:
            axes[0][0].set(xlabel="Time (ns)", ylabel=label_dict[variables[0]])
        except:
            print("Warning: couldn't find the labels in the label dict.")
            axes[0][0].set(xlabel="Time (ns)", ylabel=variables[0])

        # Add the moving average for each walker
        if moving_avg:
            # Calculate moving average and corresponding time values
            # This way the average is centered in the middle of the values it averages over (.shift()).
            df[f'moving_avg'] = df.groupby(hue)[variables[0]].rolling(moving_avg).mean().shift(-round(moving_avg/2)).droplevel(0)

            # Plot the moving avg
            sns.lineplot(ax=axes[0][0],
                            data=df,
                            x='time',
                            y='moving_avg',
                            hue=hue,
                            legend=False,
                            palette=palette,
                            linewidth=2)

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
        if p_type == 'line':  
            # Plot the timeseries
            sns.lineplot(ax=main_ax,
                         data=df,
                         x=variables[0],
                         y=variables[1],
                         hue=hue,
                         sort=False,
                         palette=palette,
                         linewidth=0.1,
                         alpha=alpha)
        elif p_type == 'scatter':
            # Plot the timeseries
            sns.scatterplot(ax = main_ax,
                            data=df,
                            x=variables[0],
                            y=variables[1],
                            hue=hue,
                            palette=palette,
                            s=s_size,
                            alpha=alpha)
        elif p_type == 'kde':
            sns.kdeplot(ax=main_ax,
                        data=df,
                        x=variables[0],
                        y=variables[1],
                        legend=False,
                        shade=True,
                        levels=20,
                        cmap=palette)

        else:
            sys.exit("ERROR: Unsupported p_type: {p_type}. Please choose 'scatter', 'line' or 'kde'.")



        # Add the proper labels for each axes
        if label_dict:
            main_ax.set(xlabel=label_dict[variables[0]], ylabel=label_dict[variables[1]])
        else:
            main_ax.set(xlabel=variables[0], ylabel=variables[1])

        # Add the moving average for each walker
        if moving_avg:
            # Calculate moving average and corresponding time values
            # This way the average is centered in the middle of the values it averages over (.shift()).
            df[f'moving_avg_x'] = df.groupby(hue)[variables[0]].moving(moving_avg).mean().shift(-round(moving_avg/2)).droplevel(0)
            df[f'moving_avg_y'] = df.groupby(hue)[variables[1]].moving(moving_avg).mean().shift(-round(moving_avg/2)).droplevel(0)

            # Plot the moving average
            sns.lineplot(ax=main_ax,
                         data=df,
                         x='moving_avg_x',
                         y='moving_avg_y',
                         hue=hue,
                         legend=False,
                         palette=palette,
                         linewidth=1)

        # Add the densityplots on the side.
        if density:
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
                if p_type == 'kde':
                    sns.scatterplot(ax = axes[0][1],
                                    data=df,
                                    x=variables[0],
                                    y=variables[1],
                                    hue=hue,
                                    palette=palette,
                                    s=s_size/5,
                                    alpha=alpha)                
                else:
                    sns.kdeplot(ax=axes[0][1],
                                data=df,
                                x=variables[0],
                                y=variables[1],
                                legend=False,
                                fill=True,
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
    palette = tools.setup_palette(palette, n_hues=n_hues)

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

    # Setup variable list
    variables = tools.setup_variables(df, variables)

    # Save variables as list
    if type(variables) is not list:
        variables = [variables]

    # fes of cv1 over time
    if len(variables) == 1:

        # If you want to cap the FE
        if fe_max:
            df['fes'] = df['fes'].where(df['fes'] < fe_max, fe_max)
        
        # Only plot last x procent
        if len(df['time'].unique()) != 1:       
            time_cutoff = (df['time'].max() - df['time'].min()) / 100 * (100 - last_x_procent)
            df = df[df['time'] > time_cutoff]

        # Setup color palette
        # Check how many different lines/inputs there are.
        n_hues = len(df['time'].unique())
        palette = tools.setup_palette(palette, n_hues)

        fig, axes = plt.subplots(1, 1, figsize=(8,4))

        g = sns.lineplot(ax=axes,
                         data=df,
                         x=variables[0],
                         y='fes',
                         hue='time',
                         linewidth=1.2,
                         palette=palette)

        if cbar == 'off' or len(df['time'].unique()) == 1:       
            # Remove legend
            g.get_legend().remove()
        elif cbar == 'continuous':
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

        # Turn df into 2d grid and only take last timeframe.
        fes = df[df['time'] == df['time'].unique()[-1]].pivot(index=variables[1], columns=variables[0], values='fes')

        # Setup color palette
        # Check how many different lines/inputs there are.
        palette = tools.setup_palette(palette, 100)

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
    tools.save_img(f"fes_{'_'.join(variables)}.pdf") if save else 0

    return fig, axes

def fes_rew(df,
            variables=None,
            fe_max=None,
            n_levels=None,
            dist=True,
            palette='Spectral_r',
            dist_palette=['#808080', '#A43820'],
            kind='contourf',
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

    # Setup color palettes
    palette = tools.setup_palette(palette, n_hues=1)
    dist_palette = tools.setup_palette(dist_palette, n_hues=2)

    # Setup variable list
    variables = tools.setup_variables(df, variables)

    # If you want to cap the FE
    if fe_max:
        df['fes'] = df['fes'].where(df['fes'] < fe_max, fe_max)

    if len(variables) == 1:
        # Setup plot
        fig, axes = plt.subplots(figsize=(8,4))
        axes_dist = axes.twinx()

        
        # Plot (re)weighted probality distributions
        if dist:
            bin_centers = df[variables[0]].values
            bin_width = bin_centers[1] - bin_centers[0]

            for index, w in enumerate(['dist_unweighted', 'dist_reweighted']):
                # Plot
                sns.lineplot(ax=axes_dist,
                            data=df,
                            x=variables[0],
                            y=w,
                            linewidth=0)

                # Fill area under the curve
                axes_dist.fill_between(bin_centers,
                                       df[w].values,
                                       color=dist_palette[index],
                                       linewidth=0,
                                       alpha=0.3,
                                       label=w.split("_")[-1])
        else:
            pass

        # Plot FES
        sns.lineplot(ax=axes,
                     data=df,
                     x=variables[0],
                     y='fes',
                     hue='time',
                     palette=palette,
                     linewidth=2)

        # Some more layout options

        # Remove the distribution's Y-label and ticks
        axes_dist.set_ylabel('')
        axes_dist.set_yticks([])

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
        axes_dist.legend(title='Sampling distribution') if dist else 0

        # Set dist legend outside of plot
        sns.move_legend(axes_dist, "upper left", bbox_to_anchor=(1, 1))

        # Remove legend
        axes.legend([],[], frameon=False)

    elif len(variables) == 2:
        # Setup plot
        fig, axes = plt.subplots(figsize=(8,4))
        pal = plt.get_cmap(palette).copy()
        pal.set_bad('white', 1.0)

        # Turn df into 2d grid and only take last timeframe.
        fes = df[df['time'] == df['time'].unique()[-1]].pivot(index=variables[1], columns=variables[0], values='fes')

        # Setup color palette
        # Check how many different lines/inputs there are.
        palette = tools.setup_palette(palette, 100)

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

    return fig, axes


# def kernels_time(kernels, save=True):
#     ''' Plot kernels over time '''

#     # Make a copy to make sure you're not editing the original df.
#     df = df.copy()

#     # Setup default matplotlib values for layout
#     tools.setup_format()

#     # Find cvs
#     # cvs = list(kernels.loc[:, 'time': kernels.columns[kernels.columns.str.startswith('sigma')].tolist()[0]].columns.values[1:-1])
#     cvs = [cv.split("_")[-1] for cv in kernels.columns[kernels.columns.str.startswith('sigma')].tolist()]

#     if len(cvs) == 2:
#         # Plot image
#         fig, axes = plt.subplots(figsize=(6,4))
#         p = axes.scatter(kernels[cvs[0]], kernels[cvs[1]], 
#                     c=kernels['time']/1000,
#                     cmap=plt.cm.get_cmap('Spectral'),
#                     s=0.01, label='Kernel')
#         axes.set(xlabel=cvs[0], ylabel=cvs[1])
#         plt.legend()

#         cbar = plt.colorbar(p)
#         cbar.ax.set_label('Time (ns)')

#     # Other dimensions are not yet implemented
#     else:
#         raise Exception("Other than 2 dimensions not (yet) supported for this plot type.")


#     # Save image if needed.
#     tools.save_img("kernels_time.pdf") if save else 0

#     return 0


def conv_params(df,
                variables=None,
                colors=['#A43820', '#A43820', '#A43820'],
                inset_last_perc=20,
                label_dict=None,
                save=True,
                **kwargs):
    """
    Plot convergence parameters. ('KLdiv', 'JSdiv' and 'dAlonso')
    
    :param df: Input dataframe. Needs at least the columns 'time' and ('KLdiv', 'JSdiv' and/or 'dAlonso')
    :param colors: Colormap
    :param label_dict: Linking codes/column names with a more elaborate title to plot on the axes
    :param save:  Save image in the ./img folder.

    :return: fig and axes
    """

    # Make a copy to make sure you're not editing the original df.
    df = df.copy()

    # First find the variables in the input dataframe
    column_names = df.columns.to_list()
    variables = [x for x in column_names if x not in ['time']]

    # What is the cutoff for the x-axis, as a fraction of the total. (e.g. if you cutoff last 20%, x_min_frac = 0.8)
    x_min_frac = (100 - inset_last_perc) / 100

    # You can assert list to confirm list is not empty
    assert list, f"No variables found.\nOnly found the following column(s): {df.columns.values}"

    # Setup default matplotlib values for layout
    tools.setup_format()

    fig, axes = plt.subplots(len(variables), sharex=True, figsize=(8, (2 * len(variables))))
    plt.subplots_adjust(wspace=0, hspace=0)

    for index, variable in enumerate(variables):
        
        # Plot data
        sns.lineplot(data=df,
                     x='time',
                     y=variable,
                     color=colors[index % len(colors)],
                     linewidth=2,
                     ax=axes[index])

        # Make an inset ax and plot the values.
        ax_ins = inset_axes(axes[index],
                          width="40%", # width = 40% of parent_bbox
                          height=0.8, # height : 0.8 inch
                          loc=1)
        
        # Setup x-range to be x percent of the total dataset.
        ax_ins.set_xlim(df['time'].max() * x_min_frac, df['time'].max())

        # Setup y-range of the inset. You want to plot the full range in that particular last x percent. Use the margin for 10% extra whitespace around the lines.
        data_range = df[variable][int(len(df) * x_min_frac):].max() - df[variable][int(len(df) * x_min_frac):].min()
        margin = 0.1 * data_range
        ymin = df[variable][int(len(df) * x_min_frac):].min() - margin
        ymax = df[variable][int(len(df) * x_min_frac):].max() + margin
        ax_ins.set_ylim(ymin, ymax)
        
        # Plot data into inset.
        sns.lineplot(data=df,
                     x='time',
                     y=variable,
                     color=colors[index % len(colors)],
                     linewidth=2,
                     ax=ax_ins)

        # Remove/resize labels and ticks
        ax_ins.tick_params(axis='y', labelsize=10)
        ax_ins.set(xticklabels=[])  # remove the tick labels
        ax_ins.tick_params(left=False, bottom=False)  # remove the ticks
        ax_ins.set(xlabel=None, ylabel=None)

        # Add text in the upper right corner.
        # The transform=ax.transAxes argument specifies that the coordinates for the text are in axis-relative coordinates (0 to 1). 
        ax_ins.text(0.96, 0.9, f"last {float(inset_last_perc):.0f}%", transform=ax_ins.transAxes, horizontalalignment='right', verticalalignment='top', size=10)

        # Mark the inset in the regular plot and connect it to the inset.
        mark_inset(axes[index], ax_ins, loc1=3, loc2=4, fc="none", ec="0.5")        


        # Some more layout options

        # Add the proper labels for each axes
        try:
            axes[index].set(xlabel="Time (ns)", ylabel=label_dict[variables[index]])
        except:
            print("Warning: couldn't find the labels in the label dict.")
            axes[index].set(xlabel="Time (ns)", ylabel=variables[index])
        
    # Save image if needed.
    tools.save_img(f"conv_params.pdf") if save else 0

    return fig, axes