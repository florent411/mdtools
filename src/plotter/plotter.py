#!/usr/bin/env python 

''' Plot images '''

import numpy as np
import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.neighbors import KernelDensity

# Included submodules
import plotter.modules.tools as tools

def timeseries(df, variables, hue='origin', density=True, stride=10, rolling_avg=None, label_dict=None, palette="flare_r", save=True):
    ''' Plot cv(s) vs time using a time series (ts).'''

    # Setup default matplotlib values for layout
    tools.setup_format()

    # Stride, otherwise plotting gets very slow. (And reset index, otherwise Searborn complains)
    df = df[df['time'] % stride == 0].reset_index()

    # Convert time from ps to ns
    df['time'] /= 1000

    # Check how many different lines/inputs there are.
    n_hues = len(df[hue].unique())

    # Setup colorpalette if a list of colors is inputted
    if type(palette) is list:
        palette = sns.blend_palette(palette, n_hues)

    # Save variables as list
    if type(variables) is not list:
        variables = [variables]

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
        sns.lineplot(ax = axes[0][0], data=df, x='time', y=variables[0],
                          hue=hue,
                          palette=palette, linewidth=0.2, alpha=alpha)

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
            sns.kdeplot(ax=axes[0][1], data=df, y=variables[0],
                        hue=hue, multiple="stack", legend=True,
                        fill=True, palette=palette, alpha=.5, linewidth=0)
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

    # Make 2D scatterplot with the time in the colorbar. 
    elif len(variables) == 2:

        # Plot the timeseries
        # sns.kdeplot(ax = axes[1][0], data=df, x=variables[0], y=variables[1], hue=hue, palette=palette, alpha=0.1)
        sns.lineplot(ax = axes[1][0], data=df, x=variables[0], y=variables[1],
                        hue=hue, sort=False,
                        palette=palette, linewidth=0.1, alpha=alpha)

        # Add the proper labels for each axes
        if label_dict:
            axes[1][0].set(xlabel=label_dict[variables[0]], ylabel=label_dict[variables[1]])
        else:
            axes[1][0].set(xlabel=variables[0], ylabel=variables[1])

        # Add the rolling average for each walker
        if rolling_avg:
            # Calculate moving average and corresponding time values
            # This way the average is centered in the middle of the values it averages over (.shift()).
            df[f'rolling_avg_x'] = df.groupby(hue)[variables[0]].rolling(rolling_avg).mean().shift(-round(rolling_avg/2)).droplevel(0)
            df[f'rolling_avg_y'] = df.groupby(hue)[variables[1]].rolling(rolling_avg).mean().shift(-round(rolling_avg/2)).droplevel(0)

            # Plot the rolling average
            sns.lineplot(ax=axes[1][0], data=df, x='rolling_avg_x', y='rolling_avg_y',
                        hue=hue, legend=False,
                        palette=palette, linewidth=1)

        # Add the densityplots on the side.
        if density:

            # Plot distribution of X in the thin upper left plot.
            sns.kdeplot(ax=axes[0][0], data=df, x=variables[0],
                        hue=hue, multiple="stack", legend=False,
                        fill=True, palette=palette, alpha=.3, linewidth=0)
            # Remove axes and add the word dist.       
            axes[0][0].get_xaxis().set_visible(False)
            axes[0][0].get_yaxis().set_visible(False)
            axes[0][0].text(0.02, 0.5, 'dist', horizontalalignment='center', verticalalignment='center', rotation=270, transform=axes[0][0].transAxes)
            
            # Plot distribution of Y in the thin lower right plot.
            sns.kdeplot(ax=axes[1][1], data=df, y=variables[0],
                        hue=hue, multiple="stack", legend=False,
                        fill=True, palette=palette, alpha=.3, linewidth=0)
            # Remove axes and add the word dist.       
            axes[1][1].get_xaxis().set_visible(False)
            axes[1][1].get_yaxis().set_visible(False)
            axes[1][1].text(0.5, 0.02, 'dist', horizontalalignment='center', verticalalignment='center', transform=axes[1][1].transAxes)

            # Plot 2D distribution in the small upper right plot.
            try:
                sns.kdeplot(ax=axes[0][1], data=df, x=variables[0], y=variables[1], legend=False,
                            shade=True, cmap=palette, alpha=.3)
            except Exception as e:
                print(f"Warning: {e} --> not showing plot in upper right corner.")

            # Remove axes       
            axes[0][1].get_xaxis().set_visible(False)
            axes[0][1].get_yaxis().set_visible(False)

        else:
            # Plot the timeseries for each walker and color by time
            # axes[1][0] is the left plot for each row. (The big one)
            sns.lineplot(ax = axes[0][0], data=df, x=variables[0], y=variables[1],
                        hue=hue, sort=False,
                        palette=palette, linewidth=0.1, alpha=0.8)

        # Add the proper labels for each axes
        if label_dict:
            axes[0][0].set(xlabel=label_dict[variables[0]], ylabel=label_dict[variables[1]])
        else:
            axes[0][0].set(xlabel=variables[0], ylabel=variables[1])

        # If only 1 walker, remove legend
        if not n_hues > 1:
            for ax in axes.reshape(-1):
                ax.legend([],[], frameon=False)
    else:
        raise Exception("More than 2 dimensions not (yet) supported for this plot type.")

    # Save image if needed.
    tools.save_img(f"{'_'.join(variables)}_time.pdf") if save else 0

    return None

def rmsf(df, palette="flare_r", hue="origin", label_dict=None, save=True):
    ''' Plot rmsf vs residue number '''

    # Setup default matplotlib values for layout
    tools.setup_format()

    # Check how many different lines/inputs there are.
    n_hues = len(df[hue].unique())

    # Setup colorpalette if a list of colors is inputted
    if type(palette) is list:
        palette = sns.blend_palette(palette, n_hues)

    # Plot figure
    fig, axes = plt.subplots(figsize=(6,4))
    sns.lineplot(ax=axes, data=df, x='resid', y='rmsf',
                hue=hue, legend=True,
                palette=palette, linewidth=2, marker='.')

    sns.move_legend(axes, "upper left", bbox_to_anchor=(1, 1))


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
            axes.set(xlabel="ResID", ylabel="RMSF (nm)")
    else:
        axes.set(xlabel="ResID", ylabel="RMSF (nm)")

    # Save image if needed.
    tools.save_img("rmsf.pdf") if save else 0

    return None

# MetaD/OPES specific 
def fes_rew(df, variables, weights, kind='hist', mintozero=True, n_bins=100, fe_max=25, truncate=False, n_levels=None, palette='Spectral_r', colors=["#808080", "#A43820"], label_dict=None, save=True, **kwargs):
    ''' Plot FES using a time series (ts) and weights for reweighting. 1D and 2D. '''

    # Setup default matplotlib values for layout
    tools.setup_format()

    # Merge weights and dataframe
    df = df.merge(weights, how="inner", on=["time", "origin"])

    # Setup colorpalette if a list of colors is inputted
    if type(palette) is list:
        palette = sns.blend_palette(palette, as_cmap=True)

    # Save variables as list
    if type(variables) is not list:
        variables = [variables]

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
                axes_dist.bar(bin_centers, hist, width=bin_width, color=colors[index], alpha=0.1, label=w)
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
                axes_dist.fill_between(bin_centers, hist, color=colors[index], linewidth=0, alpha=0.1, label=w)

        else:
            raise ValueError(f"kind ({kind}) not supported (in this dimensionality). Use 'fes', 'hist' of 'kde'.")

        # Calculate FES (and ignore divide by zero error.)
        with np.errstate(divide='ignore'):
            fes = -np.log(hist) # in kBT

        # If needed, set minimum to zero
        if mintozero:
            fes = fes - np.min(fes)

        # Plot FES
        axes.plot(bin_centers, fes, color=colors[1], linewidth=2)

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
        if truncate:
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

    return None

def kernels_time(kernels, save=True):
    ''' Plot kernels over time '''

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