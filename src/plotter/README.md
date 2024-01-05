# Plotter Module Documentation

## Introduction

This Python module provides a set of functions for creating various types of plots commonly used in data analysis, particularly for time series, root mean square fluctuations (RMSF), and free energy surfaces (FES) in molecular dynamics simulations. The module utilizes the `seaborn` and `matplotlib` libraries for creating aesthetically pleasing and informative visualizations.

## Functions

### `timeseries`

This function is designed for plotting time series data, typically representing a chemical or physical property over time. It supports various plot types such as line plots, scatter plots, and kernel density estimation (kde) plots. Additionally, it allows for the inclusion of marginal density distributions.

**Parameters:**
- `df`: Input dataframe with columns 'time', '[cv]', 'origin'.
- `variables`: Variable(s) to plot (corresponding to columns in the dataframe).
- `hue`: Column to color the different lines on.
- `density`: Boolean to plot marginal density distribution of the y-axis.
- `stride`: Number of data points to skip for faster plotting.
- `moving_avg`: Plot moving average over a faded version of the full dataset plot.
- `label_dict`: Dictionary linking codes/column names with more elaborate titles for axes.
- `palette`: Colormap for coloring lines.
- `p_type`: Type of plot - 'line', 'scatter', or 'kde'.
- `s_size`: Size of scatter points.
- `save`: Boolean to save the plot image in the `./img` folder.
- `**kwargs`: Additional keyword arguments for customization.

**Returns:**
- `fig`: Matplotlib figure.
- `axes`: Matplotlib axes.

### `rmsf`

This function plots the root mean square fluctuation (RMSF) against residue number.

**Parameters:**
- `df`: Input dataframe with columns 'rmsf', 'resid', 'origin'.
- `hue`: Column to color the different lines on.
- `palette`: Colormap.
- `label_dict`: Dictionary linking codes/column names with more elaborate titles for axes.
- `save`: Boolean to save the plot image in the `./img` folder.

**Returns:**
- `fig`: Matplotlib figure.
- `axes`: Matplotlib axes.

### `fes`

The `fes` function creates free energy surfaces (FES) plots against collective variables (CVs) over time. It supports 1D and 2D plots with options for contour, contourf, histogram, and more.

**Parameters:**
- `df`: Input dataframe with columns 'fes', '[cv]', 'time'.
- `variables`: Variable(s) to plot.
- `fe_max`: Cap the Free Energy to a certain value.
- `last_x_procent`: Plot the evolution of FES over the last x percent of time.
- `cbar`: How to show the colorbar - 'continuous', 'discrete', or 'off'.
- `palette`: Colormap.
- `kind`: Type of 2D plot - "hist", "contour", "contourf", or "dc".
- `n_levels`: Number of levels for the colormap.
- `label_dict`: Dictionary linking codes/column names with more elaborate titles for axes.
- `save`: Boolean to save the plot image in the `./img` folder.
- `**kwargs`: Additional keyword arguments for customization.

**Returns:**
- `fig`: Matplotlib figure.
- `axes`: Matplotlib axes.

### `fes_rew`

The `fes_rew` function plots a reweighted FES using a time series and weights for reweighting. It supports 1D and 2D plots.

**Parameters:**
- `df`: Input dataframe with columns 'fes', '[cv]', 'time', 'origin'.
- `variables`: Variable(s) to plot.
- `fe_max`: Cap the Free Energy to a certain value.
- `n_levels`: Number of levels for the colormap.
- `dist`: Boolean to include distribution plots.
- `palette`: Colormap.
- `dist_palette`: Colormap for distribution plots.
- `kind`: Type of plot - "fes", "hist", "kde", "contour", "contourf", or "dc".
- `label_dict`: Dictionary linking codes/column names with more elaborate titles for axes.
- `save`: Boolean to save the plot image in the `./img` folder.
- `**kwargs`: Additional keyword arguments for customization.

**Returns:**
- `fig`: Matplotlib figure.
- `axes`: Matplotlib axes.

## Usage

1. Import the module in your Python script or notebook.

```python
import plotter.main as plot
```

2. Prepare your data in a Pandas DataFrame with the required columns.

3. Call the desired plotting function with the appropriate parameters. E.g.

4. Customize the plot as needed using the returned `fig` and `axes` objects.

5. Optionally, save the plot using the `save=True` parameter.

```python
# Example usage for timeseries plot
fig, axes = timeseries(df, variables=['cv1', 'cv2'], hue='origin', density=True, save=True)
```

## Notes

- Ensure that the required dependencies (`seaborn`, `matplotlib`, `numpy`, `pandas`) are installed in your Python environment.

- Make sure that your data conforms to the expected column names and structure specified in the function documentation.

- Explore additional customization options using the provided keyword arguments.

- Check the `./img` folder for saved plot images if the `save=True` option is used.

Happy plotting!