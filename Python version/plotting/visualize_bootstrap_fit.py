#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 11:37:06 2025

@author: fangfang
"""

import jax
jax.config.update("jax_enable_x64", True)
import dill as pickled
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import replace
sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/Python version/')
from analysis.utils_load import select_file_and_get_path
sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids')
from plotting.wishart_predictions_plotting import WishartPredictionsVisualization
from analysis.ellipses_tools import find_inner_outer_contours
from plotting.wishart_plotting import PlotSettingsBase 
from plotting.wishart_predictions_plotting import Plot2DPredSettings

#specify the file name
base_dir = '/Volumes/T9/Aguirre-Brainard Lab Dropbox/Fangfang Hong/ELPS_analysis/'

#%%
#---------------------------------------------------------------------------
# SECTION 1: load the model fits to the empirical data
# --------------------------------------------------------------------------
# Select the file containing the model fits
# Navigate to the directory: ELPS_analysis/Experiment_DataFiles/sub#
input_fileDir_fits, file_name = select_file_and_get_path()

# Construct the full path to the selected file
full_path = os.path.join(input_fileDir_fits, file_name)

# Load the necessary variables from the file
with open(full_path, 'rb') as f:
    vars_dict = pickled.load(f)

# Retrieve the variables of interest from the loaded dictionary
# - Model predictions using the Wishart process
model_pred = vars_dict['model_pred_Wishart']

# - Transformation matrices for converting between DKL, RGB, and W spaces
color_thres_data = vars_dict['color_thres_data']

# - Grid used for computing model predictions
grid = vars_dict['grid']

# - Experimental trial data
expt_trial = vars_dict['expt_trial']

# - Number of grid points per dimension for model prediction computations
num_grid_pts = grid.shape[0]

# - Dimensionality of the color space (e.g., 2D for isoluminant planes)
ndims = color_thres_data.color_dimension

#%% 
#---------------------------------------------------------------------------
# SECTION 2: load the model fits to the bootstrapped data
# --------------------------------------------------------------------------
# Step 1: Select the bootstrapped data file (choose one as a reference)
input_fileDir_fits_btst, file_name_btst = select_file_and_get_path()

# Define parameters
nBtst = 10  # Number of bootstrapped datasets to load
nTheta = 1000  # Resolution of ellipses for plotting
btst_datasets = list(range(nBtst))  # List of bootstrap dataset indices

# Step 2: Initialize storage arrays
# - `params_all`: Stores ellipse parameters for all grid points across bootstrapped datasets
#   - Ellipse parameters: [x-center, y-center, major axis length, minor axis length, rotation angle]
params_all = np.full((num_grid_pts, num_grid_pts, nBtst, 5), np.nan)

# - `params_indv_fits_all`: Stores ellipse parameters for individual fits to each grid point
#   - Only for grid points used in data collection (≤ num_grid_pts)
params_indv_fits_all = np.full((num_grid_pts, num_grid_pts, nBtst, 5), np.nan)

# Step 3: Loop through each bootstrap dataset and load data
for r in btst_datasets:
    # Generate the file name for the current bootstrap dataset
    file_name_btst_r = file_name_btst.replace('AEPsych[0]', f'AEPsych[{r}]')
    full_path_btst_r = f"{input_fileDir_fits_btst}/{file_name_btst_r}"
    
    # Load the variables from the current bootstrap dataset
    with open(full_path_btst_r, 'rb') as f:
        vars_dict_btst = pickled.load(f)
    
    # Extract the ellipse parameters from the Wishart model predictions
    # - `param_ell_r`: Ellipse parameters for all grid points
    param_ell_r = vars_dict_btst['model_pred_Wishart'].params_ell
    
    # Reformat and store ellipse parameters for all grid points
    for i in range(num_grid_pts):
        for j in range(num_grid_pts):
            params_all[i, j, r] = param_ell_r[i][j]
        
#%%           
# -----------------------------------------------------------------------------
# SECTION 3: Derive the Confidence Interval of Bootstrapped Model Fits
# -----------------------------------------------------------------------------
# Define a function to compute confidence intervals (CI) for each reference color
# This function computes the upper and lower elliptical contours for each grid point
# The process is applied once for joint fits and once for individual fits
def find_CI_for_gridRefs(p_ell, nd=ndims, nTheta=nTheta):
    """
    Computes the confidence intervals for the model-predicted ellipses at each grid point.

    Args:
        p_ell (ndarray): Array of ellipse parameters for all grid points.
        nd (int): Number of dimensions in the color space (default: ndims).
        nTheta (int): Number of resolution points for each ellipse (default: nTheta).

    Returns:
        tuple:
            ell_min (ndarray): Lower (inner) contour of the ellipses for each grid point.
            ell_max (ndarray): Upper (outer) contour of the ellipses for each grid point.
    """
    # Get the grid size (assumes square grid)
    ng = p_ell.shape[0]

    # Initialize arrays to store the minimum (inner) and maximum (outer) contours
    # Shape: (grid_size, grid_size, dimensions, resolution)
    ell_min = np.full((ng, ng, nd, nTheta), np.nan)
    ell_max = np.full((ng, ng, nd, nTheta), np.nan)

    # Iterate over all grid points
    for i in range(ng):
        for j in range(ng):
            # Extract ellipse parameters for the current grid point
            params_ij = p_ell[i, j]

            # Compute the inner (xi, yi) and outer (xu, yu) contours
            xu_ij, yu_ij, xi_ij, yi_ij = find_inner_outer_contours(params_ij)

            # Determine the number of points in each contour
            idx_u = xu_ij.shape[0]  # Number of points in the outer contour
            idx_i = xi_ij.shape[0]  # Number of points in the inner contour

            # Store the outer contour (max)
            ell_max[i, j, 0, :idx_u] = xu_ij  # X-coordinates
            ell_max[i, j, 1, :idx_u] = yu_ij  # Y-coordinates

            # Store the inner contour (min)
            ell_min[i, j, 0, :idx_i] = xi_ij  # X-coordinates
            ell_min[i, j, 1, :idx_i] = yi_ij  # Y-coordinates

    return ell_min, ell_max

# Compute confidence intervals for the joint fits across all reference locations
fitEll_min, fitEll_max = find_CI_for_gridRefs(params_all)

#%%
# -----------------------------------------------------------------------------
# SECTION 4: Visualizse the CI
# -----------------------------------------------------------------------------
def plot_btst_CI(grid, ell_min, ell_max, ax):
    ng = grid.shape[0]
    for i in range(ng):
        for j in range(ng):
            if i == 0 and j == 0: lbl = f'Bootstrapped CI ({nBtst} AEPsych datasets)'
            else: lbl = None
            # Adjust the color map based on the fixed color dimension.
            cm = color_thres_data.M_2DWToRGB @ np.insert(grid[i, j], 2, 1)
            idx_max_nonan = ~np.isnan(ell_max[i, j, 0])
            ax.fill(ell_max[i,j,0,idx_max_nonan], ell_max[i,j,1,idx_max_nonan], 
                    color= cm, alpha = 0.9, lw = 0, label = lbl)
            idx_min_nonan = ~np.isnan(ell_min[i, j, 0])
            ax.fill(ell_min[i,j,0,idx_min_nonan], ell_min[i,j,1,idx_min_nonan], 
                    color='white', lw = 0)

# -----------------------------------------------------------------------------
# Plot Joint Fits
# -----------------------------------------------------------------------------
# Create the output directory if it doesn't exist
output_figDir_fits = input_fileDir_fits_btst.replace('DataFiles', 'FigFiles')
os.makedirs(output_figDir_fits, exist_ok=True)

# Create a base plotting settings instance (shared across plots)
pltSettings_base = PlotSettingsBase(fig_dir=output_figDir_fits, fontsize = 8)

# Initialize 2D prediction settings by copying from base and overriding method-specific parameters
pred2D_settings = replace(Plot2DPredSettings(), **pltSettings_base.__dict__)
pred2D_settings = replace(pred2D_settings, 
                          visualize_samples= False,
                          visualize_gt = False,
                          visualize_model_estimatedCov = False,
                          flag_rescale_axes_label = False,
                          modelpred_alpha = 0.5,
                          modelpred_lw = 0.5,
                          modelpred_lc = 'k',
                          modelpred_ls = '-') 
# Initialize Visualization Class for Wishart Predictions
wishart_pred_vis_wCI = WishartPredictionsVisualization(expt_trial,
                                                       model_pred.model, 
                                                       model_pred, 
                                                       color_thres_data,
                                                       settings = pltSettings_base,
                                                       save_fig = False)
# Create figure and axes for plotting
fig, ax = plt.subplots(1, 1, figsize=pred2D_settings.fig_size, dpi=pred2D_settings.dpi)

# Plot the bootstrap confidence intervals on top of the grid
plot_btst_CI(grid, fitEll_min, fitEll_max, ax)

# Overlay model predictions (joint fits) onto the same axes
wishart_pred_vis_wCI.plot_2D(grid, grid, ax=ax, settings=pred2D_settings)

# Set the plot title
ax.set_title('Isoluminant plane')



