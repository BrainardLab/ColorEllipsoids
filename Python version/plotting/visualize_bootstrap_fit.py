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
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/Python version/')
from analysis.utils_load import select_file_and_get_path
sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids')
from plotting.wishart_predictions_plotting import WishartPredictionsVisualization
from analysis.ellipses_tools import find_inner_outer_contours

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
full_path = f"{input_fileDir_fits}/{file_name}"

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

# - Number of grid points per dimension for experimental data collection (<= num_grid_pts)
num_grid_pts_data = int(vars_dict['nRefs']**0.5)

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
params_indv_fits_all = np.full((num_grid_pts_data, num_grid_pts_data, nBtst, 5), np.nan)

# Step 3: Loop through each bootstrap dataset and load data
for r in btst_datasets:
    # Generate the file name for the current bootstrap dataset
    file_name_btst_r = file_name_btst.replace('btst0', f'btst{r}')
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
    
    # Extract individual ellipse fits for each grid point (if available)
    # - `param_ell_indv_fits_r`: Individual fits for reference locations
    param_ell_indv_fits_r = vars_dict_btst['model_pred_Wishart_indv_ell']
    
    for idx, p in enumerate(param_ell_indv_fits_r):
        # Map the linear index to a row and column for the data grid
        row = idx // num_grid_pts_data  # Compute row index
        col = idx % num_grid_pts_data   # Compute column index
        
        # Store individual fit parameters in the appropriate location
        params_indv_fits_all[row, col, r, :] = np.array(p.params_ell).flatten()
        
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

# Compute confidence intervals for the individual fits at each reference location
fitEll_indv_min, fitEll_indv_max = find_CI_for_gridRefs(params_indv_fits_all)

#%%
# -----------------------------------------------------------------------------
# SECTION 4: Visualizse the CI
# -----------------------------------------------------------------------------
def plot_btst_CI(grid, ell_min, ell_max, ax):
    ng = grid.shape[0]
    for i in range(ng):
        for j in range(ng):
            if i == 0 and j == 0: lbl = f'Confidence interval of {nBtst} bootstrapped datasets'
            else: lbl = None
            # Adjust the color map based on the fixed color dimension.
            cm = color_thres_data.M_2DWToRGB @ np.insert(grid[i, j], 2, 1)
            idx_max_nonan = ~np.isnan(ell_max[i, j, 0])
            ax.fill(ell_max[i,j,0,idx_max_nonan], ell_max[i,j,1,idx_max_nonan], 
                    color= cm, alpha = 0.9, edgecolor = None, label = lbl)
            idx_min_nonan = ~np.isnan(ell_min[i, j, 0])
            ax.fill(ell_min[i,j,0,idx_min_nonan], ell_min[i,j,1,idx_min_nonan], 
                    color='white')

#specify figure name and path
# -----------------------------------------------------------------------------
# Plot Joint Fits
# -----------------------------------------------------------------------------
output_figDir_fits = input_fileDir_fits_btst.replace('DataFiles', 'FigFiles')
wishart_pred_vis_wCI = WishartPredictionsVisualization(expt_trial,
                                                      model_pred.model, 
                                                      model_pred, 
                                                      color_thres_data,
                                                      fig_dir = output_figDir_fits, 
                                                      save_fig = False)
fig, ax = plt.subplots(1, 1, figsize = (3.81,4.2), dpi= 1024)
plot_btst_CI(grid, fitEll_min, fitEll_max, ax)
wishart_pred_vis_wCI.plot_2D(
    grid, 
    grid,
    ax = ax,
    visualize_samples= True,
    visualize_gt = False,
    visualize_model_estimatedCov = False,
    flag_rescale_axes_label = False,
    samples_alpha = 0.5,
    samples_label = 'Experimental data',
    sigma_lw = 0.5,
    sigma_alpha = 1,
    modelpred_alpha = 1,
    modelpred_lw = 0.5,
    modelpred_lc = 'k',
    modelpred_ls = '-') 
ax.set_xlabel('Wishart space dimension 1');
ax.set_ylabel('Wishart space dimension 2');
ax.set_title('Joint fits');
fig.savefig(output_figDir_fits+f"/{file_name_btst[:-10]}_btstCI_byJointFits_wSamples.pdf",
             format='pdf', bbox_inches='tight')  

#%% 
# -----------------------------------------------------------------------------
# Plot Individual Fits
# -----------------------------------------------------------------------------
fig2, ax2 = plt.subplots(1, 1, figsize = (3.81,4.2), dpi= 1024)
grid_data = grid[::2][:,::2]
plot_btst_CI(grid_data, fitEll_indv_min, fitEll_indv_max, ax2)
wishart_pred_vis_wCI.plot_2D(
    grid, 
    grid,
    ax = ax2,
    visualize_samples= True,
    visualize_gt = False,
    visualize_model_estimatedCov = False,
    samples_alpha = 0.5,
    samples_label = 'Experimental data',
    sigma_lw = 0.5,
    sigma_alpha = 1,
    modelpred_alpha = 1,
    modelpred_lw = 0.5,
    modelpred_lc = 'k',
    modelpred_ls = '-') 
ax2.set_xlabel('Wishart space dimension 1');
ax2.set_ylabel('Wishart space dimension 2');
ax2.set_title('Individual fits');
fig2.savefig(output_figDir_fits+f"/{file_name_btst[:-10]}_btstCI_byIndvFits_wSamples.pdf",
             format='pdf', bbox_inches='tight')  



