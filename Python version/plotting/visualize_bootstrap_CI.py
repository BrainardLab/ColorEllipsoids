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
from plotting.visualize_MOCS import plot_MOCS_conditions, MOCSTrialsVisualization
sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids')
from plotting.wishart_predictions_plotting import WishartPredictionsVisualization
from analysis.ellipses_tools import find_inner_outer_contours
from plotting.wishart_plotting import PlotSettingsBase 
from plotting.wishart_predictions_plotting import Plot2DPredSettings
from core.model_predictions import wishart_model_pred

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
model = model_pred.model

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

#%% load best-fit Weibull Psychometric functions
# Step 1: Select the file : ELPS_analysis/Experiment_DataFiles/sub#
#'Fitted_weibull_psychometric_func_Isoluminant plane_6000totalTrials_25refs_MOCS_sub#.pkl'
input_fileDir_fits_MOCS, file_name_MOCS = select_file_and_get_path()

# Construct the full path to the selected file
full_path_MOCS = os.path.join(input_fileDir_fits_MOCS, file_name_MOCS)

# Load the necessary variables from the file
with open(full_path_MOCS, 'rb') as f:
    vars_dict_MOCS = pickled.load(f)
    
stim_at_targetPC_MOCS = vars_dict_MOCS['stim_at_targetPC_MOCS']
xref_unique_MOCS = vars_dict_MOCS['xref_unique']
xref_unique_expt_MOCS = vars_dict_MOCS['xref_unique_exp']
xref_unique_trans_MOCS = vars_dict_MOCS['xref_unique_trans']
model_pred_Wishart_MOCS = vars_dict_MOCS['model_pred_Wishart_MOCS']
fit_PMF_MOCS = vars_dict_MOCS['fit_PMF_MOCS']
vecLen_at_targetPC_Wishart = vars_dict_MOCS['vecLen_at_targetPC_Wishart']
slope_modelPred_empData_mean = vars_dict_MOCS['slope_modelPred_empData_mean']
slope_modelPred_empData_CI = vars_dict_MOCS['slope_modelPred_empData_CI']
                               
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
params_all = np.full((25, nBtst, 5), np.nan)

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
    model_pred_r = vars_dict_btst['model_pred_Wishart']
    param_ell_r = model_pred_r.params_ell
    W_est_r = model_pred_r.W_est
    Sigmas_est_xref_unique = model.compute_Sigmas(model.compute_U(W_est_r, xref_unique_expt_MOCS))

    # Initialize the Wishart model prediction using various parameters.
    model_pred_Wishart_MOCS = wishart_model_pred(model_pred_r.model, model_pred_r.opt_params,
                                                 model_pred_r.w_init_key,
                                                 model_pred_r.opt_key,
                                                 model_pred_r.W_init,
                                                 W_est_r, Sigmas_est_xref_unique,
                                                 color_thres_data, 
                                                 target_pC= vars_dict_btst['target_pC'],
                                                 ngrid_bruteforce = 1000,
                                                 bds_bruteforce = [0.0005, 0.25])
    # batch compute 66.7% threshold contour based on estimated weight matrix
    model_pred_Wishart_MOCS.convert_Sig_Threshold_oddity_batch(xref_unique_trans_MOCS)
    
    for j in range(25):
        params_all[j,r] = model_pred_Wishart_MOCS.params_ell[0][j]
        
#%%           
# -----------------------------------------------------------------------------
# SECTION 3: Derive the Confidence Interval of Bootstrapped Model Fits
# -----------------------------------------------------------------------------
def distance_to_ellipse_boundary(a, b, theta_deg, dx, dy):
    theta_rad = np.deg2rad(theta_deg)
    A = np.cos(theta_rad)**2 / a**2 + np.sin(theta_rad)**2 / b**2
    B = 2*(1/a**2 - 1/b**2) * np.cos(theta_rad) * np.sin(theta_rad)
    C = np.sin(theta_rad)**2 / a**2 + np.cos(theta_rad)**2 / b**2
    
    r = 1/np.sqrt(A * dx**2 + B * dx * dy + C * dy**2)
    return r

def find_CI_for_chromDir(p_ell):
    ng1, ng2, ng3 = p_ell.shape[0:3]
    opt_vec = np.full((ng1, ng2, ng3), np.nan)
    for i in range(ng1):
        for j in range(ng2):
            chromDir = fit_PMF_MOCS[i*ng1+j].unique_stim[1]
            chromDir_norm = chromDir / np.linalg.norm(chromDir)
            for k in range(ng3):
                params_ijk = p_ell[i,j,k]
                opt_vec[i,j,k] = distance_to_ellipse_boundary(*params_ijk[2:], *chromDir_norm)
    return opt_vec

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
    ng1, ng2 = p_ell.shape[0:2]

    # Initialize arrays to store the minimum (inner) and maximum (outer) contours
    # Shape: (grid_size, grid_size, dimensions, resolution)
    ell_min = np.full((ng1, ng2, nd, nTheta), np.nan)
    ell_max = np.full((ng1, ng2, nd, nTheta), np.nan)

    # Iterate over all grid points
    for i in range(ng1):
        for j in range(ng2):
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
params_all_ext = params_all[np.newaxis]
fitEll_min, fitEll_max = find_CI_for_gridRefs(params_all_ext)

opt_vecLen_Wishart = np.sort(find_CI_for_chromDir(params_all_ext)[0], axis = -1)
vecLen_at_targetPC_Wishart_lb = opt_vecLen_Wishart[:,0]
vecLen_at_targetPC_Wishart_ub = opt_vecLen_Wishart[:,-1]
vecLen_at_targetPC_Wishart_CI = np.vstack((vecLen_at_targetPC_Wishart - vecLen_at_targetPC_Wishart_lb,
                                          vecLen_at_targetPC_Wishart_ub - vecLen_at_targetPC_Wishart)).T

vecLen_at_targetPC_Wishart_CI = np.clip(vecLen_at_targetPC_Wishart_CI, a_min=1e-8, a_max=None)

#%%
# -----------------------------------------------------------------------------
# SECTION 4: Visualizse the CI
# -----------------------------------------------------------------------------
def plot_btst_CI(ref, ell_min, ell_max, ax):
    ng1, ng2 = ref.shape[0:2]
    for i in range(ng1):
        for j in range(ng2):
            if i == 0 and j == 0: lbl = f'Bootstrapped CI ({nBtst} AEPsych datasets)'
            else: lbl = None
            # Adjust the color map based on the fixed color dimension.
            cm = color_thres_data.M_2DWToRGB @ np.insert(ref[i, j], 2, 1)
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
                            modelpred_alpha = 1,
                            modelpred_lw = 0.2,
                            modelpred_lc = 'k',
                            modelpred_ls = '-',
                            ticks = np.linspace(-0.8,0.8,5),
                            legend_off = True,
                            flag_rescale_axes_label = False)

#threshold contours predicted by the Wishart
wishart_pred_vis_MOCS = WishartPredictionsVisualization(expt_trial,
                                                        model, 
                                                        model_pred_Wishart_MOCS, 
                                                        color_thres_data,
                                                        settings = pltSettings_base,
                                                        save_fig = False)

#customize cmap for the isoluminant plane
fig, ax = plt.subplots(1, 1, figsize = pred2D_settings.fig_size, dpi= pred2D_settings.dpi)
# Plot the bootstrap confidence intervals on top of the grid
plot_btst_CI(xref_unique_MOCS[np.newaxis], fitEll_min, fitEll_max, ax)

wishart_pred_vis_MOCS.plot_2D(xref_unique_expt_MOCS, 
                              xref_unique_expt_MOCS,
                              ax = ax, 
                              settings = pred2D_settings)
plot_MOCS_conditions(2, xref_unique_MOCS, stim_at_targetPC_MOCS[:,np.newaxis,:],
                     color_thres_data, 
                     ax = ax, ref_ms = 5, ref_lw = 1,ticks = np.linspace(-0.8,0.8,5),
                     easyTrials_highlight = False)
ax.set_title('Isoluminant plane');
ax.set_xlim([-1,1]); ax.set_ylim([-1,1])
# Save the figure as a PDF
fig_name = f"{file_name[:-4]}_comparison_btw_MOCS_WishartPredictions_wBtstCI.pdf"
fig.savefig(os.path.join(output_figDir_fits, fig_name), bbox_inches='tight')    
plt.show()

#%%
#visualization object
vis_MOCS = MOCSTrialsVisualization(fit_PMF_MOCS, 
                                   fig_dir=output_figDir_fits,
                                   save_fig= True)

#initialize color map
cmap_allref = []
for n in range(len(fit_PMF_MOCS)):    
    #define color map for each reference
    cmap_n = color_thres_data.M_2DWToRGB @ np.append(fit_PMF_MOCS[n].unique_stim[-1] +\
                                                     xref_unique_MOCS[n], 1)        
    #append colormap so we can reuse it for the next plot
    cmap_allref.append(cmap_n)
    
# plot the comparison of thresholds between AEPsych predictions and MOCS predictions
vis_MOCS.plot_comparison_thres(thres_Wishart = vecLen_at_targetPC_Wishart,
                               slope_mean = slope_modelPred_empData_mean,
                               slope_CI= slope_modelPred_empData_CI,
                               xref_unique = xref_unique_MOCS,
                               thres_Wishart_CI = vecLen_at_targetPC_Wishart_CI,
                               bds = np.array([0, 0.1]), #np.array([0, 0.2]),
                               ms = 6,
                               fig_size = (5.5, 5.7),
                               slope_text_loc = (0.017, 0.109),
                               alpha = 0.8,
                               lw = 1.5,
                               cmap = cmap_allref,
                               fig_name = f"{fig_name[:-4]}_v2.pdf")









