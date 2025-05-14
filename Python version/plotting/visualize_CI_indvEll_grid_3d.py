#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  9 17:30:48 2025

@author: fh862-adm



"""

import dill as pickled
from tqdm import trange
import sys
import os
import numpy as np
from dataclasses import replace
#sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/Python version/')
sys.path.append("/Users/fh862-adm/Documents/GitHub/ColorEllipsoids/Python version")
from analysis.utils_load import select_file_and_get_path
#sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids')
sys.path.append("/Users/fh862-adm/Documents/GitHub/ellipsoids/ellipsoids")
from plotting.wishart_predictions_plotting import WishartPredictionsVisualization, Plot3DPredSettings
from plotting.wishart_plotting import PlotSettingsBase 
from analysis.conf_interval import find_inner_outer_contours_for_gridRefs
from analysis.ellipsoids_tools import slice_ellipsoid_byPlane, eig_to_covMat
from analysis.ellipses_tools import fit_2d_isothreshold_contour, convert_2Dcov_to_points_on_ellipse

#specify the file name
#base_dir = '/Volumes/T9/Aguirre-Brainard Lab Dropbox/Fangfang Hong/ELPS_analysis/'
baseDir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'
COLOR_DIMENSION = 3

#%%
#---------------------------------------------------------------------------
# SECTION 1: load the model fits to the empirical data
# --------------------------------------------------------------------------
# Select the file containing the model fits
# 'META_analysis/ModelFitting_DataFiles/6dTask/CIE'
# 'Fitted_byWishart_ellipsoids_6DExpt_1500_1500_1500_25500_AEPsychSampling_EAVC_decayRate0.4_nBasisDeg5_sub1.pkl'
input_fileDir_fits, file_name = select_file_and_get_path()

# Construct the full path to the selected file
full_path = os.path.join(input_fileDir_fits, file_name)

# extract the subset size from the file name
start_idx = file_name.find('subset') + len('subset')
end_idx = file_name.find('.pkl')
size_subset = int(file_name[start_idx:end_idx])

# Load the necessary variables from the file
with open(full_path, 'rb') as f:
    vars_dict = pickled.load(f)

# - Transformation matrices for converting between DKL, RGB, and W spaces
color_thres_data = vars_dict['color_thres_data']

# - Dimensionality of the color space (e.g., 2D for isoluminant planes)
ndims = color_thres_data.color_dimension

# - Ground truth
gt_Wishart = vars_dict['gt_Wishart']
grid = vars_dict['grid']
grid_trans = np.transpose(grid,(1,0,2,3))
num_grid_pts = grid.shape[0]
expt_trial = vars_dict['data_AEPsych_subset_flatten']

# Build key names for the variables based on the desired grid size
key_indv_model_btst = f'model_pred_indv_subset{size_subset}_btst_AEPsych'
model_pred = vars_dict['model_pred_existing']

#%% 
# Set up parameters
nTheta = 1000       # Number of samples for parametric operations (e.g., contour plotting)
nDatasets = 10      # Number of bootstrap datasets to load

# Initialize arrays to store ellipsoid parameters
# Each ellipsoid is characterized by:
#   - 3 radii (lengths of principal axes)
#   - 3x3 eigenvector matrix (orientation in space)
#   - center (location in 3D color space)
grid_shape = tuple([num_grid_pts] * ndims)
base_shape = (nDatasets,) + grid_shape
radii_ell  = np.full(base_shape + (COLOR_DIMENSION,), np.nan)                   # shape: (nDatasets, grid_x, grid_y, grid_z, 3)
evecs_ell  = np.full(base_shape + (COLOR_DIMENSION, COLOR_DIMENSION), np.nan)   # shape: (nDatasets, grid_x, grid_y, grid_z, 3, 3)
center_ell = np.full(base_shape + (COLOR_DIMENSION,), np.nan)                   # shape: (nDatasets, grid_x, grid_y, grid_z, 3)

# To facilitate visualization, we will compute 2D slices through each 3D ellipsoid.
# Specifically, we slice each ellipsoid along three orthogonal planes:
#   - GB plane (y-z plane)
#   - RB plane (x-z plane)
#   - RG plane (x-y plane)
v_plane_slicer = np.array([
    [[0, 1, 0], [0, 0, 1]],  # y-z plane: basis vectors along y and z
    [[1, 0, 0], [0, 0, 1]],  # x-z plane: basis vectors along x and z
    [[1, 0, 0], [0, 1, 0]]   # x-y plane: basis vectors along x and y
])

# For these 3 planes, specify which coordinate indices vary (to label axes later)
varying_planes = np.array([[1, 2], [0, 2], [0, 1]])  # (y,z), (x,z), (x,y)

# Initialize arrays to store:
#   - `params_ell`: ellipse parameters (5 parameters per ellipse: center_x, center_y, semi_axis_x, semi_axis_y, rotation_angle) 
#                   for each planar slice
#   - `covMat_ell`: covariance matrix (3x3) for each ellipsoid
params_ell = np.full(base_shape + (5, v_plane_slicer.shape[0]), np.nan)        # shape: (nDatasets, grid_x, grid_y, grid_z, 5, 3 planes)
covMat_ell = np.full(base_shape + (COLOR_DIMENSION, COLOR_DIMENSION), np.nan)  # shape: (nDatasets, grid_x, grid_y, grid_z, 3, 3)

#%Loop through each bootstrap dataset and load data
for r in trange(nDatasets):
    # extract variables
    vars_dict_r = vars_dict[f'{key_indv_model_btst}[{r}]']
    param_ell_r_temp = vars_dict_r['fitEll_params']
    param_ell_r = np.transpose(np.reshape(param_ell_r_temp, grid_shape), (1,0,2))
    #param_ell_r   = model_pred_indv_r.params_ell
    #covMat_ell[r] = model_pred_indv_r.pred_covMat
                
    for idx in np.ndindex(grid_shape):
        i,j,k = idx
        param_ell_dict_r   = param_ell_r[i][j][k]
        radii_ell[r,*idx]  = param_ell_dict_r['radii'].ravel()
        evecs_ell[r,*idx]  = param_ell_dict_r['evecs']
        center_ell[r,*idx] = param_ell_dict_r['center'].ravel()
        covMat_ell[r,*idx] = eig_to_covMat(radii_ell[r,*idx]**2, evecs_ell[r,*idx])
        
        for l in range(COLOR_DIMENSION):       
            try:
                sliced_ellipse_ijkl, _ = slice_ellipsoid_byPlane(center_ell[r,*idx],
                                                                 radii_ell[r,*idx],
                                                                 evecs_ell[r,*idx],
                                                                 *v_plane_slicer[l])
                #fit an ellipse to the sliced ellipse
                _, _, _, params_ell[r,*idx,:,l] = fit_2d_isothreshold_contour(\
                    center_ell[r, *idx, varying_planes[l]], [], comp_RGB = \
                        sliced_ellipse_ijkl[varying_planes[l]])
            except:
                print(f'r: {r}; idx: {idx}, l: {l}')
        
#%%           
# -----------------------------------------------------------------------------
# SECTION 3: Derive the Confidence Interval of Bootstrapped Model Fits
# -----------------------------------------------------------------------------
#  `params_ell` shape: (nDatasets, grid_x, grid_y, grid_z, 5 ellipse params, 3 plane slices)
# We want to rearrange axes to:
#   (grid_x, grid_y, grid_z, 3 plane slices, nDatasets, 5 ellipse params)
# This makes it compatible with the CI computation function, 
# so that confidence intervals are computed across bootstrapped datasets.
params_ell_trans = np.transpose(params_ell, (1,2,3,5,0,4))

# Compute confidence intervals (inner and outer ellipse contours) across bootstraps
# `ndims = 4` → 3 spatial dimensions + 1 for the plane slice index
fitEll_min_slice, fitEll_max_slice = find_inner_outer_contours_for_gridRefs(params_ell_trans, 
                                                                ndims = COLOR_DIMENSION + 1) 
# Format results for plotting: add new axis for CI lower and upper bounds
modelpred_slice_CI = np.concatenate((fitEll_min_slice[np.newaxis], fitEll_max_slice[np.newaxis]))

# -------------------------------------------------------------------------
# Next, compute confidence intervals for the **projected ellipsoids**:
# Each ellipsoid projection gives a 2D covariance matrix → converted to 2D ellipse points → 
# fit an ellipse to those points to obtain ellipse parameters.
# -------------------------------------------------------------------------
modelpred_proj_ell = np.full(base_shape + (COLOR_DIMENSION, 5), np.nan)
for p in range(nDatasets):
    for idx in np.ndindex(grid_shape):
        for q in range(COLOR_DIMENSION):
            try:
                cov_ijk = covMat_ell[p,*idx][varying_planes[q]][:,varying_planes[q]]
                centers = grid_trans[*idx][varying_planes[q]]
                
                x_val, y_val = convert_2Dcov_to_points_on_ellipse(cov_ijk,
                    ref_x = centers[0], ref_y = centers[1])
                pts_on_ell = np.vstack((x_val, y_val))
                
                _, _, _, modelpred_proj_ell[p,*idx,q] = fit_2d_isothreshold_contour(\
                    centers, [], comp_RGB = pts_on_ell)
            except:
                print(f'{p}: p; idx: {idx}, q: {q}')

# now we repeat the calculations we did for the sliced elliposoids
modelpred_proj_ell_trans = np.transpose(modelpred_proj_ell, (1,2,3,4,0,5))
fitEll_min_proj, fitEll_max_proj = find_inner_outer_contours_for_gridRefs(modelpred_proj_ell_trans, 
                                                                ndims = COLOR_DIMENSION + 1)
modelpred_proj_CI = np.concatenate((fitEll_min_proj[np.newaxis], fitEll_max_proj[np.newaxis]))

#%% Create the output directory if it doesn't exist
output_figDir_fits = input_fileDir_fits.replace('DataFiles', 'FigFiles')
os.makedirs(output_figDir_fits, exist_ok=True)

# Create a base plotting settings instance (shared across plots)
pltSettings_base = PlotSettingsBase(fig_dir=output_figDir_fits, fontsize = 11)

# Initialize 2D prediction settings by copying from base and overriding method-specific parameters
predM_settings = replace(Plot3DPredSettings(), **pltSettings_base.__dict__)
predM_settings = replace(predM_settings,
                         visualize_samples = False,
                         samples_s = 1,
                         samples_alpha = 0.2, 
                         gt_ls = '--',
                         gt_lw = 1,
                         gt_lc = 'k',
                         gt_alpha = 0.85,
                         modelpred_alpha = 0.55,
                         gt_3Dproj_ls = '--',
                         gt_3Dproj_lc = 'k',
                         gt_3Dproj_lw = 1,
                         modelpred_lc = None,
                         visualize_model_pred = False,
                         visualize_modelpred_CI = True,
                         modelpred_CI_alpha = 0.9,
                         modelpred_projection_CI = modelpred_proj_CI,
                         modelpred_slice_CI = modelpred_slice_CI,
                         fig_name = f"{file_name[:-4]}_wBtstCI") 
        
# Initialize Visualization Class for Wishart Predictions
wishart_pred_vis_wCI = WishartPredictionsVisualization(expt_trial,
                                                       model_pred.model, 
                                                       model_pred, 
                                                       color_thres_data,
                                                       settings = pltSettings_base,
                                                       save_fig = False)
# Create figure and axes for plotting
wishart_pred_vis_wCI.plot_3D(grid_trans,  
                             gt_Wishart.pred_covMat, 
                             gt_Wishart.pred_slice_2d_ellipse,
                             settings = predM_settings)

