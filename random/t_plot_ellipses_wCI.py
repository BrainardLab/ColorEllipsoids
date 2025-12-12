#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 09:45:08 2025

@author: fangfang

This script visualize bootstrap confidence intervals (CI) of Wishart model 
    predictions based on a single subject's dataset:
    - Solid line contours: Model fits to the original dataset.
    - Shaded regions: Model fits across 10 bootstrap-resampled datasets.

"""

import dill as pickled
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
script_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from analysis.conf_interval import find_inner_outer_contours_for_gridRefs

#%%
#---------------------------------------------------------------------------
# SECTION 1: load the model fits
# --------------------------------------------------------------------------
# define path
input_fileDir_fits = '/Volumes/T9/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
    'ELPS_analysis/Experiment_DataFiles/pilot2/sub1/fits'

file_name = 'Fitted_ColorDiscrimination_4dExpt_Isoluminant plane_sub1_decayRate0.5_nBasisDeg5.pkl'

# Construct the full path to the selected file
full_path = os.path.join(input_fileDir_fits, file_name)

# Load the necessary variables from the file
with open(full_path, 'rb') as f:
    vars_dict = pickled.load(f)

# Load needed variables
color_thres_data = vars_dict['color_thres_data'] 
model_pred = vars_dict['model_pred_Wishart']     #WPPM model predictions
grid = vars_dict['grid']                         #grid of reference colors, size: (7, 7, 2)
fitEll = model_pred.fitEll_unscaled              #ellipses, size: (7, 7, 2, 200)

#%% 
#---------------------------------------------------------------------------
# SECTION 2: load bootstrapped dataset
# --------------------------------------------------------------------------
input_fileDir_fits_btst = '/Volumes/T9/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
    'ELPS_analysis/Experiment_DataFiles/pilot2/sub1/fits/AEPsych_btst/decayRate0.5'

file_name_btst = f'{file_name[:-4]}_btst_AEPsych[0].pkl'

# 10 bootstrapped datasets
nDatasets = 10 

# Step 2: Initialize storage arrays
# - `params_ell`: Stores ellipse parameters for all grid points across bootstrapped datasets
#   - Ellipse parameters: [x-center, y-center, major axis length, minor axis length, rotation angle]
grid_shape = grid.shape[:2]
params_ell_shape = grid_shape + (nDatasets, 5)
params_ell = np.full(params_ell_shape, np.nan)

# Loop through each bootstrap dataset and load data
for r in trange(nDatasets):
    file_name_r = file_name_btst.replace('AEPsych[0]', f'AEPsych[{r}]')
    
    # Generate the file name for the current bootstrap dataset
    full_path_others_r = f"{input_fileDir_fits_btst}/{file_name_r}"
    
    # Load the variables from the current bootstrap dataset
    with open(full_path_others_r, 'rb') as f:
        vars_dict_others = pickled.load(f)
    
    # Use precomputed results if available
    model_pred_r = vars_dict_others['model_pred_Wishart']
    param_ell_r = model_pred_r.params_ell
                
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            params_ell[i, j, r] = param_ell_r[i][j]
            
#Computes the confidence intervals for the model-predicted ellipses at each grid point.
fitEll_min, fitEll_max = find_inner_outer_contours_for_gridRefs(params_ell)

#%%
#---------------------------------------------------------------------------
# SECTION 3: plot the confidence interval
# --------------------------------------------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(4,4), dpi=1024)
for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        #we need the rgb to color code the ellipse
        cmap_ij = color_thres_data.M_2DWToRGB @ np.append(grid[i,j], 1)
        
        # Identify valid (non-NaN) points in the outer contour
        idx_max_nonan = ~np.isnan(fitEll_max[i,j,0])
        # Fill the outer contour region with solid color to indicate the CI boundary
        ax.fill(fitEll_max[i,j, 0, idx_max_nonan], fitEll_max[i,j,1, idx_max_nonan], color=cmap_ij,
                alpha= 0.8, lw=0)
    
        # Identify valid (non-NaN) points in the inner contour
        idx_min_nonan = ~np.isnan(fitEll_min[i,j,0])
        # Fill the inner contour with white to "punch out" the inside, leaving a ring
        ax.fill(fitEll_min[i,j,0, idx_min_nonan], fitEll_min[i,j,1, idx_min_nonan], color='white', lw=0)
        
        #ax.scatter(*grid[i,j], color = cmap_ij)
        ax.plot(*fitEll[i,j], color = cmap_ij)
        
ax.set_xlabel('Model space dimension 1')
ax.set_ylabel('Model space dimension 2')
ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_xticks(np.linspace(-0.7, 0.7, 5))
ax.set_yticks(np.linspace(-0.7, 0.7, 5))
ax.set_title('Isoluminant plane')
ax.set_aspect('equal')
ax.grid(True, alpha = 0.1)
plt.show()

