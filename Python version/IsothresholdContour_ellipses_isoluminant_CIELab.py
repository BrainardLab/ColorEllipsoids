#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 22:29:40 2025

@author: fangfang

The goal of this script is to compute threshold contours on the isoluminant plane
based on either CIE1976 (CIELAB), CIE1994 or CIE2000

"""

import sys
import os
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import replace
import dill as pickled
sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/Python version")
from analysis.simulations_CIELab import SimThresCIELab
from plotting.sim_CIELab_plotting import CIELabVisualization, Plot2DSinglePlaneSettings
sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids")
from analysis.ellipses_tools import fit_2d_isothreshold_contour
from plotting.wishart_plotting import PlotSettingsBase 

base_dir = '/Volumes/T9/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'
output_figDir = os.path.join(base_dir, 'ELPS_analysis','Simulation_FigFiles','Python_version','CIE')
output_fileDir = os.path.join(base_dir, 'ELPS_analysis','Simulation_DataFiles')
pltSettings_base = PlotSettingsBase(fig_dir=output_figDir, fontsize = 12)

#%% LOAD TRANSFORMATION MATRICES AND INITIALIZE SIMULATION

# Mid-gray background in RGB
background_RGB = np.array([0.5, 0.5, 0.5])

# Initialize simulation object for CIELab-based threshold calculations
sim_thres_CIELab = SimThresCIELab(background_RGB, plane_2D_list=['Isoluminant plane'])

# Load transformation matrices between color spaces
mat_file = loadmat('Transformation_btw_color_spaces.mat')
iso_mat = mat_file['DELL_02242025_texture_right'][0]

gamut_rgb = iso_mat['gamut_bg_primary'][0]
M_RGBTo2DW = iso_mat['M_RGBTo2DW'][0]  # RGB → 2D Wishart space
M_2DWToRGB = iso_mat['M_2DWToRGB'][0]  # 2D Wishart space → RGB

#%% DEFINE GRID POINTS IN WISHART SPACE
# Create a 2D grid of reference locations in Wishart space
nGridPts_ref = 7
str_ext = f'_grid{nGridPts_ref}by{nGridPts_ref}' if nGridPts_ref != 5 else ''
grid_ref = np.linspace(-0.7, 0.7, nGridPts_ref)
ref_points_W = np.stack(np.meshgrid(grid_ref, grid_ref), axis=-1)  # Shape: (9, 9, 2)
ref_points_W_col = ref_points_W.reshape(-1, 2)  # Flattened to (81, 2)

# Append a constant dimension (value = 1) to project into RGB space
append_W_val = 1
ref_points_W_ext = np.hstack((ref_points_W_col, 
                              np.full((ref_points_W_col.shape[0], 1), append_W_val)))

# Transform grid points from W space → RGB
rgb_ref_points = (M_2DWToRGB @ ref_points_W_ext.T).reshape(-1, nGridPts_ref, nGridPts_ref)

#%% DEFINE SIMULATION PARAMETERS

# Set number of chromatic directions (angles) to simulate
numDirPts = 16
grid_theta_xy = sim_thres_CIELab.set_chromatic_directions(num_dir_pts=numDirPts)

# Set JND threshold and color difference algorithm
deltaE_1JND = 2.5  # Target ΔE for threshold contour
color_diff_algorithm = 'CIE1994'  # Options: 'CIE2000', 'CIE1994', 'CIE1976'

# Ellipse fitting resolution
nThetaEllipse = 200  # Number of angles used to trace the threshold ellipse

#OPTIONAL: there is an option to scale the elliptical threshold contour up/down
scaler = 1

# INITIALIZE ARRAYS FOR COMPUTATION
W_ref      = np.reshape(ref_points_W_ext, (nGridPts_ref, nGridPts_ref, -1))
base_size  = (nGridPts_ref, nGridPts_ref)
rgb_vecDir = np.full(base_size + (3, numDirPts,), np.nan)  # Direction vectors in RGB
opt_vecLen = np.full(base_size + (numDirPts,), np.nan)     # Vector lengths for ΔE=1

# Ellipse fitting results
W_comp_contour_scaled     = np.full(base_size + (2, numDirPts,), np.nan)
W_comp_contour_unscaled   = np.full(base_size + (2, numDirPts,), np.nan)
fitEllipse_scaled         = np.full(base_size + (2, nThetaEllipse,), np.nan)
fitEllipse_unscaled       = np.full(base_size + (2, nThetaEllipse,), np.nan)
rgb_comp_contour_unscaled = np.full(base_size + (3, numDirPts,), np.nan)  # Threshold points in RGB
ellParams                 = np.full(base_size + (5,), np.nan)  # 5 free parameters for ellipse fitting

#%% **MAIN COMPUTATION LOOP**
"""
The logic:
1. **Two spaces** are involved: RGB space (3D) and W-space (2D square).
2. **Threshold points** are computed in RGB space (where CIELab operates).
3. **Transformations** are applied:
   - RGB → W using a homography matrix.
4. **Fit an ellipse** in the Wishart space.
"""

for i in range(nGridPts_ref):
    for j in range(nGridPts_ref):
        # retrieve reference in W space
        W_ref_ij = W_ref[i,j]
        
        #for each chromatic direction, derive the threshold
        for k in range(numDirPts):
            rgb_vecDir[i, j, :, k], opt_vecLen[i, j, k], \
            rgb_comp_contour_unscaled[i, j,:, k], W_comp_contour_unscaled[i,j,:,k] = \
                sim_thres_CIELab.find_threshold_point_on_isoluminant_plane(\
                                W_ref_ij, grid_theta_xy[:, k], 
                                M_RGBTo2DW, M_2DWToRGB, 
                                deltaE = deltaE_1JND,
                                coloralg = color_diff_algorithm)
        # Fit Ellipse**
        fit_results = fit_2d_isothreshold_contour(
            W_ref[i,j,:2], None, comp_RGB=W_comp_contour_unscaled[i,j],
            nThetaEllipse = nThetaEllipse, ellipse_scaler = scaler)

        # Store ellipse fitting results (in N space)
        fitEllipse_scaled[i, j], fitEllipse_unscaled[i, j],\
            W_comp_contour_scaled[i, j], ellParams[i, j] = fit_results


#%% PLOTTING 
# Compute colormap for background
cmap_W_hom = np.vstack((ref_points_W.reshape(-1, 2).T, np.ones((1, nGridPts_ref**2))))  # Homogeneous coordinates
cmap_ell_lc = (M_2DWToRGB @ cmap_W_hom).T.reshape(nGridPts_ref, nGridPts_ref, -1)  # Transform to RGB space

# Initialize visualization object
sim_CIE_vis = CIELabVisualization(sim_thres_CIELab, 
                                  settings = pltSettings_base,
                                  save_fig=False)

pred2D_settings = replace(Plot2DSinglePlaneSettings(), **pltSettings_base.__dict__)
pred2D_settings = replace(pred2D_settings, 
                          rgb_background=None, 
                          lim = [-1,1],
                          ticks = np.linspace(-0.7,0.7,3),
                          ell_lc=cmap_ell_lc)
# Plot 2D plane with computed ellipses and thresholds
sim_CIE_vis.plot_2D_single_plane(ref_points_W, fitEllipse_scaled, rawData=W_comp_contour_scaled,
                                 settings = pred2D_settings)

#%% 
#convert fitted ellipses to RGB space for future uses 
#(maybe we want to visualize that in the RGB space)
base_shape = (3, nGridPts_ref, nGridPts_ref, nThetaEllipse)
ones_append = np.full((1, nGridPts_ref**2 * nThetaEllipse), append_W_val)
fE_s1 = np.reshape(np.transpose(fitEllipse_scaled, (2, 0,1,3)), (2,-1))
fE_s2 = np.vstack((fE_s1, ones_append))
fE_s3 = M_2DWToRGB @ fE_s2
fitEllipse_scaled_RGB = np.transpose(np.reshape(fE_s3, base_shape), (1,2,3,0))

#repeat the same for fitEllipse_unscaled
fE_us1 = np.reshape(np.transpose(fitEllipse_unscaled, (2, 0,1,3)), (2,-1))
fE_us2 = np.vstack((fE_us1, ones_append))
fE_us3 = M_2DWToRGB @ fE_us2
fitEllipse_unscaled_RGB = np.transpose(np.reshape(fE_us3, base_shape), (1,2,3,0))

# visualize the isoluminant slice in the RGB cube
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(*gamut_rgb,color ='k')
# Scatter plot
ax.scatter(*fE_us3, cmap='viridis', marker='.', s=10, alpha=0.7, edgecolors='none')
ax.view_init(elev=30, azim=-120)
ax.set_xlim([0,1]); ax.set_ylim([0,1]); ax.set_zlim([0,1]); ax.set_aspect('equal')
ax.set_xlabel('R'); ax.set_ylabel('G'); ax.set_zlabel('B')
        
#%%
output_file = f'Isothreshold_ellipses_isoluminant_{color_diff_algorithm}{str_ext}.pkl'
full_path = os.path.join(output_fileDir, output_file)

#save all the stim info
fixed_RGBvec = append_W_val
ref_points = np.repeat(np.transpose(W_ref[np.newaxis,:,:,:], (0, 3, 1,2)), 3, axis = 0)
stim_keys = ['fixed_RGBvec', 'grid_ref', 'nGridPts_ref', 'ref_points', 
             'background_RGB','numDirPts', 'grid_theta_xy', 'deltaE_1JND',
             'cmap_ell_lc','color_diff_algorithm', 'nThetaEllipse',
             'mat_file','iso_mat','gamut_rgb','M_RGBTo2DW', 'M_2DWToRGB']
stim = {}
for i in stim_keys: stim[i] = eval(i)

#save the results
results_keys = ['opt_vecLen', 'fitEllipse_scaled', 'fitEllipse_unscaled',
                'rgb_comp_contour_unscaled', 'ellParams', 'fitEllipse_scaled_RGB', 
                'fitEllipse_unscaled_RGB', 'W_comp_contour_scaled',
                'W_comp_contour_unscaled']
# Loop over each variable and apply np.newaxis + np.repeat
num_repeats = 3
for var_name in results_keys:
    globals()[var_name] = np.repeat(globals()[var_name][np.newaxis, ...], num_repeats, axis=0)
    
results = {}
for i in results_keys: results[i] = eval(i)
    
# Write the list of dictionaries to a file using pickle
with open(full_path, 'wb') as f:
    pickled.dump([sim_thres_CIELab, stim, results], f)
