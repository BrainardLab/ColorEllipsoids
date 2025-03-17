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
import dill as pickled
sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/Python version")
from analysis.simulations_CIELab import SimThresCIELab
from plotting.sim_CIELab_plotting import CIELabVisualization
sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids")
from analysis.ellipses_tools import fit_2d_isothreshold_contour
from analysis.color_thres import color_thresholds

base_dir = '/Volumes/T9/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'
output_figDir = os.path.join(base_dir, 'ELPS_analysis','Simulation_FigFiles','Python_version','CIE')
output_fileDir = os.path.join(base_dir, 'ELPS_analysis','Simulation_DataFiles')

#%% **LOAD NECESSARY DATA**
color_thres_data = color_thresholds(2, os.path.join(base_dir, 'ELPS_analysis'))

# Path to transformation matrices and reference points
path_str = "/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/FilesFromPsychtoolbox/"
os.chdir(path_str)

# Initialize simulation object for CIELab calculations
background_RGB = np.array([0.5, 0.5, 0.5])  # Mid-gray background
sim_thres_CIELab = SimThresCIELab(path_str, 
                                  background_RGB, 
                                  plane_2D_list= ['Isoluminant plane'])

# Load color space transformation matrices
mat_file   = loadmat('Transformation_btw_color_spaces.mat')
iso_mat    = mat_file['DELL_02242025_texture_right'][0]
gamut_rgb  = iso_mat['gamut_bg_primary'][0]
M_RGBTo2DW = iso_mat['M_RGBTo2DW'][0]  # Transform RGB to 2D W space
M_2DWToRGB = iso_mat['M_2DWToRGB'][0]  # Transform W space back to RGB

# Load reference stimulus locations and reshape
ref_points_temp = np.transpose(iso_mat['ref_rgb'][0], (1, 0))
nGridPts_ref    = int(np.sqrt(ref_points_temp.shape[0]))  # Grid size
rgb_ref_points  = np.transpose(np.reshape(ref_points_temp, (nGridPts_ref, nGridPts_ref, -1)), 
                               (2, 1, 0))

# **DEFINE SIMULATION PARAMETERS**
numDirPts = 16  # Number of chromatic directions (angles)
grid_theta_xy = sim_thres_CIELab.set_chromatic_directions(num_dir_pts=numDirPts)

deltaE_1JND = 1  # Just-noticeable difference threshold (ΔE = 1)
color_diff_algorithm = 'CIE2000'  # Choose from 'CIE2000', 'CIE1994', 'CIE1976'

# Parameters for ellipse fitting
contour_scaler = 2.5  # Scaling factor for visualization
nThetaEllipse = 200  # Number of ellipse points

#a filler entry 
append_W_val = 1

#%% **INITIALIZE ARRAYS FOR COMPUTATION**
base_size  = (nGridPts_ref, nGridPts_ref)
W_ref      = np.full(base_size + (3,), np.nan)
rgb_vecDir = np.full(base_size + (3, numDirPts,), np.nan)  # Direction vectors in RGB
opt_vecLen = np.full(base_size + (numDirPts,), np.nan)  # Vector lengths for ΔE=1

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
1. **Three spaces** are involved: RGB space (3D) and W-space (2D square).
2. **Threshold points** are computed in RGB space (where CIELab operates).
3. **Transformations** are applied:
   - RGB → W using a homography matrix.
4. **Fit an ellipse** in Normalized space.
"""

for i in range(nGridPts_ref):
    for j in range(nGridPts_ref):
        # **Step 1: Convert Reference Stimulus**
        rgb_ref_pij = rgb_ref_points[:, i, j]  # RGB reference point
        W_ref_ij = M_RGBTo2DW @ rgb_ref_pij  # Transform to W space
        W_ref[i,j] = W_ref_ij

        # **Step 2: Compute Chromatic Directions & Thresholds**
        for k in range(numDirPts):
            # Step 2.1: Define a chromatic direction in W-space and convert to RGB
            chrom_dir_W = grid_theta_xy[:, k] + W_ref[i,j,:2]  # Shifted chromatic direction
            chrom_dir_rgb = M_2DWToRGB @ np.append(chrom_dir_W, append_W_val)  # Convert back to RGB

            # Step 2.2: Compute normalized direction vector in RGB space
            rgb_vecDir_k = chrom_dir_rgb - rgb_ref_pij  # Vector difference
            rgb_vecDir[i, j, :, k] = rgb_vecDir_k / np.linalg.norm(rgb_vecDir_k)  # Normalize

            # Step 2.3: Find vector length that produces ΔE = 1 in CIELab space
            opt_vecLen[i, j, k] = sim_thres_CIELab.find_vecLen(
                rgb_ref_pij, rgb_vecDir[i, j, :, k], deltaE_1JND, coloralg=color_diff_algorithm
            )

            # Step 2.4: Compute threshold point in RGB space
            rgb_comp_contour_unscaled[i, j,:, k] = opt_vecLen[i, j, k] * rgb_vecDir[i, j, :, k] + rgb_ref_pij

        # **Step 3: Transform Threshold Points from RGB → W → Normalized Space**
        W_comp_pij = M_RGBTo2DW @ rgb_comp_contour_unscaled[i, j]  # Convert to W-space
        W_comp_pij = W_comp_pij / W_comp_pij[-1]  # Normalize last row to 1
        W_comp_contour_unscaled[i,j] = W_comp_pij[:2]

        # **Step 4: Fit Ellipse**
        fit_results = fit_2d_isothreshold_contour(
            W_ref[i,j,:2], None, comp_RGB=W_comp_contour_unscaled[i,j],
            nThetaEllipse=nThetaEllipse, ellipse_scaler=contour_scaler
        )

        # Store ellipse fitting results (in N space)
        fitEllipse_scaled[i, j], fitEllipse_unscaled[i, j], W_comp_contour_scaled[i, j], _, ellParams[i, j] = fit_results


#%% PLOTTING 
# Define coarse grid bounds for the varying color dimension
grid_ref = np.linspace(-0.6, 0.6, nGridPts_ref)  # Lower bound: 0.2, Upper bound: 0.8

# Generate 2D grid for (GB, RB, RG) slices in RGB space
X, Y = np.meshgrid(grid_ref, grid_ref)
grid_est = np.dstack((X, Y))  # Shape: (nGridPts_ref, nGridPts_ref, 2)

# Compute colormap for background
cmap_W_hom = np.vstack((grid_est.reshape(-1, 2).T, np.ones((1, nGridPts_ref**2))))  # Homogeneous coordinates
cmap_ell_lc = (M_2DWToRGB @ cmap_W_hom).T.reshape(nGridPts_ref, nGridPts_ref, -1)  # Transform to RGB space

# Initialize visualization object
sim_CIE_vis = CIELabVisualization(SimThresCIELab(path_str, background_RGB), 
                                  fig_dir=output_figDir, save_fig=False)

# Plot 2D plane with computed ellipses and thresholds
sim_CIE_vis.plot_2D_single_plane(grid_est, 
                                 fitEllipse_scaled, 
                                 rawData=W_comp_contour_scaled, 
                                 rgb_background=None, 
                                 lim = [-1,1],
                                 ticks = np.linspace(-1,1,5),
                                 ell_lc=cmap_ell_lc)

#%% save the data
#convert fitted ellipses to RGB space for future uses (maybe we want to visualize that in the RGB space)
fitEllipse_scaled_RGB_temp1 = np.reshape(np.transpose(fitEllipse_scaled, (2, 0,1,3)), (2,-1))
fitEllipse_scaled_RGB_temp2 = np.vstack((fitEllipse_scaled_RGB_temp1, 
                                         np.ones((1, nGridPts_ref**2 * nThetaEllipse))))
fitEllipse_scaled_RGB_temp3 = M_2DWToRGB @ fitEllipse_scaled_RGB_temp2
fitEllipse_scaled_RGB = np.transpose(np.reshape(fitEllipse_scaled_RGB_temp3, 
                                                (3, nGridPts_ref, nGridPts_ref, nThetaEllipse)), (1,2,3,0))

#repeat the same for fitEllipse_unscaled
fitEllipse_unscaled_RGB_temp1 = np.reshape(np.transpose(fitEllipse_unscaled, (2, 0,1,3)), (2,-1))
fitEllipse_unscaled_RGB_temp2 = np.vstack((fitEllipse_unscaled_RGB_temp1, 
                                         np.ones((1, nGridPts_ref**2 * nThetaEllipse))))
fitEllipse_unscaled_RGB_temp3 = M_2DWToRGB @ fitEllipse_unscaled_RGB_temp2
fitEllipse_unscaled_RGB = np.transpose(np.reshape(fitEllipse_unscaled_RGB_temp3,
                                                  (3, nGridPts_ref, nGridPts_ref, nThetaEllipse)), (1,2,3,0))

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(*gamut_rgb,color ='k')
# Scatter plot
ax.scatter(*fitEllipse_scaled_RGB_temp3, cmap='viridis', marker='.', s=10, alpha=0.7, edgecolors='none')
ax.view_init(elev=30, azim=-120)
ax.set_xlim([0,1]); ax.set_ylim([0,1]); ax.set_zlim([0,1]); ax.set_aspect('equal')
ax.set_xlabel('R'); ax.set_ylabel('G'); ax.set_zlabel('B')
        
#%%
output_file = f'Isothreshold_ellipses_isoluminant_{color_diff_algorithm}.pkl'
full_path = os.path.join(output_fileDir, output_file)

#save all the stim info
fixed_RGBvec = append_W_val
ref_points = np.repeat(np.transpose(W_ref[np.newaxis,:,:,:], (0, 3, 1,2)), 3, axis = 0)
stim_keys = ['fixed_RGBvec', 'grid_ref', 'nGridPts_ref', 'ref_points', 
             'background_RGB','numDirPts', 'grid_theta_xy', 'deltaE_1JND',
             'cmap_ell_lc','color_diff_algorithm','contour_scaler', 'nThetaEllipse',
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
