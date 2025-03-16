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
nGridPts_ref = int(np.sqrt(ref_points_temp.shape[0]))  # Grid size
ref_points = np.transpose(np.reshape(ref_points_temp, (nGridPts_ref, nGridPts_ref, -1)), (2, 1, 0))

# **DEFINE SIMULATION PARAMETERS**
numDirPts = 16  # Number of chromatic directions (angles)
grid_theta_xy = sim_thres_CIELab.set_chromatic_directions(num_dir_pts=numDirPts)

deltaE_1JND = 1  # Just-noticeable difference threshold (ΔE = 1)
color_diff_algorithm = 'CIE2000'  # Choose from 'CIE2000', 'CIE1994', 'CIE1976'

# Parameters for ellipse fitting
contour_scaler = 2.5  # Scaling factor for visualization
nThetaEllipse = 200  # Number of ellipse points

#%% **INITIALIZE ARRAYS FOR COMPUTATION**
ssize      = (nGridPts_ref, nGridPts_ref)
rgb_comp   = np.full(ssize + (numDirPts, 3), np.nan)  # Threshold points in RGB
W_comp     = np.full(ssize + (numDirPts, 2), np.nan)  # Threshold points in W-space
N_comp     = np.full(W_comp.shape, np.nan)  # Threshold points in Normalized space
W_ref      = np.full(ssize + (2,), np.nan)  # Reference points in W-space
N_ref      = np.full(ssize + (2,), np.nan)  # Reference points in Normalized space
vecDir     = np.full(rgb_comp.shape, np.nan)  # Direction vectors in RGB
opt_vecLen = np.full(ssize + (numDirPts,), np.nan)  # Vector lengths for ΔE=1

# Ellipse fitting results
fitEllipse_scaled       = np.full(ssize + (2, nThetaEllipse,), np.nan)
fitEllipse_unscaled     = np.full(ssize + (2, nThetaEllipse,), np.nan)
rgb_comp_contour_scaled = np.full(ssize + (2, numDirPts,), np.nan)
ellParams               = np.full(ssize + (5,), np.nan)  # 5 free parameters for ellipse fitting

#%% **MAIN COMPUTATION LOOP**
"""
The logic:
1. **Three spaces** are involved: RGB space (3D), W-space (2D square), and Normalized space (0-1).
2. **Threshold points** are computed in RGB space (where CIELab operates).
3. **Transformations** are applied:
   - RGB → W using a homography matrix.
   - W → Normalized by scaling from (-1,1) to (0,1).
4. **Fit an ellipse** in Normalized space.
"""

for i in range(nGridPts_ref):
    for j in range(nGridPts_ref):
        # **Step 1: Convert Reference Stimulus**
        rgb_ref_pij = ref_points[:, i, j]  # RGB reference point
        W_ref_pij = M_RGBTo2DW @ rgb_ref_pij  # Transform to W space
        W_ref[i, j] = W_ref_pij[:2]  # Store in 2D W-space

        # **Step 2: Compute Chromatic Directions & Thresholds**
        for k in range(numDirPts):
            # Step 2.1: Define a chromatic direction in W-space and convert to RGB
            chrom_dir_W = grid_theta_xy[:, k] + W_ref[i, j]  # Shifted chromatic direction
            chrom_dir_rgb = M_2DWToRGB @ np.append(chrom_dir_W, 1)  # Convert back to RGB

            # Step 2.2: Compute normalized direction vector in RGB space
            vecDir_k = chrom_dir_rgb - rgb_ref_pij  # Vector difference
            vecDir[i, j, k] = vecDir_k / np.linalg.norm(vecDir_k)  # Normalize

            # Step 2.3: Find vector length that produces ΔE = 1 in CIELab space
            opt_vecLen[i, j, k] = sim_thres_CIELab.find_vecLen(
                rgb_ref_pij, vecDir[i, j, k], deltaE_1JND, coloralg=color_diff_algorithm
            )

            # Step 2.4: Compute threshold point in RGB space
            rgb_comp[i, j, k] = opt_vecLen[i, j, k] * vecDir[i, j, k] + rgb_ref_pij

        # **Step 3: Transform Threshold Points from RGB → W → Normalized Space**
        W_comp_pij = M_RGBTo2DW @ rgb_comp[i, j].T  # Convert to W-space
        W_comp_pij = W_comp_pij / W_comp_pij[-1]  # Normalize last row to 1
        W_comp[i, j] = W_comp_pij[:2].T  # Store 2D points

        # Convert W-space to Normalized space (scale from [-1,1] to [0,1])
        N_ref = color_thres_data.W_unit_to_N_unit(W_ref[i, j])
        N_comp = color_thres_data.W_unit_to_N_unit(W_comp[i, j].T)

        # **Step 4: Fit Ellipse**
        fit_results = fit_2d_isothreshold_contour(
            N_ref, None, comp_RGB=N_comp, nThetaEllipse=nThetaEllipse, ellipse_scaler=contour_scaler
        )

        # Store ellipse fitting results
        fitEllipse_scaled[i, j], fitEllipse_unscaled[i, j], rgb_comp_contour_scaled[i, j], _, ellParams[i, j, :] = fit_results


#%% PLOTTING 
# Define coarse grid bounds for the varying color dimension
color_range = np.linspace(0.2, 0.8, nGridPts_ref)  # Lower bound: 0.2, Upper bound: 0.8

# Generate 2D grid for (GB, RB, RG) slices in RGB space
X, Y = np.meshgrid(color_range, color_range)
grid_est = np.dstack((X, Y))  # Shape: (nGridPts_ref, nGridPts_ref, 2)

# Compute colormap for background
cmap_W = color_thres_data.N_unit_to_W_unit(grid_est)  # Convert from Normalized to W space
cmap_W_hom = np.vstack((cmap_W.reshape(-1, 2).T, np.ones((1, nGridPts_ref**2))))  # Homogeneous coordinates
cmap_bg = (M_2DWToRGB @ cmap_W_hom).T.reshape(nGridPts_ref, nGridPts_ref, -1)  # Transform to RGB space

# Initialize visualization object
sim_CIE_vis = CIELabVisualization(SimThresCIELab(path_str, background_RGB), 
                                  fig_dir=output_figDir, save_fig=False)

# Plot 2D plane with computed ellipses and thresholds
sim_CIE_vis.plot_2D_single_plane(grid_est, 
                                 fitEllipse_scaled, 
                                 rawData=rgb_comp_contour_scaled, 
                                 rgb_background=None, 
                                 ell_lc=cmap_bg)

#%% save the data
#convert fitted ellipses to RGB space for future uses (maybe we want to visualize that in the RGB space)
fitEllipse_scaled_RGB_temp1 = np.reshape(np.transpose(color_thres_data.N_unit_to_W_unit(fitEllipse_scaled), (2, 0,1,3)), (2,-1))
fitEllipse_scaled_RGB_temp2 = np.vstack((fitEllipse_scaled_RGB_temp1, 
                                         np.ones((1, nGridPts_ref**2 * nThetaEllipse))))
fitEllipse_scaled_RGB_temp3 = M_2DWToRGB @ fitEllipse_scaled_RGB_temp2
fitEllipse_scaled_RGB = np.transpose(np.reshape(fitEllipse_scaled_RGB_temp3, 
                                                (3, nGridPts_ref, nGridPts_ref, nThetaEllipse)), (1,2,3,0))

#repeat the same for fitEllipse_unscaled
fitEllipse_unscaled_RGB_temp1 = np.reshape(np.transpose(color_thres_data.N_unit_to_W_unit(fitEllipse_unscaled), (2, 0,1,3)), (2,-1))
fitEllipse_unscaled_RGB_temp2 = np.vstack((fitEllipse_unscaled_RGB_temp1, 
                                         np.ones((1, nGridPts_ref**2 * nThetaEllipse))))
fitEllipse_unscaled_RGB_temp3 = M_2DWToRGB @ fitEllipse_unscaled_RGB_temp2
fitEllipse_unscaled_RGB = np.transpose(np.reshape(fitEllipse_unscaled_RGB_temp3,
                                                  (3, nGridPts_ref, nGridPts_ref, nThetaEllipse)), (1,2,3,0))

# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(*gamut_rgb,color ='k')
# # Scatter plot
# ax.scatter(*fitEllipse_scaled_RGB_temp3, cmap='viridis', marker='.', s=10, alpha=0.7, edgecolors='none')
# ax.view_init(elev=30, azim=-120)
# ax.set_xlim([0,1]); ax.set_ylim([0,1]); ax.set_zlim([0,1]); ax.set_aspect('equal')


#% save data
output_file = f'Isothreshold_ellipses_isoluminant_{color_diff_algorithm}.pkl'
full_path = os.path.join(output_fileDir, output_file)

variable_names = ['color_thres_data','sim_thres_CIELab', 'mat_file','iso_mat',
                  'gamut_rgb', 'M_RGBTo2DW', 'M_2DWToRGB', 'nGridPts_ref',
                  'ref_points', 'numDirPts', 'grid_theta_xy', 'deltaE_1JND',
                  'color_diff_algorithm', 'contour_scaler','nThetaEllipse',
                  'rgb_comp', 'W_comp', 'N_comp', 'W_ref', 'N_ref', 'vecDir',
                  'opt_vecLen', 'fitEllipse_scaled', 'fitEllipse_unscaled',
                  'rgb_comp_contour_scaled', 'ellParams', 'fitEllipse_scaled_RGB',
                  'fitEllipse_unscaled_RGB']
vars_dict = {}
for var_name in variable_names:
    try:
        # Check if the variable exists in the global scope
        vars_dict[var_name] = eval(var_name)
    except NameError:
        # If the variable does not exist, assign None and print a message
        vars_dict[var_name] = None
        print(f"Variable '{var_name}' does not exist. Assigned as None.")
        
# Write the list of dictionaries to a file using pickle
with open(full_path, 'wb') as f:
    pickled.dump(vars_dict, f)
        



