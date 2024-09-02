#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 12:45:28 2024

@author: fangfang
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.io import loadmat
import sys
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

#load functions from other modules
sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids")
from analysis.color_thres import color_thresholds
from analysis.ellipsoids_tools import slice_ellipsoid_byPlane
sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/Python version/")
from plotting.sim_CIELab_plotting import CIELabVisualization

#%% load the isoluminant plane
path_str = "/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/FilesFromPsychtoolbox/"
sys.path.append(path_str)
#load data
iso_mat = loadmat('W_from_PlanarGamut.mat')
gamut_rgb = iso_mat['gamut_bg_primary']
corner_points_rgb = iso_mat['cornerPointsRGB']

#%% define output directory for output files and figures
COLOR_DIMENSION = 3
# base directory
baseDir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'
# Create an instance of the class
color_thres_data = color_thresholds(COLOR_DIMENSION, baseDir + 'ELPS_analysis/')
    
color_thres_data.load_CIE_data()
# Load Wishart model fits
color_thres_data.load_model_fits()  

# Retrieve specific data from Wishart_data
CIE_stim = color_thres_data.get_data('stim3D', dataset='CIE_data')
CIE_data = color_thres_data.get_data('results3D', dataset='CIE_data')
ref_points = CIE_stim['ref_points']
gt_Wishart = color_thres_data.get_data('model_pred_Wishart', dataset = 'Wishart_data')
fitEll_unscaled = (gt_Wishart.fitEll_unscaled - \
                   color_thres_data.N_unit_to_W_unit(ref_points[:,:,:,:,None]))/2 +\
                   ref_points[:,:,:,:,None]

#%%
fig = plt.figure(figsize=(8,8),dpi = 256)
ax = fig.add_subplot(111, projection='3d')
ax.plot(*gamut_rgb,color='k',lw =1)
# Add the filled area
verts = [list(zip(*gamut_rgb))]
ax.add_collection3d(Poly3DCollection(verts, color=[0.5,0.5,0.5], alpha=0.4))
idx_slc = np.array([0,2,4])
nIdx_slc = len(idx_slc)
nRef_slc = nIdx_slc**COLOR_DIMENSION
ref_points_slc = ref_points[idx_slc][:,idx_slc][:,:,idx_slc]
fitEll_unscaled_slc = fitEll_unscaled[idx_slc][:,idx_slc][:,:,idx_slc]
thres_points_slc = CIE_data['rgb_surface_scaled'][idx_slc][:,idx_slc][:,:,idx_slc]
CIELabVisualization.plot_3D_isothreshold_ellipsoid(np.reshape(ref_points_slc,(nRef_slc,COLOR_DIMENSION)),
                                                   np.reshape(fitEll_unscaled_slc,(nRef_slc,COLOR_DIMENSION,-1)),
                                                   ax = ax,
                                                   visualize_thresholdPoints = True,
                                                   visualize_ellipsoids = True,
                                                   scatter_alpha = 1,
                                                   scatter_ms = 1,
                                                   ref_ms = 3,
                                                   threshold_points = np.reshape(thres_points_slc,(nRef_slc,)+thres_points_slc.shape[3:]))
plt.show()

#%%
ref_points_on_plane = color_thres_data.N_unit_to_W_unit(np.transpose(iso_mat['ref_rgb'],(1,0)))
nRef_on_plane = ref_points_on_plane.shape[0]
# Transpose the grid to match the expected input format of the model's prediction functions.
# The transposition is dependent on the color dimension to ensure the correct orientation of the data.
grid_trans = ref_points_on_plane[np.newaxis, np.newaxis,:,:]
# batch compute 78% threshold contour based on estimated weight matrix
test_Wishart = gt_Wishart
test_Wishart.convert_Sig_Threshold_oddity_batch(grid_trans)

#%% Calculate the mean of the points (centroid)
#here we can use either gamut_rgb or corner_points_rgb
centroid = np.mean(corner_points_rgb, axis=1)
# Subtract the centroid to center the points
centered_points = corner_points_rgb - centroid[:,None]
# Perform Singular Value Decomposition (SVD)
U, S, Vt = np.linalg.svd(np.transpose(centered_points,(1,0)))

nTheta = 200
sliced_ell_byPlane = np.full((nRef_on_plane, COLOR_DIMENSION, nTheta), np.nan)
for n in range(nRef_on_plane):
    ell_params_n = test_Wishart.params_ell[0][0][n]
    radii_n = ell_params_n['radii']/2
    center_n = color_thres_data.W_unit_to_N_unit(np.reshape(ell_params_n['center'],(-1)))
    evecs_n = ell_params_n['evecs']
    sliced_ell_byPlane[n], _ = slice_ellipsoid_byPlane(center_n, 
                                                    radii_n, 
                                                    evecs_n, 
                                                    Vt[0], 
                                                    Vt[1],
                                                    num_grid_pts= nTheta)

#%%
fig1 = plt.figure(figsize=(8,8),dpi = 256)
ax1 = fig1.add_subplot(111, projection='3d')

ax1.plot(*gamut_rgb,color ='k')
for n in range(nRef_on_plane):
    ax1.plot(*sliced_ell_byPlane[n],color = 'k')
# Add the filled area
ax1.add_collection3d(Poly3DCollection(verts, color=[0.5,0.5,0.5], alpha=0.35))
#plot Vt, which is an orthogonal matrix where each row represents a basis vector 
#in the input space (the 3-dimensional space of the original data)
for s in range(COLOR_DIMENSION):
    ax1.plot([centroid[0], centroid[0]+Vt[s,0]],
             [centroid[1], centroid[1]+Vt[s,1]],
             [centroid[2], centroid[2]+Vt[s,2]],
             color='k', lw = 0.5)
fitEll_unscaled1 = color_thres_data.W_unit_to_N_unit(gt_Wishart.fitEll_unscaled)
CIELabVisualization.plot_3D_isothreshold_ellipsoid(color_thres_data.W_unit_to_N_unit(ref_points_on_plane),
                                                   np.reshape(fitEll_unscaled1,(nRef_on_plane,COLOR_DIMENSION,-1)),
                                                   ax = ax1,
                                                   visualize_ellipsoids = True,
                                                   ref_ms = 3,
                                                   view_angle = [30,-120])
plt.show()