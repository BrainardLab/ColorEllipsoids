#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 18:18:02 2024

@author: fangfang
"""
# -----------------------------------------------------------
# SECTION 1: import modules and set directories
# -----------------------------------------------------------
import jax
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
import os
import dill as pickled
import sys
import numpy as np
import copy
# Set font style to Arial
plt.rcParams['font.family'] = 'Arial'

sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids")
from analysis.color_thres import color_thresholds
from analysis.ellipses_tools import ellParamsQ_to_covMat, UnitCircleGenerate,convert_2Dcov_to_points_on_ellipse

# this is the dir where we store our .db files
base_dir = '/Volumes/T9/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'
            
#specify the file name
subN = 1
path_str  = base_dir+ f'ELPS_analysis/Experiment_DataFiles/sub{subN}/'
file_name = f'Color_discrimination_2d_oddity_task_isoluminantPlane_MOCS_sub{subN}_copy.db'
full_path = f"{path_str}{file_name}"
os.chdir(path_str)

# this is the dir where we store CIE simulated data
path_dir2 = base_dir + 'ELPS_analysis/'

# Define the color plane for the simulations
cdim = 2
color_thres_data = color_thresholds(cdim, path_dir2, plane_2D = 'Isoluminant plane')
color_thres_data.load_transformation_matrix()
#the last row and col is filler. If input W has only two elements, then we just need to keep a portion of the matrix
M_2DWToDKL = color_thres_data.M_2DWToDKL[:2,:2] 
M_DKLTo2DW = color_thres_data.M_DKLTo2DW

#specify figure name and path
output_figDir_fits = base_dir+f'ELPS_Analysis/Experiment_FigFiles/sub{subN}/'
output_fileDir_fits = base_dir+ f'ELPS_Analysis/Experiment_DataFiles/sub{subN}/'

#%%
# fitting file
file_fit = 'Fitted_isothreshold_isoluminant_plane_360trialsPerRef_9refs_'+\
    f'AEPsychSampling_bandwidth0.005_sub{subN}.pkl'
full_path = f"{path_str}{file_fit}"
with open(full_path, 'rb') as f: data_load = pickled.load(f)
#load model predictions
model_pred_Wishart = data_load['model_pred_Wishart']
model_pred_Wishart_copy = copy.copy(model_pred_Wishart)
model = model_pred_Wishart.model
W_est = model_pred_Wishart.W_est

#%%
def convert_ellParamsW_to_covMatDKL(a, b, theta, M_trans):
    """
    Convert ellipse parameters in W space to a covariance matrix in DKL space.

    Parameters:
    a (float): Major axis length of the ellipse in W space.
    b (float): Minor axis length of the ellipse in W space.
    theta (float): Angle of rotation of the ellipse in degrees in W space.
    M_trans (ndarray): Transformation matrix to convert W space to DKL space.

    Returns:
    covMat_ell_DKL (ndarray): Covariance matrix of the ellipse in DKL space.
    """
    # Convert ellipse parameters (a, b, theta) to covariance matrix in W space
    covMat_ell_W = ellParamsQ_to_covMat(a, b, theta)

    # Transform the covariance matrix from W space to DKL space
    covMat_ell_DKL = M_trans @ covMat_ell_W @ M_trans.T

    return covMat_ell_DKL

def convert_covMatDKL_to_unitSpace(covMat_ell_DKL):
    """
    Transform a covariance matrix in DKL space into a unit circle space.

    This function converts an ellipse represented by a covariance matrix in 
    DKL space into the space of a unit circle. Optionally, it can rotate the 
    space such that the principal axes of the ellipse align with the coordinate axes.

    Parameters:
    -----------
    covMat_ell_DKL (ndarray):
        A 2x2 covariance matrix representing the shape of the ellipse in DKL space.
    flag_rotate (bool, optional):
        If True, rotates the space to align the principal axes of the ellipse with 
        the coordinate axes. If False (default), only scales the ellipse to a unit 
        circle without changing its orientation.
    """
    # Step 1: Perform eigen decomposition of the covariance matrix.
    eigVal_covMat_DKL, eigVec_covMat_DKL = np.linalg.eigh(covMat_ell_DKL)
    
    # Ensure the first element of each eigenvector is positive
    for i in range(eigVec_covMat_DKL.shape[1]):
        if eigVec_covMat_DKL[0, i] < 0:
            eigVec_covMat_DKL[:, i] *= -1

    # Step 2: Create a diagonal matrix for scaling, with entries being the inverse square roots of the eigenvalues.
    # This scaling ensures the axes of the ellipse are normalized to unit length.
    Lambda_inv_sqrt = np.diag(1 / np.sqrt(eigVal_covMat_DKL))

    # Step 3: Construct the transformation matrix for converting DKL space to unit circle space.
    # If rotation is enabled, include the eigenvectors to align the ellipse's axes with the coordinate axes.
    # Otherwise, skip rotation and use the identity matrix, preserving the original orientation.
    covMat_DKL_to_unit = Lambda_inv_sqrt @ eigVec_covMat_DKL.T

    # Step 4: Compute the inverse of the transformation matrix.
    # This matrix converts points from the unit circle space back to DKL space.
    covMat_unit_to_DKL = np.linalg.inv(covMat_DKL_to_unit)

    return covMat_DKL_to_unit, covMat_unit_to_DKL

#%%-----------------------------------------------------------
#                           Main code
# -------------------------------------------------------------
# Select the center stimulus, which corresponds to the gray stimulus
idx_row = model_pred_Wishart.num_grid_pts1 // 2  # Row index for the center
idx_col = model_pred_Wishart.num_grid_pts2 // 2  # Column index for the center

# Retrieve the ellipse parameters for the gray stimulus (center of the grid)
# Parameters: x center, y center, major axis, minor axis, rotation angle
ell_params_grey_W = model_pred_Wishart.params_ell[idx_row][idx_col]

# Convert the ellipse parameters from W space to a covariance matrix in DKL space
covMat_ell_grey_DKL = convert_ellParamsW_to_covMatDKL(*ell_params_grey_W[2:], M_2DWToDKL)

# Compute the transformation matrices between DKL space and unit circle space
covMat_DKL_to_unit, covMat_unit_to_DKL = convert_covMatDKL_to_unitSpace(covMat_ell_grey_DKL)

# Generate points on a scaled unit circle
nPts_unit_circle = 17
scaler = 6
unit_circle_pts = (np.eye(2) * scaler) @ UnitCircleGenerate(nPts_unit_circle)

# Transform unit circle points to DKL space
ref_pts_DKL = covMat_unit_to_DKL @ unit_circle_pts
# Transform DKL points to W space using the inverse of the transformation matrix
ref_pts_W = M_DKLTo2DW @ np.vstack((ref_pts_DKL, np.full((1, nPts_unit_circle), 1)))

# Extract the original W space coordinates (excluding the homogeneous component)
ref_pts_W_org = ref_pts_W[:-1].T  # Shape: (nPts_unit_circle, 2)
ref_pts_W_orgg = ref_pts_W_org[np.newaxis, :, :]  # Add an extra dimension for batch processing

# Retrieve threshold data at the sampled locations in W space
model_pred_Wishart_copy.convert_Sig_Threshold_oddity_batch(np.transpose(ref_pts_W_orgg, (2, 0, 1)))

#initialize
nDir = 200
covMat_around_grey_DKL = np.full((nPts_unit_circle, cdim,cdim),np.nan)
fine_ell_W = np.full((nPts_unit_circle, cdim, nDir), np.nan)
fine_ell_unit = np.full(fine_ell_W.shape, np.nan)
fine_ell_DKL = np.full(fine_ell_W.shape, np.nan)
for n in range(nPts_unit_circle):
    #get threshold contours in the W space
    fine_ell_W[n] = model_pred_Wishart_copy.fitEll_unscaled[0,n]
    #retrive the ell params
    params_around_grey_n = model_pred_Wishart_copy.params_ell[0][n]   
    # convert that ell parameters in W space to ellipses in DKL space
    covMat_around_grey_DKL[n] = convert_ellParamsW_to_covMatDKL(
        *params_around_grey_n[2:], M_2DWToDKL)
    
    # convert the cov matrix to threshold contours
    fine_ell_DKL[n] = convert_2Dcov_to_points_on_ellipse(covMat_around_grey_DKL[n])
    fine_ell_unit[n] = covMat_DKL_to_unit @ fine_ell_DKL[n]

#%% visualize
# Wishart space 
cmap = np.full((nPts_unit_circle, 3), np.nan)
figg, axx = plt.subplots(1, 2, figsize = (8,5), dpi = 1024)
axx[0].plot(*model_pred_Wishart.fitEll_unscaled[idx_row, idx_col], c= 'grey')
axx[0].scatter(0,0, c = 'k', marker = '+', s = 20)
for n in range(nPts_unit_circle):
    if n == 0: mk = 'd'
    else: mk = '+'
    axx[0].scatter(ref_pts_W[0, n], ref_pts_W[1, n], c=cmap[n], marker= mk, s=20)
    cmap[n] = color_thres_data.M_2DWToRGB @ ref_pts_W[:,n]
    axx[0].plot(*fine_ell_W[n], c=cmap[n])
axx[0].set_aspect('equal', adjustable='box')  # Make the axis square
axx[0].set_xlabel('Wishart space dimension 1')
axx[0].set_ylabel('Wishart space dimension 2')
axx[0].grid(True, color='grey',linewidth=0.1)

# # DKL L-M and S
# for n in range(nPts_unit_circle):
#     if n == 0: mk = '^'
#     else: mk = '+'
#     axx[1].scatter(ref_pts_DKL[0,n], ref_pts_DKL[1,n], 
#                    marker = mk, c = 'k', s = 20)
#     axx[1].plot(fine_ell_DKL[n,0] + ref_pts_DKL[0,n], 
#                 fine_ell_DKL[n,1] + ref_pts_DKL[1,n], c = cmap[n])
# axx[1].scatter(0,0, c = 'k', marker = '+', s = 20)
# axx[1].set_aspect('equal', adjustable='box')  # Make the axis square
# axx[1].set_xlabel('DKL L-M')
# axx[1].set_ylabel('DKL S')
# axx[1].grid(True, color='grey',linewidth=0.1)
    
# DKL L-M and S (stretched)
for n in range(nPts_unit_circle):
    if n == 0: mk = 'd'
    else: mk = '+'
    axx[1].scatter(unit_circle_pts[0,n], unit_circle_pts[1,n], 
                   marker = mk, c = 'k', s = 20)
    axx[1].plot(fine_ell_unit[n,0] + unit_circle_pts[0,n], 
                fine_ell_unit[n,1] + unit_circle_pts[1,n], c = cmap[n])
unit_circ = UnitCircleGenerate(nDir)
axx[1].plot(unit_circ[0], unit_circ[1], c = 'grey')    
axx[1].scatter(0,0, c = 'k', marker = '+', s = 20)
axx[1].set_aspect('equal', adjustable='box')  # Make the axis square
axx[1].set_xlim([-10,10])
axx[1].set_ylim([-10,10])
axx[1].set_xticks(np.linspace(-10,10,5))
axx[1].set_yticks(np.linspace(-10,10,5))
axx[1].set_xlabel('DKL L-M')
axx[1].set_ylabel('DKL S')
axx[1].grid(True, color='grey',linewidth=0.1)


