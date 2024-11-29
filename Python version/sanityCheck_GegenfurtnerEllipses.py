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
subN = 2
path_str  = base_dir+ f'ELPS_analysis/Experiment_DataFiles/sub{subN}/'
file_name = f'Color_discrimination_2d_oddity_task_isoluminantPlane_MOCS_sub{subN}_copy.db'
full_path = f"{path_str}{file_name}"
os.chdir(path_str)

# Define the color plane for the simulations
cdim = 2
path_dir2 = base_dir + 'ELPS_analysis/'
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
grid_data = np.reshape(data_load['grid_stim'], [data_load['nRefs'],2])
session_order = data_load['session_order']
nTrials_perS = data_load['nTrials']
x1_raw = data_load['data'][2]

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

def stretchingMat_from_covMatDKL(covMat_ell_DKL):
    """
    Compute a stretching matrix that converts an ellipse, represented by a 
    covariance matrix in DKL space, into an ellipse with unit x- and y-axis lengths.

    Parameters:
    -----------
    covMat_ell_DKL : ndarray
        A 2x2 covariance matrix representing the shape and orientation of the 
        ellipse in DKL space.

    Returns:
    --------
    stretchingMat_DKL_to_unit : ndarray
        A 2x2 transformation matrix that scales the ellipse to have unit lengths 
        along the x- and y-axes.
    stretchingMat_unit_to_DKL : ndarray
        The inverse of stretchingMat_DKL_to_unit, which transforms the normalized ellipse 
        back to its original shape and orientation.
    """
    # Extract variances along x- and y-axes
    sigma_xx = covMat_ell_DKL[0, 0]
    sigma_yy = covMat_ell_DKL[1, 1]

    # Construct the scaling matrix to normalize x- and y-axis lengths
    scaling_x = 1 / np.sqrt(sigma_xx)
    scaling_y = 1 / np.sqrt(sigma_yy)
    stretchingMat_DKL_to_unit = np.diag([scaling_x, scaling_y])

    # Compute the inverse scaling matrix
    stretchingMat_unit_to_DKL = np.linalg.inv(stretchingMat_DKL_to_unit)

    return stretchingMat_DKL_to_unit, stretchingMat_unit_to_DKL

def normalize_ellipse_axes(stretchingMat_DKL_to_unit, cov_matrix_test):    
    """
    Normalize a covariance matrix of an ellipse using a stretching matrix.

    This function uses a precomputed stretching matrix to normalize an ellipse 
    (i.e., scale its axes).

    Parameters:
    -----------
    stretchingMat_DKL_to_unit : ndarray
        A 2x2 diagonal transformation matrix that stretches an ellipse into another
        ellipse with unit x- and y-axis lengths.
    cov_matrix_test : ndarray
        A 2x2 covariance matrix representing the ellipse to be normalized.

    Returns:
    --------
    scaled_matrix : ndarray
        A 2x2 covariance matrix of the ellipse normalized to have unit axes.
    """
    # Apply the stretching matrix to normalize the covariance matrix.
    # This effectively rescales the ellipse to have unit length along its axes.
    scaled_matrix = stretchingMat_DKL_to_unit @ cov_matrix_test @ stretchingMat_DKL_to_unit.T
    
    return scaled_matrix
    

#%%-----------------------------------------------------------
#                           Main code
# -------------------------------------------------------------
# Select the center stimulus (corresponding to the gray stimulus)
idx_row = model_pred_Wishart.num_grid_pts1 // 2  # Row index for the center
idx_col = model_pred_Wishart.num_grid_pts2 // 2  # Column index for the center

# Retrieve the ellipse parameters for the gray stimulus at the center of the grid
# Parameters: x center, y center, major axis, minor axis, rotation angle
ell_params_grey_W = model_pred_Wishart.params_ell[idx_row][idx_col]

# Convert the ellipse parameters from W space to a covariance matrix in DKL space
covMat_ell_grey_DKL = convert_ellParamsW_to_covMatDKL(*ell_params_grey_W[2:], M_2DWToDKL)

# Compute the stretching matrices for transforming between DKL space and unit space
stretchingMat_DKL_to_unit, stretchingMat_unit_to_DKL = stretchingMat_from_covMatDKL(covMat_ell_grey_DKL)

# Generate points on a scaled unit circle (to serve as reference)
nPts_unit_circle = 17
scaler = 6
unit_circle_pts = (np.eye(2) * scaler) @ UnitCircleGenerate(nPts_unit_circle)

# Transform unit circle points from unit space to DKL space
ref_pts_DKL = stretchingMat_unit_to_DKL @ unit_circle_pts

# Transform DKL points to W space using the inverse of the transformation matrix
ref_pts_W = M_DKLTo2DW @ np.vstack((ref_pts_DKL, np.full((1, nPts_unit_circle), 1)))

# Extract original W space coordinates for the reference points
ref_pts_W_trunc_trans = ref_pts_W[:-1].T  # Shape: (nPts_unit_circle, 2)
ref_pts_W_org = ref_pts_W_trunc_trans[np.newaxis, :, :]  # Add an extra dimension for batch processing

# Retrieve threshold data at the sampled locations in W space
model_pred_Wishart_copy.convert_Sig_Threshold_oddity_batch(
    np.transpose(ref_pts_W_org, (2, 0, 1))
)

#%% 
"""
Process the gray stimulus at the center
"""
# Retrieve the ellipse parameters for the gray stimulus
params_grey = model_pred_Wishart.params_ell[idx_row][idx_col]

# Convert the ellipse parameters to covariance matrices in DKL and unit-stretched DKL spaces
covMat_grey_DKL = convert_ellParamsW_to_covMatDKL(*params_grey[2:], M_2DWToDKL)
covMat_grey_unit = normalize_ellipse_axes(stretchingMat_DKL_to_unit, covMat_ell_grey_DKL)

# Reconstruct the threshold contours for the gray stimulus
fine_ell_grey_DKL = convert_2Dcov_to_points_on_ellipse(covMat_grey_DKL)
fine_ell_grey_unit = convert_2Dcov_to_points_on_ellipse(covMat_grey_unit)

        
"""
Repeat for all the stimuli around the grey stimulus
"""
# Number of fine samples for threshold contours
nDir = 200  

# Initialize arrays for covariance matrices and threshold contours in DKL and unit space
covMat_around_grey_DKL = np.full((nPts_unit_circle, cdim, cdim), np.nan)
covMat_around_grey_unit = np.full(covMat_around_grey_DKL.shape, np.nan)
fine_ell_W = np.full((nPts_unit_circle, cdim, nDir), np.nan)
fine_ell_DKL = np.full(fine_ell_W.shape, np.nan)
fine_ell_unit = np.full(fine_ell_W.shape, np.nan)

# Iterate over each point on the unit circle
for n in range(nPts_unit_circle):
    # Retrieve the threshold contour in W space for the sampled location
    fine_ell_W[n] = model_pred_Wishart_copy.fitEll_unscaled[0, n]
    
    # Retrieve the ellipse parameters for the sampled location in W space
    params_around_grey_n = model_pred_Wishart_copy.params_ell[0][n]
    
    # Convert the ellipse parameters to a covariance matrix in DKL space
    covMat_around_grey_DKL[n] = convert_ellParamsW_to_covMatDKL(
        *params_around_grey_n[2:], M_2DWToDKL
    )
    
    # Convert the covariance matrix to unit-stretched space
    covMat_around_grey_unit[n] = normalize_ellipse_axes(
        stretchingMat_DKL_to_unit, 
        covMat_around_grey_DKL[n]
    )
    
    # Reconstruct the threshold contours in DKL and unit spaces
    fine_ell_DKL[n] = convert_2Dcov_to_points_on_ellipse(covMat_around_grey_DKL[n])
    fine_ell_unit[n] = convert_2Dcov_to_points_on_ellipse(covMat_around_grey_unit[n])

#%%
"""
Repeat computations for all the reference stimuli used in the experiment. 
These stimuli cover a larger region in the DKL space compared to the points 
around the unit circle handled in the previous section.
"""

# Initialize arrays to store reference points, covariance matrices, and contours
base_shape = (model_pred_Wishart.num_grid_pts1, model_pred_Wishart.num_grid_pts2)
exptRef_pts = np.full(base_shape + (cdim,), np.nan)  # Reference points in DKL space
covMat_exptRef_DKL = np.full(base_shape + (cdim, cdim), np.nan)  # Covariance matrices in DKL space
covMat_exptRef_unit = np.full(covMat_exptRef_DKL.shape, np.nan)  # Covariance matrices in unit space
fine_exptEll_unit = np.full(base_shape + (cdim, nDir), np.nan)  # Elliptical contours in unit space

# Compute the locations of comparison stimuli in DKL space (transformed from raw data)
exptComp_pts = stretchingMat_DKL_to_unit @ (M_2DWToDKL @ x1_raw.T)

# Iterate over all grid points corresponding to the reference stimuli
for n in range(base_shape[0]):
    for m in range(base_shape[1]):
        # Transform the reference stimulus from W space to DKL space and then to unit space
        exptRef_pts[n, m] = stretchingMat_DKL_to_unit @ (M_2DWToDKL @ data_load['grid'][n, m])
        
        # Retrieve ellipse parameters for the current reference stimulus
        params_exptRef_nm = model_pred_Wishart.params_ell[n][m]
        
        # Convert the ellipse parameters from W space to a covariance matrix in DKL space
        covMat_exptRef_DKL[n, m] = convert_ellParamsW_to_covMatDKL(
            *params_exptRef_nm[2:], M_2DWToDKL
        )
        
        # Transform the covariance matrix from DKL space to stretched unit space
        covMat_exptRef_unit[n, m] = normalize_ellipse_axes(
            stretchingMat_DKL_to_unit, 
            covMat_exptRef_DKL[n, m]
        )
        
        # Compute the elliptical contours in unit space from the covariance matrix
        fine_exptEll_unit[n, m] = convert_2Dcov_to_points_on_ellipse(covMat_exptRef_unit[n, m])

#%%------------------------------------------------------------------
# Visualize the threshold contours in the Wishart space and the DKL 
# stretched (unit) space
# -------------------------------------------------------------------
# Wishart space 
cmap = np.full((nPts_unit_circle, 3), np.nan)
figg, axx = plt.subplots(1, 2, figsize = (12,5), dpi = 1024)
for n in range(nPts_unit_circle):
    if n == 0: mk = 'd'
    else: mk = '+'
    cmap[n] = color_thres_data.M_2DWToRGB @ ref_pts_W[:,n]
    axx[0].scatter(ref_pts_W[0, n], ref_pts_W[1, n], color=cmap[n], 
                   marker= mk, s=20, lw = 0.75)
    axx[0].plot(*fine_ell_W[n], c=cmap[n], lw = 1.5)
axx[0].plot(*model_pred_Wishart.fitEll_unscaled[idx_row, idx_col], c= 'grey', lw = 1.5)
axx[0].scatter(0,0, color = 'k', marker = '+', s = 20, lw = 0.75)
axx[0].set_aspect('equal', adjustable='box')  # Make the axis square
axx[0].set_xlabel('Wishart space dimension 1')
axx[0].set_ylabel('Wishart space dimension 2')
axx[0].grid(True, color='grey',linewidth=0.1)

# # DKL L-M and S
# axx[1].plot(*fine_ell_grey_DKL, c = 'grey')
# for n in range(nPts_unit_circle):
#     if n == 0: mk = 'd'
#     else: mk = '+'
#     axx[1].scatter(ref_pts_DKL[0,n], ref_pts_DKL[1,n], 
#                     marker = mk, c = 'k', s = 20)
#     axx[1].plot(fine_ell_DKL[n,0] + ref_pts_DKL[0,n], 
#                 fine_ell_DKL[n,1] + ref_pts_DKL[1,n], c = cmap[n])
# axx[1].scatter(0,0, c = 'k', marker = '+', s = 20)
# axx[1].set_aspect('equal', adjustable='box')  # Make the axis square
# axx[1].set_xlabel('DKL L-M (unstretched)')
# axx[1].set_ylabel('DKL S (unstretched)')
# axx[1].grid(True, color='grey',linewidth=0.1)
    
# DKL L-M and S (stretched)
circle_pts = stretchingMat_DKL_to_unit @ ref_pts_DKL
for n in range(nPts_unit_circle):
    if n == 0: mk = 'd'
    else: mk = '+'
    axx[1].scatter(circle_pts[0,n], circle_pts[1,n], 
                   marker = mk, color = 'k', s = 20, lw = 0.75)
    axx[1].plot(fine_ell_unit[n,0] + circle_pts[0,n], 
                fine_ell_unit[n,1] + circle_pts[1,n],
                c = cmap[n], lw = 1.5)
axx[1].plot(fine_ell_grey_unit[0], fine_ell_grey_unit[1], c = 'grey', lw = 1.5)    
axx[1].scatter(0,0, color = 'k', marker = '+', s = 20)
axx[1].set_aspect('equal', adjustable='box')  # Make the axis square
axx[1].set_xlim([-10,10])
axx[1].set_ylim([-10,10])
axx[1].set_xticks(np.linspace(-10,10,5))
axx[1].set_yticks(np.linspace(-10,10,5))
axx[1].set_xlabel('DKL L-M (stretched)')
axx[1].set_ylabel('DKL S (stretched)')
axx[1].grid(True, color='grey',linewidth=0.1)
plt.tight_layout()
figg.savefig(output_figDir_fits+f"DKL_stretchedSpace_sub{subN}.pdf", 
             format='pdf', bbox_inches='tight')

#%%------------------------------------------------------------------
# Visualize the threshold contours in the DKL stretched (unit) space
# for all reference stimulus locations tested in the experiment
# -------------------------------------------------------------------
fig, ax = plt.subplots(1, 1, figsize = (12,6), dpi = 1024)
# DKL L-M and S (stretched)
for n in range(nPts_unit_circle):
    ax.scatter(circle_pts[0,n], circle_pts[1,n], 
                   marker = '+', c = 'k', s = 20, lw = 0.5)
    ax.plot(fine_ell_unit[n,0] + circle_pts[0,n], 
                fine_ell_unit[n,1] + circle_pts[1,n],
                c = cmap[n], lw = 1)

cmap_exptRef = np.full(base_shape + (3,),np.nan)
for n in range(base_shape[0]):
    for m in range(base_shape[1]):
        cmap_exptRef[n,m] = color_thres_data.M_2DWToRGB @ np.append(data_load['grid'][n,m], 1)
        ax.scatter(exptRef_pts[n,m,0],
                   exptRef_pts[n,m,1], 
                   marker = '+', c = cmap_exptRef[n,m])
        ax.plot(fine_exptEll_unit[n,m,0] + exptRef_pts[n,m,0],
                fine_exptEll_unit[n,m,1] + exptRef_pts[n,m,1],
                c = cmap_exptRef[n,m])

for i in range(9):
    idx_lb = nTrials_perS * i
    idx_ub = nTrials_perS * (i+1)
    cmap_exptRef_i = color_thres_data.M_2DWToRGB @ np.append(grid_data[session_order[i]-1], 1)
    ax.scatter(exptComp_pts[0, idx_lb: idx_ub],
               exptComp_pts[1, idx_lb: idx_ub], 
               c = cmap_exptRef_i, s = 1, alpha = 0.3)
ax.plot(fine_ell_grey_unit[0], fine_ell_grey_unit[1], c = 'grey', lw = 1.5)    
ax.scatter(0,0, c = 'k', marker = '+', s = 20, lw = 0.5)
ax.set_aspect('equal', adjustable='box')  # Make the axis square
ax.set_xlabel('DKL L-M (stretched)')
ax.set_ylabel('DKL S (stretched)')
ax.set_xlim([-50, 50])
ax.set_ylim([-25, 25])
ax.set_xticks(np.round(np.linspace(-50,50,9),2))
ax.set_yticks(np.round(np.linspace(-30,30,5),2))
ax.grid(True, color='grey',linewidth=0.1)
fig.savefig(output_figDir_fits+f"DKL_stretchedSpace_wExptRefs_sub{subN}.pdf",
             format='pdf', bbox_inches='tight')


