#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 20:31:02 2024

@author: fangfang
"""

import jax
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
import sys
import os
import dill as pickled
import numpy as np
#sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids')
sys.path.append("/Users/fh862-adm/Documents/GitHub/ellipsoids/ellipsoids")
from analysis.ellipses_tools import ellParams_to_covMat, rotAngle_to_eigenvectors
#sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/Python version')
sys.path.append("/Users/fh862-adm/Documents/GitHub/ColorEllipsoids/Python version")
from analysis.model_performance import ModelPerformance
from analysis.utils_load import select_file_and_get_path
        
# Define base directory and figure output directory
base_dir = '/Volumes/T9/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'
fig_outputDir = base_dir+'ELPS_analysis/ModelComparison_FigFiles/2D_oddity_task/'

#%% ------------------------------------------
# Load data simulated using CIELab
# ------------------------------------------
# List to store loaded simulation data
nFiles_load_indvEll = 4
data_load_indvEll = []
input_fileDir_all1 = []
file_name_all1 = []
for _ in range(nFiles_load_indvEll): #it can be any number depending on how many files we want to compare
    #META_analysis/ModelFitting_DataFiles/2dTask_indvEll/CIE/sub2/subset14014'
    #'IndvFitted_byWishart_Isoluminant plane_49interleaved2DExpt_10_10_10_330_AEPsychSampling_EAVC_sub2_subset14014.pkl'
    input_fileDir_fits1, file_name1 = select_file_and_get_path()
    input_fileDir_all1.append(input_fileDir_fits1)
    file_name_all1.append(file_name1)
    
    subset_str = 'subset' + file_name1.split('subset')[1].split('.pkl')[0]
    
    # Construct the full path to the selected file
    full_path1 = os.path.join(input_fileDir_fits1, file_name1)
    
    # Load the necessary variables from the file
    with open(full_path1, 'rb') as f:
        vars_dict1 = pickled.load(f)
    
    NUM_GRID_PTS = vars_dict1['NUM_GRID_PTS']
    model_ellParams1 = np.reshape(vars_dict1[f'model_pred_indv_{subset_str}']['fitEll_params'],
                                 (NUM_GRID_PTS, NUM_GRID_PTS, -1))
    data_load_indvEll.append(model_ellParams1.tolist())

#load one from the Wishart fits
nFiles_load_Wishart = 2
data_load_Wishart = []
input_fileDir_all2 = []
file_name_all2 = []
for _ in range(nFiles_load_Wishart): #it can be any number depending on how many files we want to compare
    #'META_analysis/ModelFitting_DataFiles/2dTask/CIE/sub2/subset6000'
    #'Fitted_byWishart_Isoluminant plane_49interleaved2DExpt_10_10_10_330_AEPsychSampling_EAVC_sub2.pkl'
    input_fileDir_fits2, file_name2 = select_file_and_get_path()
    input_fileDir_all2.append(input_fileDir_fits2)
    file_name_all2.append(file_name2)
    
    # Construct the full path to the selected file
    full_path2 = os.path.join(input_fileDir_fits2, file_name2)
    
    # Load the necessary variables from the file
    with open(full_path2, 'rb') as f:
        vars_dict2 = pickled.load(f)
    try:
        model_ellParams2 = vars_dict2['model_pred_Wishart_grid7'].params_ell
    except:
        model_ellParams2 = vars_dict2['model_pred_Wishart'].params_ell
    data_load_Wishart.append(model_ellParams2)

#%% load ground truths
# - Transformation matrices for converting between DKL, RGB, and W spaces
color_thres_data = vars_dict1['color_thres_data']
# - Dimensionality of the color space (e.g., 2D for isoluminant planes)
COLOR_DIMENSION = color_thres_data.color_dimension

gt_ellParams = vars_dict1['gt_Wishart'].params_ell
grid = vars_dict1['grid']
num_grid_pts = grid.shape[0]

#%%
# Create an instance of ModelPerformance to evaluate model predictions
model_perf = ModelPerformance(COLOR_DIMENSION, gt_ellParams)
indices = [0, num_grid_pts-1]

# Select specific points (corners) in the 2D or 3D space for comparison
if COLOR_DIMENSION == 2:
    # For 2D, select specific corner points in the plane
    idx_corner = [[i, j] for i in indices for j in indices]
    covMat_corner = []
    for i,j in idx_corner:
        _, _, a_ij, b_ij, R_ij = gt_ellParams[i][j]
        radii_ij = np.array([a_ij, b_ij])
        eigvec_ij = rotAngle_to_eigenvectors(R_ij)
        covMat_corner_ij = ellParams_to_covMat(radii_ij, eigvec_ij) 
        covMat_corner.append(covMat_corner_ij)
else:
    # For 3D, select corner points in the 3D space
    # idx_corner = [[0,0,0],[4,0,0],[0,4,0],[0,0,4],
    #               [4,4,0],[0,4,4],[4,0,4],[4,4,4]]
    idx_corner = [[i, j, k] for i in indices for j in indices for k in indices]

    # # Retrieve covariance matrices at these corner points
    # covMat_corner = [ellParams_to_covMat(CIE_results['ellipsoidParams'][i][j][k]['radii'],\
    #                 CIE_results['ellipsoidParams'][i][j][k]['evecs']) for i, j, k in idx_corner]

# Evaluate the model performance using the loaded data and corner points
BW_distance_output_indvEll = np.full((nFiles_load_indvEll, 2, *np.tile(NUM_GRID_PTS, COLOR_DIMENSION)), np.nan)
for i in range(nFiles_load_indvEll):
    model_perf.evaluate_model_performance([data_load_indvEll[i]],covMat_corner = covMat_corner)
    BW_distance_output_indvEll[i] = model_perf.BW_distance
    
# Evaluate the model performance using the loaded data and corner points
BW_distance_output_Wishart = np.full((nFiles_load_Wishart, 2, *np.tile(NUM_GRID_PTS, COLOR_DIMENSION)), np.nan)
for i in range(nFiles_load_Wishart):
    model_perf.evaluate_model_performance([data_load_Wishart[i]],covMat_corner = covMat_corner)
    BW_distance_output_Wishart[i] = model_perf.BW_distance

# %% ------------------------------------------
# Plot Bures-Wasserstein distance
# ------------------------------------------
# plotting starts from here
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams.update({'font.size': 10})
if COLOR_DIMENSION == 2:
    # 2D case: get colors for the corner points from the reference stimulus
    cmap_temp = np.hstack((np.vstack([grid[m, n] for m, n in idx_corner]), np.ones((len(idx_corner), 1))))
    cmap_BW = (color_thres_data.M_2DWToRGB @ cmap_temp.T).T

# else:
#     cmap_BW = np.vstack([CIE_stim['ref_points'][m, n, o] 
#                                  for m, n, o in idx_corner])    

axis1_merge = tuple(list(range(1,COLOR_DIMENSION+2)))
BW_distance_output_median_indvEll = np.median(BW_distance_output_indvEll, axis = axis1_merge)
# Create the error for yerr as a tuple of arrays for asymmetrical error
yerr_indvEll = np.array([BW_distance_output_median_indvEll - \
                            np.min(BW_distance_output_indvEll, axis = axis1_merge), 
                 np.max(BW_distance_output_indvEll, axis = axis1_merge) - \
                     BW_distance_output_median_indvEll])
    
BW_distance_output_median_Wishart = np.median(BW_distance_output_Wishart, axis = axis1_merge)
# Create the error for yerr as a tuple of arrays for asymmetrical error
yerr_Wishart = np.array([BW_distance_output_median_Wishart - \
                            np.min(BW_distance_output_Wishart, axis = axis1_merge), 
                 np.max(BW_distance_output_Wishart, axis = axis1_merge) - \
                     BW_distance_output_median_Wishart])

BW_distance_circle_median = np.median(model_perf.BW_distance_minEigval)
BW_distance_corner_median = np.median(model_perf.BW_distance_corner, axis = axis1_merge[:-1])

#%%plotting
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams.update({'font.size': 13})

fig1, ax1 = plt.subplots(1,1, figsize = (4.5, 3.25), dpi = 1024) #; format 1: (3.5, 2.2); format 2:  (3.2, 2.2)
y_ub = 0.1
ax1.plot([-3, nFiles_load_indvEll+2], [BW_distance_circle_median, BW_distance_circle_median],
         c = 'k',ls = '-',lw = 2, alpha = 0.8)
for i in range(len(covMat_corner)):
    ax1.plot([-3, nFiles_load_indvEll+2], np.array([1,1])*BW_distance_corner_median[i],
             c = cmap_BW[i],ls = '-',lw = 2, alpha = 0.8)
#Wishart
for i in range(nFiles_load_indvEll):
    ax1.errorbar(i+1, BW_distance_output_median_indvEll[i], 
                 yerr=yerr_indvEll[:, i].reshape(2, 1),
                fmt='o', capsize=0, c = np.array([0,0,0])+i*0.2, 
                marker = 'D', markersize = 10, lw = 2)
for i in range(nFiles_load_Wishart):
    ax1.errorbar(-i-1, BW_distance_output_median_Wishart[i], 
                 yerr=yerr_Wishart[:, i].reshape(2, 1),
                fmt='o', capsize=0, c = 'k', 
                marker = 'o', markersize = 12, lw = 2)
ax1.plot([0,0],[0,y_ub],ls = '--', lw=0.5, c = 'k')
ax1.set_xticks([-2,-1] + list(range(1,nFiles_load_indvEll+1)))
ax1.set_xticklabels(['6000\n(2D)','6000\n(4D)', '6027', '10045', '14014', '17640'], rotation= 30) #[6027, 10045, 14014, 17640]
ax1.set_xlabel('Total number of trial')
#ax1.set_title(slc_color)
ax1.set_xlim([-3, nFiles_load_indvEll + 1])
ax1.set_yticks(np.linspace(0,y_ub,3))
ax1.set_ylim([0,y_ub])
ax1.set_ylabel('Bures-Wasserstein distance')
plt.tight_layout()
figName1 = f"ModelPerformance_BuresWassersteinDistance_Wishart_vs_IndvEll.pdf"
full_path1 = os.path.join(fig_outputDir, figName1)
if saveFig: fig1.savefig(full_path1)   
