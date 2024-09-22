#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 22:04:13 2024

@author: fangfang
"""

import jax
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import os
import dill as pickle
import numpy as np
sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids')
from analysis.ellipses_tools import ellParams_to_covMat, rotAngle_to_eigenvectors
from analysis.color_thres import color_thresholds
sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/Python version')
from analysis.model_performance import ModelPerformance
        
# Define base directory and figure output directory
base_dir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'
fig_outputDir = base_dir+'ELPS_analysis/ModelPerformance_FigFiles/'

#%% ------------------------------------------
# Load data simulated using CIELab
# ------------------------------------------
plane_2D = 'GB plane'
jitter = 0.3
seed_list = list(range(10))
nSims = [6000, 4000, 2000]     # Number of simulations in total
nLevels = len(nSims)
saveFig = True            # Whether to save the figures

# Path to the directory containing the simulation data files
path_str1 = base_dir + 'ELPS_analysis/ModelFitting_DataFiles/2D_oddity_task/' 

# List to store loaded simulation data
data_load_noFixedRef, data_load_fixedRef = [], []
    
# Loop through each jitter level and load corresponding simulation data
for k in nSims:        
    data_load_noFixedRef_k, data_load_fixedRef_k = [],[]
    for l in seed_list:
        # Construct the file name based on the jitter level and other parameters
        file_name_j = f'Fitted_isothreshold_{plane_2D}_sim{k}total_samplingNearContour_noFixedRef_' +\
                      f'jitter{jitter}_seed{l}_bandwidth0.005_oddity.pkl'
        full_path_j = f"{path_str1}random_ref/{file_name_j}"
        
        # Change directory and load the simulation data using pickle
        os.chdir(path_str1+'random_ref/')
        with open(full_path_j, 'rb') as f:
            data_load_l = pickle.load(f)
            data_load_noFixedRef_k.append(data_load_l) # Append the loaded data to the list
            
        # Construct the file name based on the jitter level and other parameters
        file_name_jj = f'Fitted_isothreshold_{plane_2D}_sim{int(k/25)}perCond_samplingNearContour_' +\
                      f'jitter{jitter}_seed{l}_bandwidth0.005_oddity.pkl'
        full_path_jj = f"{path_str1}{file_name_jj}"
        
        # Change directory and load the simulation data using pickle
        os.chdir(path_str1)
        with open(full_path_jj, 'rb') as f:
            data_load_ll = pickle.load(f)
            data_load_fixedRef_k.append(data_load_ll) # Append the loaded data to the list
    data_load_noFixedRef.append(data_load_noFixedRef_k) 
    data_load_fixedRef.append(data_load_fixedRef_k)

#%% load ground truths
# Create an instance of the color_thresholds class
color_thres_data = color_thresholds(2, base_dir + 'ELPS_analysis/', plane_2D = plane_2D)

# Load CIE data for the ground truth ellipses/ellipsoids
color_thres_data.load_CIE_data()
CIE_results = color_thres_data.get_data('results2D',  dataset = 'CIE_data')  
CIE_stim = color_thres_data.get_data('stim2D', dataset = 'CIE_data')  
#scaler for the ellipse/ellipsoids 
scaler_x1 = 5

#%%
# Create an instance of ModelPerformance to evaluate model predictions
model_perf = ModelPerformance(2,
                              CIE_results, 
                              CIE_stim, 
                              list(range(len(seed_list))), 
                              plane_2D = plane_2D,
                              verbose = True)

indices = [0, 2, 4]
idx_corner = [[i, j] for i in indices for j in indices]
ellParams_slc = CIE_results['ellParams'][model_perf.plane_2D_idx]
covMat_corner = []
for i,j in idx_corner:
    _, _, a_ij, b_ij, R_ij = ellParams_slc[i][j]
    radii_ij = np.array([a_ij, b_ij])*scaler_x1
    eigvec_ij = rotAngle_to_eigenvectors(R_ij)
    covMat_corner_ij = ellParams_to_covMat(radii_ij, eigvec_ij) 
    covMat_corner.append(covMat_corner_ij)

# Evaluate the model performance using the loaded data and corner points
BW_distance_output_noFixedRef = np.full((nLevels, len(seed_list), 5, 5), np.nan)
for i in range(nLevels):
    model_perf.evaluate_model_performance(data_load_noFixedRef[i],covMat_corner = covMat_corner)
    BW_distance_output_noFixedRef[i] = model_perf.BW_distance
    
# Evaluate the model performance using the loaded data and corner points
BW_distance_output_fixedRef = np.full((nLevels, len(seed_list), 5, 5), np.nan)
for i in range(nLevels):
    model_perf.evaluate_model_performance(data_load_fixedRef[i])
    BW_distance_output_fixedRef[i] = model_perf.BW_distance

# %% ------------------------------------------
# Plot Bures-Wasserstein distance
# ------------------------------------------
# plotting starts from here
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams.update({'font.size': 10})
cmap_BW = np.vstack([CIE_stim['ref_points'][model_perf.plane_2D_idx][:, m, n] 
                             for m, n in idx_corner])

BW_distance_output_median_noFixedRef = np.median(BW_distance_output_noFixedRef, axis = (1,2,3))
# Create the error for yerr as a tuple of arrays for asymmetrical error
yerr_noFixedRef = np.array([BW_distance_output_median_noFixedRef - \
                            np.min(BW_distance_output_noFixedRef, axis = (1,2,3)), 
                 np.max(BW_distance_output_noFixedRef, axis = (1,2,3)) - \
                     BW_distance_output_median_noFixedRef])
    
BW_distance_output_median_fixedRef = np.median(BW_distance_output_fixedRef, axis = (1,2,3))
# Create the error for yerr as a tuple of arrays for asymmetrical error
yerr_fixedRef = np.array([BW_distance_output_median_fixedRef - \
                            np.min(BW_distance_output_fixedRef, axis = (1,2,3)), 
                 np.max(BW_distance_output_fixedRef, axis = (1,2,3)) - \
                     BW_distance_output_median_fixedRef])
BW_distance_circle_median =np.median(model_perf.BW_distance_minEigval)
BW_distance_corner_median = np.median(model_perf.BW_distance_corner, axis = (1,2))
# Create a square using the Rectangle patch
#square = patches.Rectangle((-1, np.min(model_perf.BW_distance_minEigval)),
#                           5, np.max(model_perf.BW_distance_minEigval) - \
#                               np.min(model_perf.BW_distance_minEigval),
#                           linewidth=2, edgecolor=[0.9,0.9,0.9], facecolor=[0.9,0.9,0.9])

fig1, ax1 = plt.subplots(1,1, figsize = (3,1.5), dpi = 256)
x_left= np.linspace(0,1,nLevels)
x_right = np.linspace(2,3,nLevels)
for i in range(nLevels):
    ax1.errorbar(x_left[i], BW_distance_output_median_noFixedRef[i], yerr=yerr_noFixedRef[:, i].reshape(2, 1),
                fmt='o', capsize=5, c = np.array([0,0,0])+i*0.3, marker = 's', markersize = 5)
    ax1.errorbar(x_right[i], BW_distance_output_median_fixedRef[i], yerr=yerr_fixedRef[:,i].reshape(2, 1),
                fmt='o', capsize=5, c = np.array([0,0,0])+i*0.3, marker = 'o', markersize = 6)
# xx = np.linspace(0,1,nLevels)
# for i in range(nLevels):
#     ax1.scatter(xx[i] + np.random.randn(10*25)*0.02, BW_distance_output_fixedRef[i].flatten(), 
#                 facecolor = [0.5,0.5,0.5],
#                 edgecolor = 'none', alpha = 0.5, s = 1)
ax1.plot([-1, nLevels*2], [BW_distance_circle_median, BW_distance_circle_median],c = 'k',ls = '--',lw = 0.5)
# Add the square to the plot
#ax1.add_patch(square)

for i in range(len(covMat_corner)):
    ax1.plot([-1, nLevels*2], np.array([1,1])*BW_distance_corner_median[i],c = cmap_BW[i],ls = '--',lw = 0.5)
ax1.set_xticks([1.5])
ax1.set_xticklabels([plane_2D])
ax1.set_xlim([-1,4])
ax1.set_yticks([0,0.03, 0.06])
ax1.set_ylim([0,0.06])
ax1.set_ylabel('BW distance')
plt.tight_layout()
figName1 = f"ModelPerformance_BuresWassersteinDistance_2D{plane_2D}_"+\
    f"wbenchmark_nSims{nSims}_jitter{jitter}_fixedRef.pdf"
full_path1 = os.path.join(fig_outputDir, figName1)
if saveFig: fig1.savefig(full_path1)   

