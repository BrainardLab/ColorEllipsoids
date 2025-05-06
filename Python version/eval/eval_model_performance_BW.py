#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 00:39:55 2024

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

#%% ------------------------------------------
# Load data simulated using CIELab
# ------------------------------------------
nFiles_load = 1
input_fileDir_all = []
file_name_all = []
model_ellParams_set = []
for _ in range(nFiles_load): #it can be any number depending on how many files we want to compare
    input_fileDir_fits, file_name = select_file_and_get_path()
    input_fileDir_all.append(input_fileDir_fits)
    file_name_all.append(file_name)
    
    # Construct the full path to the selected file
    full_path = os.path.join(input_fileDir_fits, file_name)
    
    # Load the necessary variables from the file
    with open(full_path, 'rb') as f:
        vars_dict = pickled.load(f)
    try:
        model_ellParams = vars_dict['model_pred_Wishart_grid7'].params_ell
    except:
        model_ellParams = vars_dict['model_pred_Wishart'].params_ell
    model_ellParams_set.append(model_ellParams)

# - Transformation matrices for converting between DKL, RGB, and W spaces
color_thres_data = vars_dict['color_thres_data']
# - Dimensionality of the color space (e.g., 2D for isoluminant planes)
COLOR_DIMENSION = color_thres_data.color_dimension
gt_ellParams = vars_dict['gt_Wishart'].params_ell
grid_trans = vars_dict['grid_trans']
num_grid_pts = grid_trans.shape[-1]

#%%
# Create an instance of ModelPerformance to evaluate model predictions
model_perf = ModelPerformance(COLOR_DIMENSION, gt_ellParams)

# Select specific points (corners) in the 2D or 3D space for comparison
if COLOR_DIMENSION == 2:
    # For 2D, select specific corner points in the plane
    indices = [0, num_grid_pts-1]
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
    indices = [0, num_grid_pts-1]
    idx_corner = [[i, j, k] for i in indices for j in indices for k in indices]

    # # Retrieve covariance matrices at these corner points
    # covMat_corner = [ellParams_to_covMat(CIE_results['ellipsoidParams'][i][j][k]['radii'],\
    #                 CIE_results['ellipsoidParams'][i][j][k]['evecs']) for i, j, k in idx_corner]
# Evaluate the model performance using the loaded data and corner points
model_perf.evaluate_model_performance(model_ellParams_set, covMat_corner = covMat_corner)
# Concatenate benchmark results for plotting
model_perf.concatenate_benchamrks()

# %% ------------------------------------------
# Plotting specifics
# ------------------------------------------
legend_str = ['6000 trials (2D)']  # Legend labels for different noise levels
assert len(legend_str) == nFiles_load, "The number of legend does not match the loaded dataset."
cmap_temp = plt.get_cmap('Dark2')       # Colormap for the plots
cmap_t = cmap_temp(np.linspace(0.3, 0.8, nFiles_load))[:, :3]

# Prepare colormap for corner points and benchmarks
cmap_BW = np.full((len(idx_corner) + 2, 3), np.nan)
cmap_BW[0] = np.zeros((3))  # Color for the largest bounding sphere
cmap_BW[1] = np.zeros((3))  # Color for the smallest inscribed sphere

if COLOR_DIMENSION == 2:
    # 2D case: get colors for the corner points from the reference stimulus
    cmap_temp = np.hstack((np.vstack([grid_trans[:, m, n] for m, n in idx_corner]), np.ones((len(idx_corner), 1))))
    cmap_BW[2:] = (color_thres_data.M_2DWToRGB @ cmap_temp.T).T
    y_ub = 24         # Upper bound for y-axis
    x_ub_BW = 0.14      # Upper bound for x-axis (Bures-Wasserstein distance)
    x_ub_LU = 3.5      # Upper bound for x-axis (Log-Euclidean distance)
    nBins_curves = 11   # Number of bins for curves 11
    nBins_hist = 33     # Number of bins for histograms 33
    fig_size = (4.5, 2.5)  # Figure size(2.5, 3.2)
    nyticks = 3
else:
    # 3D case: get colors for the corner points from the reference grid
    for c in range(len(idx_corner)):
        cmap_BW[c + 2] = [grid_trans[m] for m in idx_corner[c]]
    y_ub = 100
    x_ub_BW = 0.18
    x_ub_LU = 4
    nBins_curves = 11
    nBins_hist = 33
    fig_size = (2.5, 3.2)
    nyticks = 5
saveFig = False

# %% ------------------------------------------
# Plot Bures-Wasserstein distance
# ------------------------------------------
output_figDir_fits = input_fileDir_all[0].replace('DataFiles', 'FigFiles')
os.makedirs(output_figDir_fits, exist_ok=True)

BW_bins = np.linspace(0, x_ub_BW,nBins_curves)
BW_bins2 = np.linspace(0, x_ub_BW, nBins_hist)
# plotting starts from here
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams.update({'font.size': 13})

fig1, ax1 = plt.subplots(1,1, figsize = fig_size, dpi = 1024)
model_perf.plot_benchmark_similarity(ax1, model_perf.BW_benchmark, BW_bins, #ls = [':','--']+['-']*len(idx_corner), 
                                     lw = 2, cmap = cmap_BW)
#jitter = np.linspace(-0.002,0.002, len(idx_corner)+2),
model_perf.plot_similarity_metric_scores(ax1, model_perf.BW_distance, BW_bins2,
                                         y_ub = y_ub, cmap = cmap_t, alpha = 0.6,
                                         legend_labels = legend_str)

ax1.set_xlabel('Bures-Wasserstein distance')
ax1.set_ylabel('Frequency')
ax1.set_xticks(np.around(BW_bins[::3],2))
ax1.set_yticks(np.linspace(0, y_ub, nyticks))
ax1.set_xlim([0, x_ub_BW])
ax1.set_ylim([0, y_ub])
#ax1.legend(title = 'File')
plt.tight_layout()
figName1 = f"ModelPerformance_BuresWassersteinDistance_File1_{file_name_all[0][:-4]}.pdf"
fig1.savefig(os.path.join(output_figDir_fits, figName1))   

#%%  
LU_bins = np.linspace(0,x_ub_LU, nBins_curves)
LU_bins2 = np.linspace(0,x_ub_LU, nBins_hist)

fig2, ax2 = plt.subplots(1,1, figsize = fig_size, dpi = 256)
model_perf.plot_similarity_metric_scores(ax2, model_perf.LU_distance, LU_bins2,
                              y_ub = y_ub, cmap = cmap_t,
                              legend_labels = legend_str)
model_perf.plot_benchmark_similarity(ax2, model_perf.LU_benchmark, 
                                     LU_bins,cmap = cmap_BW, lw = 1,
                                     linestyle = [':','--']+['-']*len(idx_corner)) 
#jitter = np.linspace(-0.02,0.02, len(idx_corner)+2)
ax2.set_xlabel('Log Euclidean distance')
ax2.set_ylabel('Frequency')
ax2.set_xticks(np.around(LU_bins[::3],3))
ax2.set_yticks(np.linspace(0, y_ub, nyticks))
ax2.set_ylim([0, y_ub])
ax2.set_xlim([0, x_ub_LU])
ax2.legend(title = 'Noise level')
# figName2 = f"ModelPerformance_LogEuclideanDistance_{color_dimension}D{s_ell}_wbenchmark"+\
#     f"_nSims{nSims}_jitter{jitter}.pdf"
# full_path2 = os.path.join(fig_outputDir, figName2)
# plt.tight_layout()
# if saveFig: fig2.savefig(full_path2)  
plt.show()









