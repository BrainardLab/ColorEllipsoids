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
jitter  = [0.1, 0.3, 0.5]   # Different noise levels (jitter)
nLevels = len(jitter)      # Number of noise levels
nSims   = 240                # Number of simulations per condition
saveFig = False            # Whether to save the figures
color_dimension = 2        # Dimensionality of the color space (2D or 3D)

# Path to the directory containing the simulation data files
path_str1 = base_dir + f'ELPS_analysis/ModelFitting_DataFiles/{color_dimension}D_oddity_task/' 

# List to store loaded simulation data
data_load = []
    
# Loop through each jitter level and load corresponding simulation data
for j in jitter:
    if color_dimension == 2: 
        plane_2D = 'GB plane'  # For 2D simulations, select a specific plane (e.g., GB plane)
        s_ell = plane_2D       # Label used for file naming
    else:
        s_ell = 'ellipsoids'   # For 3D simulations, no specific plane
        plane_2D = None
    
    # Construct the file name based on the jitter level and other parameters
    file_name_j = f'Fitted_isothreshold_{s_ell}_sim{nSims}perCond_samplingNearContour_' +\
                  f'jitter{j:01}_seed0_bandwidth0.005_oddity.pkl'
    full_path_j = f"{path_str1}{file_name_j}"
    
    # Change directory and load the simulation data using pickle
    os.chdir(path_str1)
    with open(full_path_j, 'rb') as f:
        data_load_j = pickle.load(f)
        data_load.append(data_load_j) # Append the loaded data to the list

#%% load ground truths
# Create an instance of the color_thresholds class
color_thres_data = color_thresholds(color_dimension, 
                                    base_dir + 'ELPS_analysis/',
                                    plane_2D = plane_2D)

# Load CIE data for the ground truth ellipses/ellipsoids
color_thres_data.load_CIE_data()
CIE_results = color_thres_data.get_data(f'results{color_dimension}D',
                                        dataset = 'CIE_data')  
CIE_stim = color_thres_data.get_data(f'stim{color_dimension}D', 
                                     dataset = 'CIE_data')  
#scaler for the ellipse/ellipsoids 
scaler_x1 = 5

#%%
# Create an instance of ModelPerformance to evaluate model predictions
model_perf = ModelPerformance(color_dimension,
                              CIE_results, 
                              CIE_stim, 
                              jitter, 
                              plane_2D = plane_2D,
                              verbose = True)

# Select specific points (corners) in the 2D or 3D space for comparison
if color_dimension == 2:
    # For 2D, select specific corner points in the plane
    indices = [0, 4]
    idx_corner = [[i, j] for i in indices for j in indices]
    ellParams_slc = CIE_results['ellParams'][model_perf.plane_2D_idx]
    covMat_corner = []
    for i,j in idx_corner:
        _, _, a_ij, b_ij, R_ij = ellParams_slc[i][j]
        radii_ij = np.array([a_ij, b_ij])*scaler_x1
        eigvec_ij = rotAngle_to_eigenvectors(R_ij)
        covMat_corner_ij = ellParams_to_covMat(radii_ij, eigvec_ij) 
        covMat_corner.append(covMat_corner_ij)
else:
    # For 3D, select corner points in the 3D space
    # idx_corner = [[0,0,0],[4,0,0],[0,4,0],[0,0,4],
    #               [4,4,0],[0,4,4],[4,0,4],[4,4,4]]
    indices = [0, 4]
    idx_corner = [[i, j, k] for i in indices for j in indices for k in indices]

    # Retrieve covariance matrices at these corner points
    covMat_corner = [ellParams_to_covMat(CIE_results['ellipsoidParams'][i][j][k]['radii']*scaler_x1,\
                    CIE_results['ellipsoidParams'][i][j][k]['evecs']) for i, j, k in idx_corner]
# Evaluate the model performance using the loaded data and corner points
model_perf.evaluate_model_performance(data_load, covMat_corner = covMat_corner)
# Concatenate benchmark results for plotting
model_perf.concatenate_benchamrks()

# %% ------------------------------------------
# Plotting specifics
# ------------------------------------------
legend_str = ['Low', 'Medium', 'High']  # Legend labels for different noise levels
cmap_temp = plt.get_cmap('Dark2')       # Colormap for the plots
cmap_t = cmap_temp(np.linspace(0.3, 0.8, nLevels))[:, :3]

# Prepare colormap for corner points and benchmarks
cmap_BW = np.full((len(idx_corner) + 2, 3), np.nan)
cmap_BW[0] = np.zeros((3))  # Color for the largest bounding sphere
cmap_BW[1] = np.zeros((3))  # Color for the smallest inscribed sphere

if color_dimension == 2:
    # 2D case: get colors for the corner points from the reference stimulus
    cmap_BW[2:] = np.vstack([CIE_stim['ref_points'][model_perf.plane_2D_idx][:, m, n] 
                             for m, n in idx_corner])
    y_ub = 16          # Upper bound for y-axis
    x_ub_BW = 0.1      # Upper bound for x-axis (Bures-Wasserstein distance)
    x_ub_LU = 3.5      # Upper bound for x-axis (Log-Euclidean distance)
    nBins_curves = 11   # Number of bins for curves
    nBins_hist = 33     # Number of bins for histograms
    fig_size = (2.5, 3.2)  # Figure size
    nyticks = 5
else:
    # 3D case: get colors for the corner points from the reference grid
    for c in range(len(idx_corner)):
        cmap_BW[c + 2] = [CIE_stim['grid_ref'][m] for m in idx_corner[c]]
    y_ub = 100
    x_ub_BW = 0.18
    x_ub_LU = 4
    nBins_curves = 11
    nBins_hist = 33
    fig_size = (2.5, 3.2)
    nyticks = 5
saveFig = True

# %% ------------------------------------------
# Plot Bures-Wasserstein distance
# ------------------------------------------
BW_bins = np.linspace(0, x_ub_BW,nBins_curves)
BW_bins2 = np.linspace(0, x_ub_BW, nBins_hist)
# plotting starts from here
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams.update({'font.size': 10})

fig1, ax1 = plt.subplots(1,1, figsize = fig_size, dpi = 256)
model_perf.plot_benchmark_similarity(ax1, model_perf.BW_benchmark, BW_bins,
                                     linestyle = [':','--']+['-']*len(idx_corner), 
                                     lw = 1, cmap = cmap_BW)
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
ax1.legend(title = 'Level of noise')
plt.tight_layout()
figName1 = f"ModelPerformance_BuresWassersteinDistance_{color_dimension}D{s_ell}_"+\
    f"wbenchmark_nSims{nSims}_jitter{jitter}.pdf"
full_path1 = os.path.join(fig_outputDir, figName1)
if saveFig: fig1.savefig(full_path1)   

#%%  
LU_bins = np.linspace(0,x_ub_LU, nBins_curves)
LU_bins2 = np.linspace(0,x_ub_LU, nBins_hist)

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams.update({'font.size': 10})

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
ax2.set_xticks(np.around(LU_bins[::2],3))
ax2.set_yticks(np.linspace(0, y_ub, nyticks))
ax2.set_ylim([0, y_ub])
ax2.set_xlim([0, x_ub_LU])
ax2.legend(title = 'Noise level')
figName2 = f"ModelPerformance_LogEuclideanDistance_{color_dimension}D{s_ell}_wbenchmark"+\
    f"_nSims{nSims}_jitter{jitter}.pdf"
full_path2 = os.path.join(fig_outputDir, figName2)
plt.tight_layout()
if saveFig: fig2.savefig(full_path2)  
plt.show()









