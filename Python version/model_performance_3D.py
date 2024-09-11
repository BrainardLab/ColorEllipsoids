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
from analysis.ellipses_tools import ellParams_to_covMat
from analysis.color_thres import color_thresholds
sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/Python version')
from analysis.model_performance import ModelPerformance
        
base_dir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'
fig_outputDir = base_dir+'ELPS_analysis/ModelPerformance_FigFiles/'

#%% ------------------------------------------
# Load data simulated using CIELab
# ------------------------------------------
varyingFactor = 'jitter'
jitter = [0.1, 0.3, 0.5]
fixedFactor = 'nSims'
nSims = 240 #
saveFig = False

nLevels = len(jitter)
path_str1 = base_dir +'ELPS_analysis/ModelFitting_DataFiles/3D_oddity_task/' #
data_load = []
    
for j in jitter:
    file_name_j = f'Fitted_isothreshold_ellipsoids_sim{nSims}perCond_samplingNearContour_'+\
                    f'jitter{j:01}_seed0_bandwidth0.005.pkl' #
    full_path_j = f"{path_str1}{file_name_j}"
    os.chdir(path_str1)
    #load data 
    with open(full_path_j, 'rb') as f:
        # Load the object from the file
        data_load_j = pickle.load(f)
        data_load.append(data_load_j)

#%% load ground truths
color_dimension = 3
# Create an instance of the class
color_thres_data = color_thresholds(color_dimension, base_dir + 'ELPS_analysis/')
# Load Wishart model fits
color_thres_data.load_CIE_data()
results3D = color_thres_data.get_data('results3D', dataset = 'CIE_data')  
stim3D = color_thres_data.get_data('stim3D', dataset = 'CIE_data')  
scaler_x1 =5

#%%
# Select specific points (corners) of the 3D space to focus on
idx_corner = [[0,0,0],[4,0,0],[0,4,0],[0,0,4],[4,4,0],[0,4,4],[4,0,4],[4,4,4]]
#[[x, y, z] for x in (0, 4) for y in (0, 4) for z in (0, 4)]

# Retrieve the covariance matrices at these corner points
covMat_corner = [ellParams_to_covMat(results3D['ellipsoidParams'][i][j][k]['radii']*scaler_x1,\
                results3D['ellipsoidParams'][i][j][k]['evecs']) \
                for i, j, k in idx_corner]

model_perf = ModelPerformance(color_dimension, results3D, stim3D, jitter)
model_perf.evaluate_model_performance(data_load, covMat_corner = covMat_corner)

model_perf.concatenate_benchamrks()

#%% plotting starts from here
cmap_BW = np.full((len(idx_corner)+2,3),np.nan)
cmap_BW[0] = np.zeros((3))
cmap_BW[1] = np.zeros((3))
for c in range(len(idx_corner)):
    cmap_BW[c+2] = [stim3D['grid_ref'][m] for m in idx_corner[c]]

legend_str = ['Low', 'Medium', 'High']
cmap_temp = plt.get_cmap('Dark2')
cmap_t = cmap_temp(np.linspace(0.3, 0.8, nLevels))[:, :3]

#%%
y_ub = 80
BW_bins = np.linspace(0,0.18,11)
# plotting starts from here
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams.update({'font.size': 10})

fig1, ax1 = plt.subplots(1,1, figsize = (8.2, 2.8), dpi = 256)
model_perf.plot_benchmark_similarity(ax1, model_perf.BW_benchmark, BW_bins,
                          jitter = np.linspace(-0.002,0.002, len(idx_corner)+2),
                          linestyle = [':','--']+['-']*len(idx_corner), 
                          lw = 1.5, cmap = cmap_BW)

BW_bins2 = np.linspace(0,0.18,44)
model_perf.plot_similarity_metric_scores(ax1, model_perf.BW_distance, BW_bins2,
                              y_ub = 80, cmap = cmap_t, alpha = 0.6,
                              legend_labels = legend_str)

ax1.set_xlabel('The Bures-Wasserstein distance')
ax1.set_ylabel('Frequency')
ax1.set_xticks(np.around(BW_bins[::2],2))
ax1.set_yticks(np.linspace(0, y_ub, 5))
ax1.set_xlim([0, 0.18])
ax1.set_ylim([0, y_ub])
ax1.legend(title = 'Level of noise')
plt.tight_layout()
figName1 = f"ModelPerformance_BuresWassersteinDistance_3Dellipsoids_wbenchmark_nSims{nSims}_jitter{jitter}.pdf"
full_path1 = os.path.join(fig_outputDir, figName1)
#if saveFig: fig1.savefig(full_path1)   

#%%  
x_ub = 4
LU_bins = np.linspace(0,x_ub,11)
LU_bins2 = np.linspace(0,x_ub,44)

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams.update({'font.size': 10})

fig2, ax2 = plt.subplots(1,1, figsize = (8.2, 2.8), dpi = 256)
model_perf.plot_similarity_metric_scores(ax2, model_perf.LU_distance, LU_bins2,
                              y_ub = y_ub, cmap = cmap_t,
                              legend_labels = legend_str)
model_perf.plot_benchmark_similarity(ax2, model_perf.LU_benchmark, 
                                     LU_bins,cmap = cmap_BW,\
                              linestyle = [':','--']+['-']*len(idx_corner), 
                          jitter = np.linspace(-0.02,0.02, len(idx_corner)+2),
                          lw = 1)

ax2.set_xlabel('Log Euclidean distance')
ax2.set_ylabel('Frequency')
ax2.set_xticks(np.around(LU_bins[::2],3))
ax2.set_yticks(np.linspace(0, y_ub, 5))
ax2.set_ylim([0, y_ub])
ax2.set_xlim([0,x_ub])
ax2.legend(title = 'Noise level')
figName2 = f"ModelPerformance_LogEuclideanDistance_3Dellipsoids_wbenchmark_nSims{nSims}_jitter{jitter}.pdf"
full_path2 = os.path.join(fig_outputDir, figName2)
plt.tight_layout()
#if saveFig: fig2.savefig(full_path2)  
plt.show()









