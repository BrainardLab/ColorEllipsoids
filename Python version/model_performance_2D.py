#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 15:32:27 2024

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
        
base_dir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'
fig_outputDir = base_dir+'ELPS_analysis/ModelPerformance_FigFiles/'

#%% ------------------------------------------
# Load data simulated using CIELab
# ------------------------------------------
varyingFactor = 'jitter'
jitter = [0.1, 0.3, 0.5]
fixedFactor = 'nSims'
nSims = 240
plane_2D = 'GB plane'
saveFig = False
nLevels = len(jitter)
path_str1 = base_dir +'ELPS_analysis/ModelFitting_DataFiles/2D_oddity_task/'
data_load = []
    
for j in jitter:
    file_name_j = f'Fitted_isothreshold_{plane_2D}_sim{nSims}perCond_' +\
                f'samplingNearContour_jitter{j:01}_seed0_bandwidth0.005_oddity.pkl'
    full_path_j = f"{path_str1}{file_name_j}"
    os.chdir(path_str1)
    #load data 
    with open(full_path_j, 'rb') as f:
        # Load the object from the file
        data_load_j = pickle.load(f)
        data_load.append(data_load_j)

#%% load ground truths
color_dimension = 2
# Create an instance of the class
color_thres_data = color_thresholds(color_dimension, base_dir + 'ELPS_analysis/', plane_2D = plane_2D)
# Load Wishart model fits
color_thres_data.load_CIE_data()
results2D = color_thres_data.get_data('results2D', dataset = 'CIE_data')  
stim2D = color_thres_data.get_data('stim2D', dataset = 'CIE_data')  
ref_size = stim2D['nGridPts_ref'] 
scaler_x1 = 5
            
model_perf = ModelPerformance(color_dimension, results2D, stim2D, jitter, plane_2D = plane_2D)

# Select specific points (corners) of the 3D space to focus on
indices = [0, 2, 4]
idx_corner = [[i, j] for i in indices for j in indices]
ellParams_slc = results2D['ellParams'][model_perf.plane_2D_idx]
covMat_corner = []
for i,j in idx_corner:
    _, _, a_ij, b_ij, R_ij = ellParams_slc[i][j]
    """ 
    Note in the following code, we have to scale the radii from results2D by 10,
    because when those ellipses were fit, they were originally bounded by 
    [0, 1], but then were rescaled by 2 to fit in [-1, 1], and all the data
    were scaled up by 5 to make the fitting easier. 
    """
    radii_ij = np.array([a_ij, b_ij])*scaler_x1
    eigvec_ij = rotAngle_to_eigenvectors(R_ij)
    covMat_corner_ij = ellParams_to_covMat(radii_ij, eigvec_ij) 
    covMat_corner.append(covMat_corner_ij)
model_perf.evaluate_model_performance(data_load, covMat_corner = covMat_corner)

model_perf.concatenate_benchamrks()

#%% plotting starts from here
cmap_BW = np.full((len(idx_corner)+2,3),np.nan)
cmap_BW[0] = np.zeros((3))
cmap_BW[1] = np.zeros((3))
cmap_BW[2::] = np.vstack([stim2D['ref_points'][model_perf.plane_2D_idx][:,m,n] for m,n in idx_corner])
cmap_temp = plt.get_cmap('Dark2')
cmap_t = cmap_temp(np.linspace(0.3, 0.8, nLevels))[:, :3]
  
x_ub = 0.1
y_ub = 25
BW_bins = np.linspace(0,x_ub,11)
BW_bins2 = np.linspace(0, x_ub,33)
legend_str = ['Low', 'Medium', 'High']

# plotting starts from here
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams.update({'font.size': 10})

fig2, ax1 = plt.subplots(1,1, figsize = (3.2, 2.5))
model_perf.plot_similarity_metric_scores(ax1, model_perf.BW_distance, BW_bins2,
                              y_ub = y_ub, cmap = cmap_t,
                              legend_labels = [str(jitter[i])+legend_str[i] \
                                               for i in range(nLevels)])
model_perf.plot_benchmark_similarity(ax1, model_perf.BW_benchmark, BW_bins,cmap = cmap_BW,
                          linestyle = [':','--']+['-']*len(idx_corner),
                          jitter = np.linspace(-0.002,0.002, len(idx_corner)+2), 
                          lw = 1)
ax1.set_xticks(np.linspace(0,x_ub,3))
ax1.set_xlim(right=x_ub)
ax1.set_ylim([0, y_ub])
ax1.set_yticks(np.linspace(0,y_ub,6))
ax1.set_xlabel('The Bures-Wasserstein distance')
ax1.set_ylabel('Frequency')
ax1.legend(title = 'Level of noise')
ax1.set_title(plane_2D, fontsize = 10)
figName2 = f"ModelPerformance_BuresWassersteinDistance_2Dellipses_{plane_2D}_nSims{nSims}_jitter{jitter}_seed0.pdf"
full_path2 = os.path.join(fig_outputDir, figName2)
plt.tight_layout()
#if saveFig: fig2.savefig(full_path2)
plt.show()

#%%
x_ub = 3.5
LU_bins = np.linspace(0,x_ub,11)
LU_bins2 = np.linspace(0,x_ub,33)

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams.update({'font.size': 10})

fig3, ax2 = plt.subplots(1,1, figsize = (3.2, 2.5), dpi = 256)
model_perf.plot_similarity_metric_scores(ax2, model_perf.LU_distance, LU_bins2,
                              y_ub = y_ub, cmap = cmap_t,
                              legend_labels = legend_str)
model_perf.plot_benchmark_similarity(ax2, model_perf.LU_benchmark,
                                     LU_bins,cmap = cmap_BW,
                          linestyle = [':','--']+['-']*len(idx_corner), 
                          jitter = np.linspace(-0.02,0.02, len(idx_corner)+2),
                          lw = 1)

ax2.set_xlabel('Log Euclidean distance')
ax2.set_ylabel('Frequency')
ax2.set_xticks(np.around(LU_bins[::2],3))
ax2.set_yticks(np.linspace(0, y_ub, 6))
ax2.set_ylim([0, y_ub])
ax2.set_xlim([0, x_ub])
ax2.set_title(plane_2D, fontsize = 10)
ax2.legend(title = 'Noise level')
figName3 = f"ModelPerformance_LogEuclideanDistance_2Dellipses_{plane_2D}_nSims{nSims}_jitter{jitter}_seed0.pdf"
full_path3 = os.path.join(fig_outputDir, figName3)
plt.tight_layout()
#if saveFig: fig3.savefig(full_path3)  
plt.show()




