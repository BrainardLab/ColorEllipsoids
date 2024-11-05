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
#import jax._src.interpreters.pxla
#dill._dill._reverse_typemap['shard_arg'] = jax._src.interpreters.pxla.shard_arg_handlers
import numpy as np
sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids')
from analysis.ellipses_tools import ellParams_to_covMat, rotAngle_to_eigenvectors
from analysis.color_thres import color_thresholds
sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/Python version')
from analysis.model_performance import ModelPerformance

##this new part below is needed since I updated jax
class CustomUnpickler(pickled.Unpickler):
    def find_class(self, module, name):
        # Skip shard_arg by returning a dummy function
        if name == 'shard_arg':
            return lambda *args, **kwargs: None  # Return a dummy function or None
        return super().find_class(module, name)
        
# Define base directory and figure output directory
base_dir = '/Volumes/T9/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'
fig_outputDir = base_dir+'ELPS_analysis/ModelComparison_FigFiles/2D_oddity_task/'

#%% ------------------------------------------
# Load data simulated using CIELab
# ------------------------------------------
ndims = 3
if ndims == 2:
    slc_color  = 'GB plane'
    jitter    = 0.3
    seed_list = list(range(10))
    nSims     = [480, 320, 160, 80]     # Number of simulations per reference
    nRefs     = 25
else:
    slc_color = 'ellipsoids'
    jitter    = 0.3
    seed_list = list(range(10))
    nSims     = [4800, 1280, 800, 200]     # Number of simulations per reference,1280, 800
    nRefs     = 125
nSims_total = np.array(nSims)*nRefs
nLevels   = len(nSims)
saveFig   = False            # Whether to save the figures

# Path to the directory containing the simulation data files
path_str1 = base_dir + f'ELPS_analysis/ModelFitting_DataFiles/{ndims}D_oddity_task_indvEll/' 
path_str2 = base_dir + f'ELPS_analysis/ModelFitting_DataFiles/{ndims}D_oddity_task/'

# List to store loaded simulation data
data_load_indvEll = []
    
# Loop through each jitter level and load corresponding simulation data
for k in nSims:        
    data_load_indvEll_k = []
    for l in seed_list:            
        # Construct the file name based on the jitter level and other parameters
        file_name_jj = f'Fitted_isothreshold_{slc_color}_sim{k}perCond_samplingNearContour_' +\
                      f'jitter{jitter}_seed{l}_oddity_indvEll.pkl'
        full_path_jj = f"{path_str1}{file_name_jj}"
        
        # Change directory and load the simulation data using pickle
        os.chdir(path_str1)
        with open(full_path_jj, 'rb') as f:
            data_load_ll = CustomUnpickler(f).load()
            data_load_indvEll_k.append(data_load_ll) # Append the loaded data to the list
    data_load_indvEll.append(data_load_indvEll_k) 
    
#load one from the Wishart fits
data_load_Wishart= []
if ndims == 2:
    nSims_Wishart = 60
else:
    nSims_Wishart = 160
nSims_total_Wishart = nSims_Wishart * nRefs
for l in seed_list:
    # Construct the file name based on the jitter level and other parameters
    file_name_jj = f'Fitted_isothreshold_{slc_color}_sim{nSims_Wishart}perCond_'+\
        f'samplingNearContour_jitter{jitter}_seed{l}_bandwidth0.005_oddity.pkl'
    full_path_jj = f"{path_str2}{file_name_jj}"
    # Change directory and load the simulation data using pickle
    os.chdir(path_str1)
    with open(full_path_jj, 'rb') as f:
        data_load_Wishart_l = CustomUnpickler(f).load()
    data_load_Wishart.append(data_load_Wishart_l)

#%% load ground truths
# Create an instance of the color_thresholds class
if ndims == 2:
    plane_2D = slc_color
else:
    plane_2D = None
color_thres_data = color_thresholds(ndims, base_dir + 'ELPS_analysis/', plane_2D = plane_2D)

# Load CIE data for the ground truth ellipses/ellipsoids
color_thres_data.load_CIE_data()
CIE_results = color_thres_data.get_data(f'results{ndims}D',  dataset = 'CIE_data')  
CIE_stim = color_thres_data.get_data(f'stim{ndims}D', dataset = 'CIE_data')  
#scaler for the ellipse/ellipsoids 
scaler_x1 = 5

#%%
# Create an instance of ModelPerformance to evaluate model predictions
model_perf = ModelPerformance(ndims,
                              CIE_results, 
                              CIE_stim, 
                              list(range(len(seed_list))), 
                              plane_2D = plane_2D,
                              verbose = True)
indices = [0, 4]
if ndims == 2:
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
    idx_corner = [[i, j, k] for i in indices for j in indices for k in indices]

    # Retrieve covariance matrices at these corner points
    covMat_corner = [ellParams_to_covMat(CIE_results['ellipsoidParams'][i][j][k]['radii']*scaler_x1,\
                    CIE_results['ellipsoidParams'][i][j][k]['evecs']) for i, j, k in idx_corner]

# Evaluate the model performance using the loaded data and corner points
BW_distance_output_indvEll = np.full((nLevels, len(seed_list), *np.tile(5, ndims)), np.nan)
for i in range(nLevels):
    model_perf.evaluate_model_performance(data_load_indvEll[i],covMat_corner = covMat_corner)
    BW_distance_output_indvEll[i] = model_perf.BW_distance
    
# Evaluate the model performance using the loaded data and corner points
model_perf.evaluate_model_performance(data_load_Wishart)
BW_distance_output_Wishart = model_perf.BW_distance

# %% ------------------------------------------
# Plot Bures-Wasserstein distance
# ------------------------------------------
# plotting starts from here
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams.update({'font.size': 10})
if ndims == 2:
    cmap_BW = np.vstack([CIE_stim['ref_points'][model_perf.plane_2D_idx][:, m, n] 
                                 for m, n in idx_corner])
else:
    cmap_BW = np.vstack([CIE_stim['ref_points'][m, n, o] 
                                 for m, n, o in idx_corner])    

axis1_merge = tuple(list(range(1,ndims+2)))
BW_distance_output_median_indvEll = np.median(BW_distance_output_indvEll, axis = axis1_merge)
# Create the error for yerr as a tuple of arrays for asymmetrical error
yerr_indvEll = np.array([BW_distance_output_median_indvEll - \
                            np.min(BW_distance_output_indvEll, axis = axis1_merge), 
                 np.max(BW_distance_output_indvEll, axis = axis1_merge) - \
                     BW_distance_output_median_indvEll])
    
BW_distance_output_median_Wishart = np.median(BW_distance_output_Wishart)
# Create the error for yerr as a tuple of arrays for asymmetrical error
yerr_Wishart = np.array([BW_distance_output_median_Wishart - np.min(BW_distance_output_Wishart), 
                 np.max(BW_distance_output_Wishart) - BW_distance_output_median_Wishart])
BW_distance_circle_median = np.median(model_perf.BW_distance_minEigval)
BW_distance_corner_median = np.median(model_perf.BW_distance_corner, axis = axis1_merge[:-1])

#%%plotting
fig1, ax1 = plt.subplots(1,1, figsize = (2.5, 3.75), dpi = 256) #; format 1: (3.5, 2.2); format 2:  (3.2, 2.2)
x_left= 0
x_right = np.linspace(2,nLevels, nLevels)
y_ub = 0.14#0.12
ax1.plot([-1, nLevels+2], [BW_distance_circle_median, BW_distance_circle_median],
         c = 'k',ls = '-',lw = 2, alpha = 0.8)
for i in range(len(covMat_corner)):
    ax1.plot([-1, nLevels+2], np.array([1,1])*BW_distance_corner_median[i],
             c = cmap_BW[i],ls = '-',lw = 2, alpha = 0.8)
#Wishart
ax1.errorbar(x_left, BW_distance_output_median_Wishart, 
             yerr=yerr_Wishart.reshape(2, 1),
            fmt='o', capsize=0, c = 'k', 
            marker = 'o', markersize = 8, lw = 2)
for i in range(nLevels):
    ax1.errorbar(x_right[i], BW_distance_output_median_indvEll[i], 
                 yerr=yerr_indvEll[:, i].reshape(2, 1),
                fmt='o', capsize=0, c = np.array([0,0,0])+i*0.2, 
                marker = 'D', markersize = 6, lw = 2)
ax1.plot([1,1],[0,y_ub],ls = '--', lw=0.5, c = 'k')
ax1.set_xticks(np.hstack([x_left, x_right]))
ax1.set_xticklabels([f'{nSims_total_Wishart}'] + [str(s) for s in nSims_total], rotation= 45)
ax1.set_xlabel('Total number of trial')
#ax1.set_title(slc_color)
ax1.set_title('RGB cube')
ax1.set_xlim([-1, nLevels + 1])
ax1.set_yticks(np.linspace(0,y_ub,3))
ax1.set_ylim([0,y_ub])
ax1.set_ylabel('BW distance')
plt.tight_layout()
figName1 = f"ModelPerformance_BuresWassersteinDistance_{ndims}D{slc_color}_"+\
    f"wbenchmark_jitter{jitter}_Wishart_vs_IndvEll_format2.pdf"
full_path1 = os.path.join(fig_outputDir, figName1)
if saveFig: fig1.savefig(full_path1)   
