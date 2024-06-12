#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 00:39:55 2024

@author: fangfang
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
import sys
sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids')
sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/Python version')
fig_outputDir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                        'ELPS_analysis/ModelPerformance_FigFiles/'

#%% ------------------------------------------
# Load data simulated using CIELab
# ------------------------------------------
import os
import pickle
import numpy as np
from scipy.linalg import sqrtm

nSims     = 240
bandwidth = 0.005
jitter    = [0.1, 0.3, 0.5]
path_str1 = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
            'ELPS_analysis/ModelFitting_DataFiles/'
data_load = []
for j in jitter:
    file_name_j = 'Fitted_isothreshold_ellipsoids_sim'+str(nSims) +\
                'perCond_samplingNearContour_jitter'+str(j)+'_bandwidth' +\
                str(bandwidth) + '.pkl'
    full_path_j = f"{path_str1}{file_name_j}"
    os.chdir(path_str1)
    #load data 
    with open(full_path_j, 'rb') as f:
        # Load the object from the file
        data_load_j = pickle.load(f)
        data_load.append(data_load_j)

#%% load ground truths
path_str2 = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
        'ELPS_analysis/Simulation_DataFiles/'
file_name2 = 'Isothreshold_ellipsoid_CIELABderived.pkl'
full_path2 = f"{path_str2}{file_name2}"
os.chdir(path_str2)

#Here is what we do if we want to load the data
with open(full_path2, 'rb') as f:
    # Load the object from the file
    data_load2 = pickle.load(f)
_, stim3D, results3D = data_load2[0], data_load2[1], data_load2[2]

#%% 
def compute_normalized_Bures_similarity(M1, M2):
    # Compute the product inside the trace
    inner_product = sqrtm(sqrtm(M1) @ M2 @ sqrtm(M1))  
    # Calculate the trace of the product
    trace_value = np.trace(inner_product)    
    # Normalize by the geometric mean of the traces of M1 and M2
    normalization_factor = np.sqrt(np.trace(M1) * np.trace(M2))    
    # Calculate NBS
    NBS = trace_value / normalization_factor    
    return NBS
            
#%% evalutate model performance
"""
calculate the Bures-Wasserstein distance
"""
#model predictions
covMat_gt        = np.full(stim3D['ref_points'].shape[0:3]+(3,3,), np.nan)
covMat_modelPred = np.full((len(jitter),)+stim3D['ref_points'].shape[0:3]+(3,3,), np.nan)
NBW_distance     = np.full((len(jitter),)+ stim3D['ref_points'].shape[0:3], np.nan)
NBW_distance_maxEigval = np.full(stim3D['ref_points'].shape[0:3], np.nan)
             
#% the Bures-Wasserstein distance
for ii in range(stim3D['ref_points'].shape[0]):
    for jj in range(stim3D['ref_points'].shape[1]):
        for kk in range(stim3D['ref_points'].shape[2]):
            #ground truth
            eigVec_gt_jiijjkk = results3D['ellipsoidParams'][ii,jj,kk]['evecs']
            eigVal_gt_jiijjkk = results3D['ellipsoidParams'][ii,jj,kk]['radii']
            eigVal_gt_max_jiijjkk = np.ones((3))*np.max(eigVal_gt_jiijjkk)
            
            covMat_gt[ii,jj,kk] = eigVec_gt_jiijjkk @ np.diag(eigVal_gt_jiijjkk)*2 @\
                eigVec_gt_jiijjkk.T
            
            covMat_max = np.eye(3) @ np.diag(eigVal_gt_max_jiijjkk) @ np.eye(3).T
            NBW_distance_maxEigval[ii,jj,kk] = compute_normalized_Bures_similarity(\
                covMat_gt[ii,jj,kk],covMat_max)
                
            #model predictions
            for l in range(len(jitter)):   
                eigVec_jiijjkk = data_load[l]['params_ellipsoids'][ii][jj][kk]['evecs']
                eigVal_jiijjkk = data_load[l]['params_ellipsoids'][ii][jj][kk]['radii']
                
                covMat_modelPred[l,ii,jj,kk] = eigVec_jiijjkk @ np.diag(eigVal_jiijjkk) @ eigVec_jiijjkk.T
                NBW_distance[l,ii,jj,kk] = compute_normalized_Bures_similarity(\
                    covMat_gt[ii,jj,kk],covMat_modelPred[l,ii,jj,kk])
                    
                    
#%%
y_ub = 35
lgd_list = ['small', 'medium', 'large']
cmap = np.array([[247,152,29],[75,40,73],[107,142,35]])/255
fig, ax = plt.subplots(1, 1, figsize=(7,4.5))
plt.rcParams['figure.dpi'] = 250
for j in range(len(jitter)):
    ax.hist(NBW_distance[j].flatten(), bins = 50, range = [0.99,1], color = cmap[j],\
            alpha = 0.6, edgecolor = [1,1,1], label = str(jitter[j]) + ' ('+\
            lgd_list[j]+' noise)')
    ax.plot([np.median(NBW_distance[j].flatten()),np.median(NBW_distance[j].flatten())],\
             [0,y_ub],color = cmap[j], linestyle = '--')
ax.grid(True, alpha=0.3)
ax.legend(title = 'Amount of jitter added to\nsampled comparison stimuli')
ax.set_ylim([0,y_ub])
ax.set_xlabel('The normalized Bures similarity')
ax.set_ylabel('Frquency')
full_path = os.path.join(fig_outputDir, 'ModelPerformance_BuresWassersteinDistance_3Dellipsoids_jitters.png')
#fig.savefig(full_path)          

#%%
fig, ax = plt.subplots(1, 1, figsize=(7,4.5))
plt.rcParams['figure.dpi'] = 250
ax.hist(NBW_distance_maxEigval.flatten(), bins = 50, range = [0.93,0.97], \
        color = "grey", alpha = 0.6, edgecolor = [1,1,1], label = 'Sphere with maximum eigenvalues')
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_ylim([0,y_ub])
ax.set_xlabel('The normalized Bures similarity')
ax.set_ylabel('Frquency')




