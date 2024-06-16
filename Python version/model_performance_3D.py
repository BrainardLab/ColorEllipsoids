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

nSims     = 120
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
ref_size = stim3D['nGridPts_ref']

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

#compute mean radii
idx_corner = [[0,0,0],[4,0,0],[0,4,0],[0,0,4],[4,4,0],[0,4,4],[4,0,4],[4,4,4]]
covMat_corner = [results3D['ellipsoidParams'][i][j][k]['evecs'] @ \
    np.diag(results3D['ellipsoidParams'][i][j][k]['radii']) * 2 @ \
    results3D['ellipsoidParams'][i][j][k]['evecs'].T for i, j, k in idx_corner]
NBW_distance_corner = np.full((8,)+NBW_distance_maxEigval.shape, np.nan)
             
#% the Bures-Wasserstein distance
for ii in range(ref_size):
    for jj in range(ref_size):
        for kk in range(ref_size):
            #ground truth
            eigVec_gt_jiijjkk = results3D['ellipsoidParams'][ii,jj,kk]['evecs']
            eigVal_gt_jiijjkk = results3D['ellipsoidParams'][ii,jj,kk]['radii']
            eigVal_gt_max_jiijjkk = np.ones((3))*np.max(eigVal_gt_jiijjkk)
            
            covMat_gt[ii,jj,kk] = eigVec_gt_jiijjkk @ np.diag(eigVal_gt_jiijjkk)*2 @\
                eigVec_gt_jiijjkk.T
            
            covMat_max = np.eye(3) @ np.diag(eigVal_gt_max_jiijjkk) @ np.eye(3).T
            NBW_distance_maxEigval[ii,jj,kk] = compute_normalized_Bures_similarity(\
                covMat_gt[ii,jj,kk],covMat_max)      
            
            for m in range(len(idx_corner)):
                NBW_distance_corner[m,ii,jj,kk] = compute_normalized_Bures_similarity(\
                    covMat_corner[m],covMat_gt[ii,jj,kk])
                
            #model predictions
            for l in range(len(jitter)):   
                eigVec_jiijjkk = data_load[l]['params_ellipsoids'][ii][jj][kk]['evecs']
                eigVal_jiijjkk = data_load[l]['params_ellipsoids'][ii][jj][kk]['radii']
                
                covMat_modelPred[l,ii,jj,kk] = eigVec_jiijjkk @ np.diag(eigVal_jiijjkk) @ eigVec_jiijjkk.T
                NBW_distance[l,ii,jj,kk] = compute_normalized_Bures_similarity(\
                    covMat_gt[ii,jj,kk],covMat_modelPred[l,ii,jj,kk])
                    
                    
#%%
y_ub = 25
lgd_list = ['small', 'medium', 'large']
cmap = np.array([[247,152,29],[75,40,73],[107,142,35]])/255
fig, ax = plt.subplots(1, 1, figsize=(5,4.5))
plt.rcParams['figure.dpi'] = 250
bin_edges = np.arange(0.9802,1.0003,0.0004) #0.9899, 0.9919, 0.9939, np.arange(0.9899,1.0002,0.0002)
for j in range(len(jitter)):
    ax.hist(NBW_distance[j].flatten(), bins = bin_edges, color = cmap[j],\
            alpha = 0.6, edgecolor = [1,1,1], label = ' ('+ lgd_list[j]+' noise)')#str(nSims[j]) +' trials'
    ax.plot([np.median(NBW_distance[j].flatten()),np.median(NBW_distance[j].flatten())],\
             [0,y_ub],color = cmap[j], linestyle = '--')
ax.grid(True, alpha=0.3)
ax.legend(title = 'Amount of jitter added to\nsampled comparison stimuli')
ax.set_xticks(np.arange(0.98,1,0.005))
ax.set_xticklabels(np.arange(0.98,1,0.005))
ax.set_ylim([0,y_ub])
ax.set_xlabel('The normalized Bures similarity')
ax.set_ylabel('Frquency')
full_path = os.path.join(fig_outputDir, 'ModelPerformance_BuresWassersteinDistance'+\
                         '_3Dellipsoids_jitter_trial'+str(nSims)+'.png')
fig.savefig(full_path)          

#%%
y_ub = 80
fig, ax = plt.subplots(1, 1, figsize=(5,4.5))
plt.rcParams['figure.dpi'] = 250
bin_edges = np.arange(0.895,1.01,0.01)
counts, bin_edges = np.histogram(NBW_distance_maxEigval.flatten(), bins=bin_edges)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
ax.plot(bin_centers, counts,  color=[0,0,0])
for m in range(len(idx_corner)):
    counts_m,_ = np.histogram(NBW_distance_corner[m].flatten(), bins=bin_edges)
    ax.plot(bin_centers, counts_m,  color=[stim3D['grid_ref'][n] for n in idx_corner[m]])
ax.grid(True, alpha=0.3)
ax.set_yticks(np.linspace(0,80,5))
ax.set_ylim([0,y_ub])
ax.set_xlabel('The normalized Bures similarity')
ax.set_ylabel('Frquency')
full_path = os.path.join(fig_outputDir, 'ModelPerformance_BuresWassersteinDistance_3Dellipsoids_benchmark.png')
#fig.savefig(full_path)    



