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
sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/Python version')
from analysis.model_performance import compute_Bures_Wasserstein_distance, \
    plot_similarity_metric_scores, plot_benchmark_similarity, log_operator_norm_distance
        
base_dir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'
fig_outputDir = base_dir+'ELPS_analysis/ModelPerformance_FigFiles/'

#%% ------------------------------------------
# Load data simulated using CIELab
# ------------------------------------------

varyingFactor = 'jitter'
varyingLevels = [0.1, 0.3, 0.5]
fixedFactor = 'nSims'
fixedLevel = 240
plane_2D = 'GB plane'
saveFig = False

nLevels = len(varyingLevels)
path_str1 = base_dir +'ELPS_analysis/ModelFitting_DataFiles/2D_oddity_task/'
data_load = []
    
if fixedFactor == 'jitter':
    nSims     = varyingLevels #[240, 160, 120] #[240]
    jitter    = fixedLevel #0.1#[0.1, 0.3, 0.5]

    for j in varyingLevels:
        if  varyingFactor == 'nSims_total':
            file_name_j = f'Fitted_isothreshold_{plane_2D}_samplingRandom_' +\
                        f'wFittedW_jitter{jitter}_nSims{j}total.pkl'
        elif varyingFactor == 'nSims': #default is per condition
            file_name_j = f'Fitted_isothreshold_{plane_2D}_sim{j}perCond_samplingNearContour_' +\
                        f'jitter{jitter}_bandwidth0.005_seed0.pkl'
        full_path_j = f"{path_str1}{file_name_j}"
        os.chdir(path_str1)
        #load data 
        with open(full_path_j, 'rb') as f:
            # Load the object from the file
            data_load_j = pickle.load(f)
            data_load.append(data_load_j)

elif varyingFactor == 'jitter' and fixedFactor == 'nSims':
    nSims     = fixedLevel #[240, 160, 120] #[240]
    jitter    = varyingLevels #0.1#[0.1, 0.3, 0.5]
    for j in varyingLevels:
        # file_name_j = f'Fitted_isothreshold_{plane_2D}_samplingRandom_' +\
        #             f'wFittedW_jitter{j:01}_nSims{nSims}total.pkl'
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
path_str2 = base_dir + 'ELPS_analysis/Simulation_DataFiles/'
file_name2 = 'Isothreshold_contour_CIELABderived_fixedVal0.5.pkl'
full_path2 = f"{path_str2}{file_name2}"
os.chdir(path_str2)

#Here is what we do if we want to load the data
with open(full_path2, 'rb') as f:
    # Load the object from the file
    data_load2 = pickle.load(f)
_, stim2D, results2D = data_load2[0], data_load2[1], data_load2[2]
ref_size = stim2D['nGridPts_ref'] 
        
#%% evalutate model performance
"""
Initialize arrays to store covariance matrices and distance metrics for ground 
truth and predictions
covMat_gt: Ground truth covariance matrices
covMat_modelPred: model-predicted covariance matrices
BW_distance: Bures-Wasserstein distance for each simulation
BW_distance_maxEigval: Max eigenvalue-based Bures-Wasserstein distance
BW_distance_minEigval: Min eigenvalue-based Bures-Wasserstein distance
LU_distance: log Euclidean distance

"""
base_shape = stim2D['ref_points'].shape[2:]
covMat_gt        = np.full(base_shape+(2,2,), np.nan)
covMat_modelPred = np.full((nLevels,)+ base_shape+(2,2,), np.nan)
BW_distance      = np.full((nLevels,)+ base_shape, np.nan)
BW_distance_maxEigval   = np.full(base_shape, np.nan)
BW_distance_minEigval   = np.full(base_shape, np.nan)
LU_distance     = np.full(BW_distance.shape, np.nan)

# Select specific points (corners) of the 3D space to focus on
indices = [0, 2, 4]
idx_corner = [[i, j] for i in indices for j in indices]
# Retrieve the covariance matrices at these corner points
plane_2D_dict = {'GB plane': 0, 'RB plane': 1, 'RG plane': 2}
plane_2D_idx  = plane_2D_dict[plane_2D]

ellParams_slc = results2D['ellParams'][plane_2D_idx]
covMat_corner = []
for i,j in idx_corner:
    _, _, a_ij, b_ij, R_ij = ellParams_slc[i][j]
    """ 
    Note in the following code, we have to scale the radii from results2D by 10,
    because when those ellipses were fit, they were originally bounded by 
    [0, 1], but then were rescaled by 2 to fit in [-1, 1], and all the data
    were scaled up by 5 to make the fitting easier. 
    """
    radii_ij = np.array([a_ij, b_ij])*10
    eigvec_ij = rotAngle_to_eigenvectors(R_ij)
    covMat_corner_ij = ellParams_to_covMat(radii_ij, eigvec_ij) 
    covMat_corner.append(covMat_corner_ij)

# Initialize arrays for corner-based metrics
BW_distance_corner = np.full((len(idx_corner),)+BW_distance_maxEigval.shape, np.nan)
LU_distance_corner = np.full(BW_distance_corner.shape, np.nan)
             
# Loop through each reference point in the 3D space
for ii in range(ref_size):
    for jj in range(ref_size):
        _, _, a_iijj, b_iijj, R_iijj = ellParams_slc[ii][jj]
        # Retrieve eigenvalues and eigenvectors for the ground truth ellipsoids
        radii_gt_iijj  = np.array([a_iijj, b_iijj]) * 10
        eigvec_gt_iijj = rotAngle_to_eigenvectors(R_iijj)
        # Use the eigenvalues and eigenvectors to derive the cov matrix
        covMat_gt[ii,jj] = ellParams_to_covMat(radii_gt_iijj, eigvec_gt_iijj)
            
        #--------- Benchmark for evaluating model performance ------------
        # Evaluate using maximum eigenvalue (creates a bounding sphere)
        radii_gt_max = np.ones((2))*np.max(radii_gt_iijj)
        covMat_max = ellParams_to_covMat(radii_gt_max, np.eye(2))
        
        #compute Bures-Wasserstein distance between the ground truth cov matrix
        #and the smallest sphere that can just contain the ellipsoid
        BW_distance_maxEigval[ii,jj] = compute_Bures_Wasserstein_distance(\
            covMat_gt[ii,jj],covMat_max)
            
        # Evaluate using minimum eigenvalue (creates an inscribed sphere)
        radii_gt_min = np.ones((2))*np.min(radii_gt_iijj)
        covMat_min = ellParams_to_covMat(radii_gt_min, np.eye(2))
        
        #compute Bures-Wasserstein distance between the ground truth cov matrix
        #and the largest sphere that can just be put inside the ellipsoid
        BW_distance_minEigval[ii,jj] = compute_Bures_Wasserstein_distance(\
            covMat_gt[ii,jj],covMat_min)
        
        #Compute normalized bures similarity and Bures-Wasserstein distance
        #between ground truth and ellipsoids at selected corner locations
        for m in range(len(idx_corner)):
            BW_distance_corner[m,ii,jj] = compute_Bures_Wasserstein_distance(\
                covMat_corner[m],covMat_gt[ii,jj])
            LU_distance_corner[m,ii,jj] = log_operator_norm_distance(\
                covMat_corner[m],covMat_gt[ii,jj])
        
        #------------ Evaluaing performance -------------
        #The score is based on the comparison between model-predicted ellipsoids
        #and ground truth ellipsoid
        for l in range(nLevels): 
            eigVec_jiijj = rotAngle_to_eigenvectors(data_load[l]['model_pred_Wishart'].params_ell[ii][jj][-1])
            radii_jiijj = np.array(data_load[l]['model_pred_Wishart'].params_ell[ii][jj][2:4])
            
            covMat_modelPred[l,ii,jj] = ellParams_to_covMat(radii_jiijj, eigVec_jiijj)
            BW_distance[l,ii,jj] = compute_Bures_Wasserstein_distance(\
                covMat_gt[ii,jj],covMat_modelPred[l,ii,jj])
            LU_distance[l,ii,jj] = log_operator_norm_distance(\
                covMat_gt[ii,jj],covMat_modelPred[l,ii,jj])
                

#%% plotting starts from here
BW_benchmark = np.concatenate((BW_distance_minEigval[np.newaxis,:,:],\
                               BW_distance_maxEigval[np.newaxis,:,:],\
                               BW_distance_corner), axis = 0)

cmap_BW = np.full((len(idx_corner)+2,3),np.nan)
cmap_BW[0] = np.zeros((3))
cmap_BW[1] = np.zeros((3))
cmap_BW[2::] = np.vstack([stim2D['ref_points'][plane_2D_idx][:,m,n] for m,n in idx_corner])
cmap_temp = plt.get_cmap('Dark2')
cmap_t = cmap_temp(np.linspace(0.3, 0.8, nLevels))[:, :3]
  
x_ub = 0.20
y_ub = 20
BW_bins = np.linspace(0,x_ub,11)
BW_bins2 = np.linspace(0, x_ub,22)
if varyingFactor == 'jitter': legend_str = [' (low)', ' (medium)', ' (high)']
else: legend_str = ['']*nLevels

# plotting starts from here
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams.update({'font.size': 10})

fig2, ax1 = plt.subplots(1,1, figsize = (3.2, 2.5))
plot_similarity_metric_scores(ax1, BW_distance, BW_bins2,
                              y_ub = y_ub, cmap = cmap_t,
                              legend_labels = [str(varyingLevels[i])+legend_str[i] \
                                               for i in range(nLevels)])
plot_benchmark_similarity(ax1, BW_benchmark, BW_bins,cmap = cmap_BW,\
                          linestyle = [':','--']+['-']*len(idx_corner),\
                          jitter = np.linspace(-0.001,0.001, len(idx_corner)+2))
ax1.set_xticks(np.linspace(0,x_ub,5))
ax1.set_xlim(right=x_ub)
ax1.set_ylim([0, y_ub])
ax1.set_yticks(np.linspace(0,y_ub,5))
ax1.set_xlabel('The Bures-Wasserstein distance')
ax1.set_ylabel('Frequency')
if varyingFactor == 'nSims_total': ax1.legend(title = 'Number of total trial')
elif varyingFactor == 'nSims': ax1.legend(title = 'Number of trials per ref')
elif varyingFactor == 'jitter':ax1.legend(title = 'Amount of jitter')
ax1.set_title(plane_2D, fontsize = 10)
figName2 = f"ModelPerformance_BuresWassersteinDistance_2Dellipses_{plane_2D}_nSims{nSims}_jitter{jitter}_seed0.pdf"
full_path2 = os.path.join(fig_outputDir, figName2)
plt.tight_layout()
if saveFig: fig2.savefig(full_path2)
plt.show()

#%%
cmap_LU = np.full((len(idx_corner),3),np.nan)
cmap_LU = np.vstack([stim2D['ref_points'][plane_2D_idx][:,m,n] for m,n in idx_corner])
  
x_ub = 2.25
LU_bins = np.linspace(0,x_ub,11)
LU_bins2 = np.linspace(0,x_ub,22)

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams.update({'font.size': 10})

fig3, ax2 = plt.subplots(1,1, figsize = (3.2, 2.5), dpi = 256)
plot_similarity_metric_scores(ax2, LU_distance, LU_bins2,
                              y_ub = y_ub, cmap = cmap_t,
                              legend_labels = [str(varyingLevels[i])+legend_str[i] \
                                               for i in range(nLevels)])
plot_benchmark_similarity(ax2, LU_distance_corner, LU_bins,cmap = cmap_LU,\
                          linestyle = ['-']*len(idx_corner),\
                          jitter = np.linspace(-0.02,0.02, len(idx_corner)+2))

ax2.set_xlabel('Log Euclidean distance')
ax2.set_ylabel('Frequency')
ax2.set_xticks(np.around(LU_bins[::2],3))
ax2.set_yticks(np.linspace(0, y_ub, 5))
ax2.set_ylim([0, y_ub])
ax2.set_xlim(right=x_ub)
ax2.set_title(plane_2D, fontsize = 10)
ax2.legend(title = 'Amount of jitter')
figName3 = f"ModelPerformance_LogEuclideanDistance_2Dellipses_{plane_2D}_nSims{nSims}_jitter{jitter}_seed0.pdf"
full_path3 = os.path.join(fig_outputDir, figName3)
plt.tight_layout()
if saveFig: fig3.savefig(full_path3)  
plt.show()




