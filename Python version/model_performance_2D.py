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
import pickle
import numpy as np
sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids')
sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/Python version')
from model_performance import compute_Bures_Wasserstein_distance, \
    compute_normalized_Bures_similarity, plot_similarity_metric_scores,\
    plot_benchmark_similarity
from core.model_predictions import ellParams_to_covMat, rotAngle_to_eigenvectors
        
fig_outputDir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                        'ELPS_analysis/ModelPerformance_FigFiles/'

#%% ------------------------------------------
# Load data simulated using CIELab
# ------------------------------------------

def evaluate_modelPerformance_3D(varyingFactor, varyingLevels, fixedFactor,\
                                 fixedLevel, plane_2D = 'RB plane',\
                                 saveFig = False):
    nLevels = len(varyingLevels)
    path_str1 = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                'ELPS_analysis/ModelFitting_DataFiles/2D_oddity_task/'
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
                            f'jitter{jitter}_bandwidth0.005.pkl'
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
            file_name_j = f'Fitted_isothreshold_{plane_2D}_samplingRandom_' +\
                        f'wFittedW_jitter{j:01}_nSims{nSims}total.pkl'
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
    
    """
    covMat_gt        = np.full(stim2D['ref_points'].shape[2:]+(2,2,), np.nan)
    covMat_modelPred = np.full((nLevels,)+stim2D['ref_points'].shape[2:]+(2,2,), np.nan)
    BW_distance      = np.full((nLevels,)+ stim2D['ref_points'].shape[2:], np.nan)
    BW_distance_maxEigval   = np.full(stim2D['ref_points'].shape[2:], np.nan)
    BW_distance_minEigval   = np.full(stim2D['ref_points'].shape[2:], np.nan)
    
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
            
            #------------ Evaluaing performance -------------
            #The score is based on the comparison between model-predicted ellipsoids
            #and ground truth ellipsoid
            for l in range(nLevels): 
                eigVec_jiijj = rotAngle_to_eigenvectors(data_load[l]['params_ellipses'][ii][jj][-1])
                radii_jiijj = np.array(data_load[l]['params_ellipses'][ii][jj][2:4])
                
                covMat_modelPred[l,ii,jj] = ellParams_to_covMat(radii_jiijj, eigVec_jiijj)
                BW_distance[l,ii,jj] = compute_Bures_Wasserstein_distance(\
                    covMat_gt[ii,jj],covMat_modelPred[l,ii,jj])
                    
    
    #%% plotting starts from here
    BW_benchmark = np.concatenate((BW_distance_minEigval[np.newaxis,:,:],\
                                   BW_distance_maxEigval[np.newaxis,:,:],\
                                   BW_distance_corner), axis = 0)
    
    cmap_NBW = np.full((len(idx_corner)+2,3),np.nan)
    cmap_NBW[0] = np.zeros((3))
    cmap_NBW[1] = np.zeros((3))
    cmap_NBW[2::] = np.vstack([stim2D['ref_points'][plane_2D_idx][:,m,n] for m,n in idx_corner])
      
    x_ub = 0.20
    BW_bins = np.linspace(0,x_ub,11)
    BW_bin_edges = BW_bins - (BW_bins[1] - BW_bins[0])/2
    
    fig1, ax1 = plt.subplots(1,1, figsize = (4.5, 5.5))
    plot_benchmark_similarity(ax1, BW_benchmark, BW_bin_edges,cmap = cmap_NBW,\
                              linestyle = [':','--']+['-']*len(idx_corner),\
                              jitter = np.linspace(-0.005,0.005, len(idx_corner)+2))
    
    ax1.set_xlabel('The Bures-Wasserstein distance')
    ax1.set_ylabel('Frequency')
    ax1.set_xticks(np.around(BW_bins[::2],3))
    ax1.set_yticks(np.linspace(0, 16, 5))
    ax1.set_ylim([0, 16])
    ax1.set_title(plane_2D)
    figName1 = f"ModelPerformance_BuresWassersteinDistance_2Dellipses_{plane_2D}_benchmark.png"
    full_path1 = os.path.join(fig_outputDir, figName1)
    if saveFig: fig1.savefig(full_path1)   
    
    #%%
    if varyingFactor == 'jitter': legend_str = [' (small)', ' (medium)', ' (large)']
    else: legend_str = ['']*nLevels
    cmap_temp = plt.get_cmap('Dark2')
    cmap_t = cmap_temp(np.linspace(0, 1, nLevels))[:, :3]

    fig2, ax2 = plt.subplots(1,1, figsize = (4.5, 5.5))
    x_ub = np.max([np.around(np.max(BW_distance), 2), x_ub])
    BW_bins2 = np.linspace(0, x_ub,35)
    BW_bin_edges2 = BW_bins2 - (BW_bins2[1] - BW_bins2[0])/2
    plot_similarity_metric_scores(ax2, BW_distance, BW_bin_edges2,\
                                  y_ub = 16, cmap = cmap_t,\
                                  legend_labels = [str(varyingLevels[i])+legend_str[i] \
                                                   for i in range(nLevels)])
    ax2.set_xticks(np.linspace(0,x_ub,5))
    ax2.set_ylim([0, 16])
    ax2.set_yticks(np.linspace(0,16,5))
    ax2.set_xlabel('The Bures-Wasserstein distance')
    ax2.set_ylabel('Frequency')
    if varyingFactor == 'nSims_total': ax2.legend(title = 'Number of total trial')
    elif varyingFactor == 'nSims': ax2.legend(title = 'Number of trials per ref')
    elif varyingFactor == 'jitter':ax2.legend(title = 'Amount of jitter')
    ax2.set_title(plane_2D)
    figName2 = figName1[:-14] +f"_nSims{nSims}_jitter{jitter}.png"
    full_path2 = os.path.join(fig_outputDir, figName2)
    if saveFig: fig2.savefig(full_path2)
    

#%% Main code
evaluate_modelPerformance_3D('nSims_total', [6000,5000,4000,3000,2000,1500,1000],\
                             'jitter', 0.1, plane_2D = 'GB plane', saveFig = True)
    
#%%
#evaluate_modelPerformance_3D('nSims', [240, 200, 160, 120, 80, 60, 40],\
#                             'jitter', 0.1, plane_2D = 'RG plane', saveFig = True) 






