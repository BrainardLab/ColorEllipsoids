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
import pickle
import numpy as np
sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids')
sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/Python version')
from model_performance import compute_Bures_Wasserstein_distance, \
    compute_normalized_Bures_similarity, plot_similarity_metric_scores,\
    plot_benchmark_similarity
from core.model_predictions import ellParams_to_covMat
        
fig_outputDir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                        'ELPS_analysis/ModelPerformance_FigFiles/'

#%% ------------------------------------------
# Load data simulated using CIELab
# ------------------------------------------

def evaluate_modelPerformance_3D(varyingFactor, varyingLevels, fixedFactor,\
                                 fixedLevel, bandwidth= 0.005, saveFig = False):
    nLevels = len(varyingLevels)
    path_str1 = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                'ELPS_analysis/ModelFitting_DataFiles/3D_oddity_task_reference/'
    data_load = []
        
    if varyingFactor == 'nSims' and fixedFactor == 'jitter':
        nSims     = varyingLevels #[240, 160, 120] #[240]
        jitter    = fixedLevel #0.1#[0.1, 0.3, 0.5]

        for j in varyingLevels:
            file_name_j = f'Fitted_isothreshold_ellipsoids_sim{j}' +\
                        f'perCond_samplingNearContour_jitter{jitter}_bandwidth{bandwidth}.pkl'
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
            file_name_j = f'Fitted_isothreshold_ellipsoids_sim{nSims}' +\
                        f'perCond_samplingNearContour_jitter{j:01}_bandwidth{bandwidth}.pkl'
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
            
    #%% evalutate model performance
    """
    Initialize arrays to store covariance matrices and distance metrics for ground 
    truth and predictions
    covMat_gt: Ground truth covariance matrices
    covMat_modelPred: model-predicted covariance matrices
    NB_similarity:  Normalized Bures similarity for each simulation
    BW_distance: Bures-Wasserstein distance for each simulation
    NB_similarity_maxEigval: Max-eigenvalue-based normalized Bures similarity
    NB_similarity_minEigval: Min eigenvalue-based normalized Bures similarity
    BW_distance_maxEigval: Max eigenvalue-based Bures-Wasserstein distance
    BW_distance_minEigval: Min eigenvalue-based Bures-Wasserstein distance
    
    """
    covMat_gt        = np.full(stim3D['ref_points'].shape[0:3]+(3,3,), np.nan)
    covMat_modelPred = np.full((nLevels,)+stim3D['ref_points'].shape[0:3]+(3,3,), np.nan)
    NB_similarity    = np.full((nLevels,)+ stim3D['ref_points'].shape[0:3], np.nan)
    BW_distance      = np.full(NB_similarity.shape, np.nan)
    NB_similarity_maxEigval = np.full(stim3D['ref_points'].shape[0:3], np.nan)
    NB_similarity_minEigval = np.full(NB_similarity_maxEigval.shape, np.nan)
    BW_distance_maxEigval   = np.full(NB_similarity_maxEigval.shape, np.nan)
    BW_distance_minEigval   = np.full(NB_similarity_maxEigval.shape, np.nan)
    
    # Select specific points (corners) of the 3D space to focus on
    idx_corner = [[0,0,0],[4,0,0],[0,4,0],[0,0,4],[4,4,0],[0,4,4],[4,0,4],[4,4,4]]
    #[[x, y, z] for x in (0, 4) for y in (0, 4) for z in (0, 4)]
    # Retrieve the covariance matrices at these corner points
    """
    Note in the following code, we have to scale the radii from results3D by 2,
    because when those ellipsoids were fit, they are bounded by [0, 1]. On the
    other hand, the radii from data_load were derived from Wishart model 
    predictions, which were bounded by [-1, 1]. To make the scale match, we 
    have to scale the former by 2.
    """
    covMat_corner = [ellParams_to_covMat(results3D['ellipsoidParams'][i][j][k]['radii']*2,\
                    results3D['ellipsoidParams'][i][j][k]['evecs']) \
                    for i, j, k in idx_corner]
    # Initialize arrays for corner-based metrics
    NB_similarity_corner = np.full((len(idx_corner),)+NB_similarity_maxEigval.shape, np.nan)
    BW_distance_corner  = np.full(NB_similarity_corner.shape, np.nan)
                 
    # Loop through each reference point in the 3D space
    for ii in range(ref_size):
        for jj in range(ref_size):
            for kk in range(ref_size):
                # Retrieve eigenvalues and eigenvectors for the ground truth ellipsoids
                eigVec_gt_jiijjkk = results3D['ellipsoidParams'][ii,jj,kk]['evecs']
                radii_gt_jiijjkk = results3D['ellipsoidParams'][ii,jj,kk]['radii']*2
                # Use the eigenvalues and eigenvectors to derive the cov matrix
                covMat_gt[ii,jj,kk] = ellParams_to_covMat(radii_gt_jiijjkk, eigVec_gt_jiijjkk)
                    
                #--------- Benchmark for evaluating model performance ------------
                # Evaluate using maximum eigenvalue (creates a bounding sphere)
                radii_gt_max_jiijjkk = np.ones((3))*np.max(radii_gt_jiijjkk)
                covMat_max = ellParams_to_covMat(radii_gt_max_jiijjkk, np.eye(3))
                
                #compute normalized bures similarity between the ground truth cov matrix
                #and the smallest sphere that can just contain the ellipsoid
                NB_similarity_maxEigval[ii,jj,kk] = compute_normalized_Bures_similarity(\
                    covMat_gt[ii,jj,kk],covMat_max)   
                #compute Bures-Wasserstein distance
                BW_distance_maxEigval[ii,jj,kk] = compute_Bures_Wasserstein_distance(\
                    covMat_gt[ii,jj,kk],covMat_max)
                    
                # Evaluate using minimum eigenvalue (creates an inscribed sphere)
                raidii_gt_min_jiijjkk = np.ones((3))*np.min(radii_gt_jiijjkk)
                covMat_min = ellParams_to_covMat(raidii_gt_min_jiijjkk, np.eye(3))
                
                #compute normalized bures similarity between the ground truth cov matrix
                #and the largest sphere that can just be put inside the ellipsoid
                NB_similarity_minEigval[ii,jj,kk] = compute_normalized_Bures_similarity(\
                    covMat_gt[ii,jj,kk],covMat_min)   
                #compute Bures-Wasserstein distance
                BW_distance_minEigval[ii,jj,kk] = compute_Bures_Wasserstein_distance(\
                    covMat_gt[ii,jj,kk],covMat_min)
                
                #Compute normalized bures similarity and Bures-Wasserstein distance
                #between ground truth and ellipsoids at selected corner locations
                for m in range(len(idx_corner)):
                    NB_similarity_corner[m,ii,jj,kk] = compute_normalized_Bures_similarity(\
                        covMat_corner[m],covMat_gt[ii,jj,kk])
                    BW_distance_corner[m,ii,jj,kk] = compute_Bures_Wasserstein_distance(\
                        covMat_corner[m],covMat_gt[ii,jj,kk])
                
                #------------ Evaluaing performance -------------
                #The score is based on the comparison between model-predicted ellipsoids
                #and ground truth ellipsoid
                for l in range(nLevels): 
                    eigVec_jiijjkk = data_load[l]['params_ellipsoids'][ii][jj][kk]['evecs']
                    radii_jiijjkk = data_load[l]['params_ellipsoids'][ii][jj][kk]['radii']
                    
                    covMat_modelPred[l,ii,jj,kk] = ellParams_to_covMat(radii_jiijjkk, eigVec_jiijjkk)
                    NB_similarity[l,ii,jj,kk] = compute_normalized_Bures_similarity(\
                        covMat_gt[ii,jj,kk],covMat_modelPred[l,ii,jj,kk])
                    BW_distance[l,ii,jj,kk] = compute_Bures_Wasserstein_distance(\
                        covMat_gt[ii,jj,kk],covMat_modelPred[l,ii,jj,kk])
                        
    
    #%% plotting starts from here
    NBW_benchmark = np.concatenate((NB_similarity_minEigval[np.newaxis,:,:,:],\
                                    NB_similarity_maxEigval[np.newaxis,:,:,:],\
                                    NB_similarity_corner), axis = 0)
    BW_benchmark = np.concatenate((BW_distance_minEigval[np.newaxis,:,:,:],\
                                   BW_distance_maxEigval[np.newaxis,:,:,:],\
                                   BW_distance_corner), axis = 0)
    
    cmap_NBW = np.full((len(idx_corner)+2,3),np.nan)
    cmap_NBW[0] = np.zeros((3))
    cmap_NBW[1] = np.zeros((3))
    for c in range(len(idx_corner)):
        cmap_NBW[c+2] = [stim3D['grid_ref'][m] for m in idx_corner[c]]
    
    NBW_bins = np.arange(0.72,1.02, 0.02)
    NBW_bin_edges = NBW_bins - (NBW_bins[1] - NBW_bins[0])/2
    
    #figure for benchmark
    fig1, ax1 = plt.subplots(1, 1, figsize=(5,4.5))
    plot_benchmark_similarity(ax1, NBW_benchmark, NBW_bin_edges,\
                              linestyle = [':','--']+['-']*len(idx_corner),\
                              cmap = cmap_NBW)
    ax1.set_xticks(np.around(NBW_bins[::2],3))
    ax1.set_yticks(np.linspace(0,80,5))
    ax1.set_xlim([0.72, 1])
    ax1.set_ylim([0,80])
    ax1.set_xlabel('The normalized Bures similarity')
    ax1.set_ylabel('Frequency')
    figName1 = "ModelPerformance_NormalizedBuresSimilarity_3Dellipsoids_benchmark.png"
    full_path1 = os.path.join(fig_outputDir, figName1)
    if saveFig: fig1.savefig(full_path1)    
       
    #%%
    BW_bins = np.linspace(0,0.09,11)
    BW_bin_edges = BW_bins - (BW_bins[1] - BW_bins[0])/2
    
    fig2, ax2 = plt.subplots(1,1, figsize = (5, 4.5))
    plot_benchmark_similarity(ax2, BW_benchmark, BW_bin_edges,\
                              linestyle = [':','--']+['-']*len(idx_corner),cmap = cmap_NBW)
    
    ax2.set_xlabel('The Bures-Wasserstein distance')
    ax2.set_ylabel('Frequency')
    ax2.set_xticks(np.around(BW_bins[::2],3))
    ax2.set_yticks(np.linspace(0, 80, 5))
    ax2.set_xlim([0, 0.09])
    ax2.set_ylim([0, 80])
    figName2 = "ModelPerformance_BuresWassersteinDistance_3Dellipsoids_benchmark.png"
    full_path2 = os.path.join(fig_outputDir, figName2)
    if saveFig: fig2.savefig(full_path2)   
    
    #%%
    if varyingFactor == 'jitter': legend_str = [' (small)', ' (medium)', ' (large)']
    elif varyingFactor == 'nSims': legend_str = ['','','']
    NBW_scores_edges = np.arange(0.9312,1.0012,0.0024)
    NBW_score_cmap = np.array([[247,152,29],[75,40,73],[107,142,35]])/255
    
    fig3, ax3 = plt.subplots(1,1, figsize = (5, 4.5))
    plot_similarity_metric_scores(ax3, NB_similarity, NBW_scores_edges, y_ub = 80,\
                                  cmap = NBW_score_cmap, \
                                  legend_labels = [str(varyingLevels[i])+legend_str[i] \
                                                   for i in range(nLevels)])
    ax3.set_xticks(np.linspace(0.93, 1,8))
    ax3.set_ylim([0, 80])
    ax3.set_yticks(np.linspace(0, 80, 5))
    ax3.set_xlabel('The normalized Bures similarity')
    ax3.set_ylabel('Frequency')
    if varyingFactor == 'nSims': ax3.legend(title = 'Number of trial\nper reference stimulus')
    elif varyingFactor == 'jitter': ax3.legend(title = 'Amount of jitter')
    figName3 = figName1[:-14] +f"_nSims{nSims}_jitter{jitter}.png"
    full_path3 = os.path.join(fig_outputDir, figName3)
    if saveFig: fig3.savefig(full_path3)
    
    #%%
    fig4, ax4 = plt.subplots(1,1, figsize = (5, 4.5))
    BW_bins = np.linspace(0,0.04,26)
    BW_edges = BW_bins - (BW_bins[2] - BW_bins[1])/2
    plot_similarity_metric_scores(ax4, BW_distance, BW_edges,\
                                  y_ub = 80, cmap = NBW_score_cmap,\
                                  legend_labels = [str(varyingLevels[i])+legend_str[i] \
                                                   for i in range(nLevels)])
    ax4.set_xlim([0, 0.04])
    ax4.set_xticks(BW_bins[::5])
    ax4.set_ylim([0, 80])
    ax4.set_yticks(np.linspace(0, 80, 5))
    ax4.set_xlabel('The Bures-Wasserstein distance')
    ax4.set_ylabel('Frequency')
    if varyingFactor == 'nSims': ax4.legend(title = 'Number of trial\nper reference stimulus')
    elif varyingFactor == 'jitter':ax4.legend(title = 'Amount of jitter')
    figName4 = figName2[:-14] +f"_nSims{nSims}_jitter{jitter}.png"
    full_path4 = os.path.join(fig_outputDir, figName4)
    if saveFig: fig4.savefig(full_path4)
    

#%% Main code
#evaluate_modelPerformance_3D('nSims', [240, 160, 120],\
#                             'jitter', 0.1, saveFig = False)

evaluate_modelPerformance_3D('jitter', [0.1, 0.3, 0.5],\
                             'nSims', 240, saveFig = True)    






