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

nSims     = [240, 160, 120] #[240]
jitter    = 0.1#[0.1, 0.3, 0.5]
bandwidth = 0.005
path_str1 = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
            'ELPS_analysis/ModelFitting_DataFiles/'
data_load = []
for j in nSims:
    file_name_j = 'Fitted_isothreshold_ellipsoids_sim'+str(j) +\
                'perCond_samplingNearContour_jitter'+str(jitter)+'_bandwidth' +\
                str(bandwidth) + '.pkl'
    # file_name_j = 'Fitted_isothreshold_ellipsoids_sim'+str(nSims) +\
    #             'perCond_samplingNearContour_jitter'+str(j)+'_bandwidth' +\
    #             str(bandwidth) + '.pkl'
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
def compute_Bures_Wasserstein_distance(M1, M2):
    # Compute the square root of M1
    sqrt_M1 = sqrtm(M1)
    # Compute the product sqrt(M1) * M2 * sqrt(M1)
    product = sqrt_M1 @ M2 @ sqrt_M1
    # Compute the square root of the product
    sqrt_product = sqrtm(product)
    # Calculate the Bures-Wasserstein distance
    BW_distance = np.sqrt(np.trace(M1) + np.trace(M2) - 2 * np.trace(sqrt_product))
    return BW_distance
        
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
                    
def plot_similarity_metric_scores(similarity_score, bin_edges, **kwargs):
    nSets = similarity_score.shape[0]
    
    # Default parameters for ellipsoid fitting. Can be overridden by kwargs.
    pltP = {
        'y_ub':25,
        'legend_labels':[None for i in range(nSets)],
        'legend_title':'',
        'cmap':[],
        'xlabel':'Similarity metric',
        'ylabel':'Frequency',
        'xticks':[],
        'figDir':'',
        'saveFig': False,
        'figName':'ModelPerformance_metricScores_3Dellipsoids',
        'figName_ext':'',
            } 
    pltP.update(kwargs)
    
    fig, ax = plt.subplots(1, 1, figsize=(5,4.5))
    plt.rcParams['figure.dpi'] = 250
    for j in range(nSets):
        if len(pltP['cmap']) == 0: cmap_l = np.random.rand(1,3)
        else: cmap_l = pltP['cmap'][j];
        ax.hist(similarity_score[j].flatten(), bins = bin_edges, color = cmap_l,\
                alpha = 0.6, edgecolor = [1,1,1], label = pltP['legend_labels'][j])
        #plot the median
        median_j = np.median(similarity_score[j].flatten())
        ax.plot([median_j,median_j], [0,pltP['y_ub']],color = cmap_l, linestyle = '--')
    ax.grid(True, alpha=0.3)
    if pltP['legend_title'] != '':
        ax.legend(title = pltP['legend_title'])
    ax.set_xticks(pltP['xticks'])
    ax.set_ylim([0, pltP['y_ub']])
    ax.set_xlabel(pltP['xlabel'])
    ax.set_ylabel(pltP['ylabel'])
    if pltP['saveFig'] and pltP['figDir'] != '':
        full_path = os.path.join(pltP['figDir'], pltP['figName']+pltP['figName_ext']+'.png')
        fig.savefig(full_path)          

def plot_benchmark_similarity(similarity_score, bin_edges, **kwargs):
    nSets = similarity_score.shape[0]
    
    # Default parameters for ellipsoid fitting. Can be overridden by kwargs.
    pltP = {
        'y_ub':80,
        'cmap':[],
        'xticks':[],
        'xlabel':'Similarity metric',
        'ylabel':'Frequency',
        'linestyle':[],
        'figDir':'',
        'saveFig': False,
        'figName':'ModelPerformance_benchmark_3Dellipsoids',
        'figName_ext':'',
            } 
    pltP.update(kwargs)
    
    fig, ax = plt.subplots(1, 1, figsize=(5,4.5))
    plt.rcParams['figure.dpi'] = 250
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    for m in range(nSets):
        if len(pltP['cmap']) == 0: cmap_m = np.random.rand(1,3)
        else: cmap_m = pltP['cmap'][m];
        if len(pltP['linestyle']) == 0: ls_m = '-';
        else: ls_m = pltP['linestyle'][m]
        counts_m,_ = np.histogram(similarity_score[m].flatten(), bins=bin_edges)
        ax.plot(bin_centers, counts_m,  color = cmap_m, ls = ls_m)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(pltP['xticks'])
    ax.set_yticks(np.linspace(0,pltP['y_ub'],5))
    ax.set_ylim([0,pltP['y_ub']])
    ax.set_xlabel(pltP['xlabel'])
    ax.set_ylabel(pltP['ylabel'])
    if pltP['saveFig'] and pltP['figDir'] != '':
        full_path = os.path.join(pltP['figDir'], pltP['figName']+pltP['figName_ext']+'.png')
        fig.savefig(full_path)    
        
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
covMat_modelPred = np.full((len(nSims),)+stim3D['ref_points'].shape[0:3]+(3,3,), np.nan)
NB_similarity    = np.full((len(nSims),)+ stim3D['ref_points'].shape[0:3], np.nan)
BW_distance      = np.full(NB_similarity.shape, np.nan)
NB_similarity_maxEigval = np.full(stim3D['ref_points'].shape[0:3], np.nan)
NB_similarity_minEigval = np.full(NB_similarity_maxEigval.shape, np.nan)
BW_distance_maxEigval   = np.full(NB_similarity_maxEigval.shape, np.nan)
BW_distance_minEigval   = np.full(NB_similarity_maxEigval.shape, np.nan)

# Select specific points (corners) of the 3D space to focus on
idx_corner = [[0,0,0],[4,0,0],[0,4,0],[0,0,4],[4,4,0],[0,4,4],[4,0,4],[4,4,4]]
# Retrieve the covariance matrices at these corner points
covMat_corner = [results3D['ellipsoidParams'][i][j][k]['evecs'] @ \
    np.diag(results3D['ellipsoidParams'][i][j][k]['radii']) * 2 @ \
    results3D['ellipsoidParams'][i][j][k]['evecs'].T for i, j, k in idx_corner]
# Initialize arrays for corner-based metrics
NB_similarity_corner = np.full((len(idx_corner),)+NB_similarity_maxEigval.shape, np.nan)
BW_distance_corner  = np.full(NB_similarity_corner.shape, np.nan)
             
# Loop through each reference point in the 3D space
for ii in range(ref_size):
    for jj in range(ref_size):
        for kk in range(ref_size):
            # Retrieve eigenvalues and eigenvectors for the ground truth ellipsoids
            eigVec_gt_jiijjkk = results3D['ellipsoidParams'][ii,jj,kk]['evecs']
            eigVal_gt_jiijjkk = results3D['ellipsoidParams'][ii,jj,kk]['radii']  
            # Use the eigenvalues and eigenvectors to derive the cov matrix
            covMat_gt[ii,jj,kk] = eigVec_gt_jiijjkk @ np.diag(eigVal_gt_jiijjkk)*2 @\
                eigVec_gt_jiijjkk.T
                
            #--------- Benchmark for evaluating model performance ------------
            # Evaluate using maximum eigenvalue (creates a bounding sphere)
            eigVal_gt_max_jiijjkk = np.ones((3))*np.max(eigVal_gt_jiijjkk)
            covMat_max = np.eye(3) @ np.diag(eigVal_gt_max_jiijjkk) @ np.eye(3).T
            
            #compute normalized bures similarity between the ground truth cov matrix
            #and the smallest sphere that can just contain the ellipsoid
            NB_similarity_maxEigval[ii,jj,kk] = compute_normalized_Bures_similarity(\
                covMat_gt[ii,jj,kk],covMat_max)   
            #compute Bures-Wasserstein distance
            BW_distance_maxEigval[ii,jj,kk] = compute_Bures_Wasserstein_distance(\
                covMat_gt[ii,jj,kk],covMat_max)
                
            # Evaluate using minimum eigenvalue (creates an inscribed sphere)
            eigVal_gt_min_jiijjkk = np.ones((3))*np.min(eigVal_gt_jiijjkk)
            covMat_min = np.eye(3) @ np.diag(eigVal_gt_min_jiijjkk) @ np.eye(3).T
            
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
            for l in range(len(nSims)): #jitter   
                eigVec_jiijjkk = data_load[l]['params_ellipsoids'][ii][jj][kk]['evecs']
                eigVal_jiijjkk = data_load[l]['params_ellipsoids'][ii][jj][kk]['radii']
                
                covMat_modelPred[l,ii,jj,kk] = eigVec_jiijjkk @ np.diag(eigVal_jiijjkk) @ eigVec_jiijjkk.T
                NB_similarity[l,ii,jj,kk] = compute_normalized_Bures_similarity(\
                    covMat_gt[ii,jj,kk],covMat_modelPred[l,ii,jj,kk])
                BW_distance[l,ii,jj,kk] = compute_Bures_Wasserstein_distance(\
                    covMat_gt[ii,jj,kk],covMat_modelPred[l,ii,jj,kk])
                    

#%%
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

NBW_bins = np.arange(0.88,1.015, 0.01)
NBW_bin_edges = NBW_bins - (NBW_bins[1] - NBW_bins[0])/2
plot_benchmark_similarity(NBW_benchmark, NBW_bin_edges,\
                          xticks = np.around(NBW_bins[::2],3), \
                          linestyle = [':','--']+['-']*len(idx_corner),\
                          xlabel = 'The normalized Bures similarity',\
                          cmap = cmap_NBW, saveFig = False, figDir = fig_outputDir,\
                          figName_ext = '_NBW_nSims'+str(nSims) +'_jitter' + str(jitter))

BW_bins = np.linspace(0,0.24,12)
BW_bin_edges = BW_bins - (BW_bins[1] - BW_bins[0])/2
plot_benchmark_similarity(BW_benchmark, BW_bin_edges,\
                          linestyle = [':','--']+['-']*len(idx_corner),\
                          xlabel = 'The Bures-Wasserstein distance',y_ub = 100,\
                          cmap = cmap_NBW, xticks = np.around(BW_bins[::2],3),\
                          saveFig = False, figDir = fig_outputDir,\
                          figName_ext = '_BW_nSims'+str(nSims) +'_jitter' + str(jitter))

#legend_str = [' (small)', ' (medium)', ' (large)']
legend_str = ['','','']
NBW_scores_edges = np.arange(0.9902,1.0003,0.0004)
NBW_score_cmap = np.array([[247,152,29],[75,40,73],[107,142,35]])/255
plot_similarity_metric_scores(NB_similarity, NBW_scores_edges, \
                              xticks = np.linspace(0.99, 1,5), y_ub = 50,\
                              cmap = NBW_score_cmap,\
                              legend_labels = [str(nSims[i])+legend_str[i] for i in range(len(nSims))],\
                              legend_title = 'Number of trial\nper reference stimulus',\
                              xlabel = 'The normalized Bures similarity',\
                              saveFig = True, figDir = fig_outputDir,\
                              figName_ext = '_NBWscores_nSims'+str(nSims) +'_jitter' + str(jitter))

plot_similarity_metric_scores(BW_distance, np.linspace(0,0.08,25),\
    xticks = np.linspace(0,0.07,5),y_ub = 50,\
    cmap = np.array([[247,152,29],[75,40,73],[107,142,35]])/255,\
    legend_labels = [str(nSims[i])+legend_str[i] for i in range(len(nSims))],\
    legend_title = 'Number of trial\nper reference stimulus',\
    xlabel = 'The Bures-Wasserstein distance',\
    saveFig = True, figDir = fig_outputDir,\
    figName_ext = '_BWscores_nSims'+str(nSims) +'_jitter' + str(jitter))
    




