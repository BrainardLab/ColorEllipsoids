#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 13:15:58 2024

@author: fangfang
"""

import jax
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
import numpy as np
import sys
sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids')
from analysis.ellipses_tools import ellParams_to_covMat, rotAngle_to_eigenvectors

#%%
class ModelPerformance():
    def __init__(self, color_dimension, CIE_results, CIE_stim, varying_levels, 
                 plane_2D = None):
        self.ndims       = color_dimension 
        self.CIE_results = CIE_results        
        self.CIE_stim    = CIE_stim
        self.plane_2D    = plane_2D
        self.levels      = varying_levels
        self.nLevels     = len(varying_levels)
        if plane_2D is not None:
            # Retrieve the covariance matrices at these corner points
            plane_2D_dict = {'GB plane': 0, 'RB plane': 1, 'RG plane': 2}
            self.plane_2D_idx  = plane_2D_dict[plane_2D]
        self.ref_size = self.CIE_stim['nGridPts_ref'] 
        self._retrieve_ellParams_gt()
        
    def _retrieve_ellParams_gt(self):
        if self.ndims == 2:
            self.ellParams_gt = self.CIE_results['ellParams'][self.plane_2D_idx]
        else:
            self.ellParams_gt = self.CIE_results['ellipsoidParams']
        
    def _initialize(self):
        if self.ndims == 2:
            base_shape = (self.ref_size, self.ref_size)
        else:
            base_shape = (self.ref_size, self.ref_size, self.ref_size)
        self.BW_distance           = np.full((self.nLevels,)+ base_shape, np.nan)
        self.BW_distance_maxEigval = np.full(base_shape, np.nan)
        self.BW_distance_minEigval = np.full(base_shape, np.nan)
        
        self.LU_distance           = np.full((self.nLevels,) + base_shape, np.nan)
        self.LU_distance_maxEigval = np.full(base_shape, np.nan)
        self.LU_distance_minEigval = np.full(base_shape, np.nan)
        
        self.covMat_gt             = np.full(base_shape+(self.ndims, self.ndims,), np.nan)
        self.covMat_modelPred      = np.full((self.nLevels,)+base_shape+(self.ndims, self.ndims,), np.nan)
    
    def load_modelPreds_ellParams(self, data_load):
        if self.ndims == 2:
            for l in range(self.nLevels): 
                for ii in range(self.ref_size):
                    for jj in range(self.ref_size):
                        eigVec_jiijj = rotAngle_to_eigenvectors(data_load[l]['model_pred_Wishart'].params_ell[ii][jj][-1])
                        radii_jiijj = np.array(data_load[l]['model_pred_Wishart'].params_ell[ii][jj][2:4])/2
                        self.covMat_modelPred[l,ii,jj] = ellParams_to_covMat(radii_jiijj, eigVec_jiijj)
        else:
            for l in range(self.nLevels):
                for ii in range(self.ref_size):
                    for jj in range(self.ref_size):
                        for kk in range(self.ref_size):
                            eigVec_jiijjkk = data_load[l]['model_pred_Wishart'].params_ell[ii][jj][kk]['evecs']
                            radii_jiijjkk = data_load[l]['model_pred_Wishart'].params_ell[ii][jj][kk]['radii']/2                            
                            self.covMat_modelPred[l,ii,jj,kk] = ellParams_to_covMat(radii_jiijjkk, eigVec_jiijjkk)
                        
        
    def _convert_ellParams_to_covMat(self, ellParams, scaler_x1 = 5):
        if self.ndims == 2:
            _, _, a, b, R = ellParams
            radii = np.array([a, b])*scaler_x1
            eigvec = rotAngle_to_eigenvectors(R)
            covMat = ellParams_to_covMat(radii, eigvec) 
        else:
            radii = ellParams['radii']*scaler_x1
            covMat = ellParams_to_covMat(radii,
                                                ellParams['evecs'])
        return covMat, radii
                
    def compare_with_extreme_ell(self, ell1Params, scaler_x1 = 5):  
        # Use the eigenvalues and eigenvectors to derive the cov matrix
        covMat_gt, radii_gt = self._convert_ellParams_to_covMat(ell1Params)
            
        #--------- Benchmark for evaluating model performance ------------
        # Evaluate using maximum eigenvalue (creates a bounding sphere)
        radii_gt_max = np.ones((self.ndims))*np.max(radii_gt)
        covMat_max = ellParams_to_covMat(radii_gt_max, np.eye(self.ndims))
        
        # Evaluate using minimum eigenvalue (creates an inscribed sphere)
        radii_gt_min = np.ones((self.ndims))*np.min(radii_gt)
        covMat_min = ellParams_to_covMat(radii_gt_min, np.eye(self.ndims))
        
        #compute Bures-Wasserstein distance between the ground truth cov matrix
        #and the smallest sphere that can just contain the ellipsoid
        BW_distance_maxEigval = self.compute_Bures_Wasserstein_distance(\
            covMat_gt,covMat_max)
        #compute Bures-Wasserstein distance between the ground truth cov matrix
        #and the largest sphere that can just be put inside the ellipsoid
        BW_distance_minEigval = self.compute_Bures_Wasserstein_distance(\
            covMat_gt,covMat_min)
            
        LU_distance_maxEigval = self.log_operator_norm_distance(covMat_gt,
                                                                covMat_max)       
        LU_distance_minEigval = self.log_operator_norm_distance(covMat_gt,
                                                                covMat_min)     
        
        return covMat_gt, BW_distance_minEigval, BW_distance_maxEigval,\
            LU_distance_minEigval, LU_distance_maxEigval
    
    def compare_with_corner_ell(self, covMat_gt, covMat_corner):
        #initialize
        BW_distance_corner = np.full((self.nCorners,), np.nan)
        LU_distance_corner = np.full((self.nCorners,), np.nan)
        #Compute normalized bures similarity and Bures-Wasserstein distance
        #between ground truth and ellipsoids at selected corner locations
        for m in range(self.nCorners):
            BW_distance_corner[m] = self.compute_Bures_Wasserstein_distance(\
                covMat_corner[m],covMat_gt)
            LU_distance_corner[m] = self.log_operator_norm_distance(\
                covMat_corner[m],covMat_gt)
        return BW_distance_corner, LU_distance_corner
        
    def compare_gt_model_pred_one_instance(self, covMat_gt, covMat_modelPred):
        #------------ Evaluaing performance -------------
        #The score is based on the comparison between model-predicted ellipsoids
        #and ground truth ellipsoid
        BW_distance = np.full((self.nLevels,), np.nan)
        LU_distance = np.full((self.nLevels,), np.nan)
        for l in range(self.nLevels):    
            BW_distance[l] = self.compute_Bures_Wasserstein_distance(\
                covMat_gt,covMat_modelPred[l])
            LU_distance[l] = self.log_operator_norm_distance(covMat_gt,
                                                             covMat_modelPred[l])
        return BW_distance, LU_distance

    def evaluate_model_performance(self, model_pred_data, covMat_corner = None):
        #initialize 
        self._initialize()
        self.load_modelPreds_ellParams(model_pred_data)
        if covMat_corner is not None:
            self.nCorners = len(covMat_corner)
            self.BW_distance_corner = np.full((self.nCorners,)+ \
                                              self.BW_distance_maxEigval.shape, np.nan)
            self.LU_distance_corner = np.full(self.BW_distance_corner.shape, np.nan)
            
        if self.ndims == 2:
            for ii in range(self.ref_size):
                for jj in range(self.ref_size):
                    #smallest and largest ellipses/ellipsoids
                    self.covMat_gt[ii,jj], self.BW_distance_minEigval[ii,jj], \
                        self.BW_distance_maxEigval[ii,jj],\
                        self.LU_distance_minEigval[ii,jj],\
                        self.LU_distance_maxEigval[ii,jj] = \
                        self.compare_with_extreme_ell(self.ellParams_gt[ii][jj])
                    
                    #compare each one with the corner
                    if covMat_corner is not None:
                        self.BW_distance_corner[:,ii,jj], self.LU_distance_corner[:,ii,jj] = \
                            self.compare_with_corner_ell(self.covMat_gt[ii,jj],
                                                              covMat_corner)
                    
                    #ground truth vs. model predictions
                    self.BW_distance[:,ii,jj], self.LU_distance[:,ii,jj] = \
                        self.compare_gt_model_pred_one_instance(self.covMat_gt[ii,jj], 
                                                            self.covMat_modelPred[:,ii,jj])
                        
        else:
            for ii in range(self.ref_size):
                for jj in range(self.ref_size):
                    for kk in range(self.ref_size):
                        #smallest and largest ellipses/ellipsoids
                        self.covMat_gt[ii,jj,kk], self.BW_distance_minEigval[ii,jj,kk], \
                            self.BW_distance_maxEigval[ii,jj,kk],\
                            self.LU_distance_minEigval[ii,jj,kk],\
                            self.LU_distance_maxEigval[ii,jj,kk] = \
                            self.compare_with_extreme_ell(self.ellParams_gt[ii][jj][kk])
                        
                        #compare each one with the corner
                        if covMat_corner is not None:
                            self.BW_distance_corner[:,ii,jj,kk], self.LU_distance_corner[:,ii,jj,kk] = \
                                self.compare_with_corner_ell(self.covMat_gt[ii,jj,kk],
                                                                  covMat_corner)
                        
                        #ground truth vs. model predictions
                        self.BW_distance[:,ii,jj,kk], self.LU_distance[:,ii,jj,kk] = \
                            self.compare_gt_model_pred_one_instance(self.covMat_gt[ii,jj,kk], 
                                                                self.covMat_modelPred[:,ii,jj,kk])            
        
    def concatenate_benchamrks(self):
        #we pick multiple ellipses/ellipsoids for computing benchmarks, including
        #the 
        self.BW_benchmark = np.concatenate((self.BW_distance_minEigval[np.newaxis],\
                                           self.BW_distance_maxEigval[np.newaxis],\
                                           self.BW_distance_corner), axis = 0)
            
        self.LU_benchmark = np.concatenate((self.LU_distance_minEigval[np.newaxis],\
                                       self.LU_distance_maxEigval[np.newaxis],\
                                       self.LU_distance_corner), axis = 0)

#%%
    @staticmethod
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
            
    @staticmethod
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
    
    @staticmethod
    def log_psd_matrix(S, tol=1e-4):
    	v, U = np.linalg.eigh(S)
    	d = np.log(np.clip(v, tol, None))
    	return U @ np.diag(d) @ U.T
    
    @staticmethod
    def log_operator_norm_distance(A, B):
    	lgA = ModelPerformance.log_psd_matrix(A)
    	lgB = ModelPerformance.log_psd_matrix(B)
    	return np.linalg.norm(lgA - lgB, 2)
         
    #%%               
    @staticmethod
    def plot_similarity_metric_scores(ax, similarity_score, bin_edges, **kwargs):
        nSets = similarity_score.shape[0]
        
        # Default parameters for ellipsoid fitting. Can be overridden by kwargs.
        pltP = {
            'y_ub':25,
            'legend_labels':[None for i in range(nSets)],
            'legend_title':'',
            'cmap':[],
            'alpha': 0.6
            } 
        pltP.update(kwargs)
        
        #fig, ax = plt.subplots(1, 1, figsize=(5,4.5))
        plt.rcParams['figure.dpi'] = 256
        for j in range(nSets):
            if len(pltP['cmap']) == 0: cmap_l = np.random.rand(1,3)
            else: cmap_l = pltP['cmap'][j];
            ax.hist(similarity_score[j].flatten(), bins = bin_edges,\
                    color = cmap_l, alpha = pltP['alpha'], edgecolor = [1,1,1],\
                    label = pltP['legend_labels'][j])
            #plot the median
            median_j = np.median(similarity_score[j].flatten())
            ax.plot([median_j,median_j], [0,pltP['y_ub']],color = cmap_l, linestyle = '--', lw = 1)
        ax.grid(True, alpha=0.3)
        
    @staticmethod
    def plot_benchmark_similarity(ax, similarity_score, bin_edges, **kwargs):
        nSets = similarity_score.shape[0]
        
        # Default parameters for ellipsoid fitting. Can be overridden by kwargs.
        pltP = {
            'cmap':[],
            'linestyle':[],
            'jitter':np.zeros((nSets)),
            'lw': 1,
            } 
        pltP.update(kwargs)
        
        #fig, ax = plt.subplots(1, 1, figsize=(5,4.5))
        plt.rcParams['figure.dpi'] = 256
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        for m in range(nSets):
            if len(pltP['cmap']) == 0: cmap_m = np.random.rand(1,3)
            else: cmap_m = pltP['cmap'][m];
            if len(pltP['linestyle']) == 0: ls_m = '-';
            else: ls_m = pltP['linestyle'][m]
            counts_m,_ = np.histogram(similarity_score[m].flatten(), bins=bin_edges)
            ax.plot(bin_centers+pltP['jitter'][m], counts_m,  color = cmap_m, ls = ls_m, lw = pltP['lw'])
        ax.grid(True, alpha=0.3)
