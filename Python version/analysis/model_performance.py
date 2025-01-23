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
    def __init__(self, color_dimension, gt_results, gt_stim, varying_levels, 
                 plane_2D = None, isgt_CIE = True, verbose = True):
        """
        Initializes the ModelPerformance object with the necessary data for 
        evaluating model performance against ground-truth ellipses or ellipsoids.
        
        Parameters:
        - color_dimension: int, dimensionality of the color space (2 for ellipses, 3 for ellipsoids).
        - gt_results: dict, contains the ground-truth ellipse/ellipsoid parameters.
        - gt_stim: dict, contains stimulus-related data (e.g., grid size).
        - varying_levels: list, specifies the different jitter (noise level)
        - plane_2D: str, specifies the 2D plane (optional, e.g., 'GB plane', 'RB plane').
        - isgt_CIE: bool, if True, the ground truths are assumed to be CIE
                          if False, the ground truths could be the wishart fits of pilot data
        - verbose: bool, if True, prints additional information during model performance evaluation.
        """
        self.ndims       = color_dimension 
        self.gt_results  = gt_results        
        self.gt_stim     = gt_stim
        self.plane_2D    = plane_2D
        self.levels      = varying_levels
        self.nLevels     = len(varying_levels)
        self.verbose     = verbose 
        self.isgt_CIE    = isgt_CIE
        
        # If a 2D plane is specified, retrieve its corresponding index.
        if plane_2D is not None:
            plane_2D_dict = {'GB plane': 0, 'RB plane': 1, 'RG plane': 2}
            self.plane_2D_idx  = plane_2D_dict[plane_2D]
            
        # Set up reference grid size and retrieve ground-truth ellipse/ellipsoid parameters.
        if self.isgt_CIE:
            self.ref_size = self.gt_stim['nGridPts_ref'] 
            self._retrieve_ellParams_gt_CIE()
        else:
            self.ref_size = self.gt_stim.num_grid_pts1
            self._retrieve_ellParams_gt_WishartFits()
            
        
    def _retrieve_ellParams_gt_CIE(self):
        """
        Retrieves ground-truth ellipsoid or ellipse parameters based on the color dimensionality.
        """
        if self.ndims == 2:
            # For 2D, select the appropriate plane's ellipses.
            self.ellParams_gt = self.gt_results['ellParams'][self.plane_2D_idx]
        else:
            # For 3D, retrieve the ellipsoid parameters directly.
            self.ellParams_gt = self.gt_results['ellipsoidParams']
            
    def _retrieve_ellParams_gt_WishartFits(self):
        self.ellParams_gt = self.gt_results.params_ell
        
    def _initialize(self):
        """
        This method initializes arrays that save model performance results
        BW_distance: Bures-Wasserstein distance between the ground-truth and model predictions
        BW_distance_maxEigval: BW distance between the ground-truth and ellipses/ellipsoids with the largest eigenvalue
        BW_distance_minEigval: BW distance between the ground-truth and ellipses/ellipsoids with the smallest eigenvalue
        LU_distance: Log-Euclidean distance between the ground-truth and model predictions
        LU_distance_maxEigval: LU distance between the ground-truth and ellipses/ellipsoids with the largest eigenvalue
        LU_distance_minEigval: LU distance between the ground-truth and ellipses/ellipsoids with the smallest eigenvalue

        """
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
        
        #ground truth covariance matrices and model predictions
        self.covMat_gt             = np.full(base_shape+(self.ndims, self.ndims,), np.nan)
        self.covMat_modelPred      = np.full((self.nLevels,)+base_shape+(self.ndims, self.ndims,), np.nan)
    
    def load_modelPreds_ellParams(self, data_load, scaler_x1 = 5):
        """
        Loads model predictions (ellipsoid or ellipse parameters) from the Wishart model
        and converts them to covariance matrices.
        
        Scales the radii by 1/2 because the Wishart model operates in the W space [-1, 1],
        while the ground-truth ellipses/ellipsoids are in the N space [0,1].
        """
        if self.ndims == 2:
            for l in range(self.nLevels): 
                for ii in range(self.ref_size):
                    for jj in range(self.ref_size):
                        # Retrieve predicted ellipse parameters.
                        try:
                            #l_pred = data_load[l]['model_pred_Wishart']
                            l_pred = data_load[l]
                        except:
                            try:
                                l_pred = data_load[l]['model_pred_Wishart_wRandx']
                            except:
                                l_pred = data_load[l]['model_pred_indvEll']
                        eigVec_jiijj = rotAngle_to_eigenvectors(l_pred.params_ell[ii][jj][-1])
                        
                        #if the ground truth is based on CIE lab, then we need to divide the 
                        #radii by 2 because CIELab is bounded within [0 1] and Wishart fits 
                        #are bounded within [-1 1]
                        if self.isgt_CIE:
                            radii_jiijj = np.array(l_pred.params_ell[ii][jj][2:4])/2
                        else:
                            radii_jiijj = np.array(l_pred.params_ell[ii][jj][2:4])
                        # Sort the radii and eigenvectors.
                        radii_jiijj, eigVec_jiijj = ModelPerformance.sort_eig(radii_jiijj, eigVec_jiijj)
                        # Convert the sorted parameters into a covariance matrix.
                        self.covMat_modelPred[l,ii,jj] = ellParams_to_covMat(radii_jiijj, eigVec_jiijj)
                        # Print radii comparison if verbose mode is enabled.
                        if self.verbose and l == 0:
                            _, radii_gt = self._convert_ellParams_to_covMat(self.ellParams_gt[ii][jj], scaler_x1)
                            print(f"[i,j] = [{ii}, {jj}]")
                            print(f"Ground truths: {np.sort(radii_gt)}")
                            print(f"W Model preds: {np.sort(radii_jiijj)}")
        else:
            for l in range(self.nLevels):
                for ii in range(self.ref_size):
                    for jj in range(self.ref_size):
                        for kk in range(self.ref_size):
                            # Retrieve predicted ellipse parameters.
                            try:
                                l_pred = data_load[l]['model_pred_Wishart']
                            except:
                                try:
                                    l_pred = data_load[l]['model_pred_Wishart_wRandx']
                                except:
                                    l_pred = data_load[l]['model_pred_indvEll']
                            eigVec_jiijjkk = l_pred.params_ell[ii][jj][kk]['evecs']
                            radii_jiijjkk = l_pred.params_ell[ii][jj][kk]['radii']/2     
                            radii_jiijjkk, eigVec_jiijjkk = ModelPerformance.sort_eig(radii_jiijjkk, eigVec_jiijjkk)
                            self.covMat_modelPred[l,ii,jj,kk] = \
                                ellParams_to_covMat(radii_jiijjkk, eigVec_jiijjkk)
                            if self.verbose and l == 0:
                                _, radii_gt = self._convert_ellParams_to_covMat(self.ellParams_gt[ii][jj][kk])
                                print(f"[i,j,k] = [{ii}, {jj}, {kk}]")
                                print(f"Ground truths: {np.sort(radii_gt)}")
                                print(f"W Model preds: {np.sort(radii_jiijjkk)}")
                                
    def _convert_ellParams_to_covMat(self, ellParams, scaler_x1 = 5):
        """
        Converts ellipse or ellipsoid parameters into a covariance matrix.
        
        Scales the radii by a specified factor (default is 5) and sorts the radii and
        corresponding eigenvectors in descending order.
        """
        if self.ndims == 2:
            _, _, a, b, R = ellParams
            radii = np.array([a, b]) * scaler_x1
            eigvecs = rotAngle_to_eigenvectors(R)
        else:
            radii = ellParams['radii'] * scaler_x1
            eigvecs = ellParams['evecs']
        
        # Sort radii and eigenvectors in descending order.
        radii, eigvecs = ModelPerformance.sort_eig(radii, eigvecs)
        
        # Convert to covariance matrix.
        covMat = ellParams_to_covMat(radii, eigvecs)
        return covMat, radii
                
    def compare_with_extreme_ell(self, ell1Params, scaler_x1 = 5):  
        """
        Compares the ground-truth ellipse/ellipsoid to extreme cases (largest and smallest eigenvalue),
        generating covariance matrices for bounding spheres and computing performance metrics.
        
        Returns covariance matrix, Bures-Wasserstein and Log-Euclidean distances for both extreme cases.
        """

        # Use the eigenvalues and eigenvectors to derive the cov matrix
        covMat_gt, radii_gt = self._convert_ellParams_to_covMat(ell1Params, scaler_x1)
            
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
            
        LU_distance_maxEigval = self.log_operator_norm_distance(covMat_gt, covMat_max)       
        LU_distance_minEigval = self.log_operator_norm_distance(covMat_gt, covMat_min)     
        
        return covMat_gt, BW_distance_minEigval, BW_distance_maxEigval,\
            LU_distance_minEigval, LU_distance_maxEigval
    
    def compare_with_corner_ell(self, covMat_gt, covMat_corner):
        """
        Compares the ground-truth covariance matrix with corner ellipsoids and computes the
        Bures-Wasserstein and Log-Euclidean distances for each corner.
        
        Returns arrays of distances for each corner.
        """
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
        """
        Compares the ground-truth covariance matrix to model predictions for each level,
        computing Bures-Wasserstein and Log-Euclidean distances.
        
        Returns arrays of distances for each model level.
        """
        BW_distance = np.full((self.nLevels,), np.nan)
        LU_distance = np.full((self.nLevels,), np.nan)
        for l in range(self.nLevels):    
            BW_distance[l] = self.compute_Bures_Wasserstein_distance(\
                covMat_gt,covMat_modelPred[l])
            LU_distance[l] = self.log_operator_norm_distance(covMat_gt,
                                                             covMat_modelPred[l])
        return BW_distance, LU_distance

    def evaluate_model_performance(self, model_pred_data, covMat_corner = None, scaler_x1 = 5):
        """
        Evaluates the overall performance of the model by comparing the ground truth 
        with model predictions using both Bures-Wasserstein and Log-Euclidean distances.
        
        Optionally compares with ellipsoids at selected corner locations.
        """
        # Initialize arrays for storing model performance results.
        self._initialize()
        
        # Load model predictions and convert them to covariance matrices.
        self.load_modelPreds_ellParams(model_pred_data, scaler_x1)
        
        # If corner ellipsoids are provided, initialize arrays for storing corner distances.
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
                        self.compare_with_extreme_ell(self.ellParams_gt[ii][jj], scaler_x1)
                    
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
                            self.compare_with_extreme_ell(self.ellParams_gt[ii][jj][kk], scaler_x1)
                        
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
    def sort_eig(radii, eigvecs, order='descending'):
        # Sort radii and eigenvectors
        sorted_indices = np.argsort(radii)
        if order == 'descending':
            sorted_indices = sorted_indices[::-1]  # Reverse for descending order
        radii_sorted = radii[sorted_indices]
        eigvecs_sorted = eigvecs[:, sorted_indices]
        return radii_sorted, eigvecs_sorted
    
    @staticmethod
    def compute_Bures_Wasserstein_distance(M1, M2):
        # Compute the square root of M1
        sqrt_M1 = sqrtm(M1)
        # Compute the product sqrt(M1) * M2 * sqrt(M1)
        product = sqrt_M1 @ M2 @ sqrt_M1
        # Compute the square root of the product
        sqrt_product = sqrtm(product)
        # Ensure the result is real
        if np.iscomplexobj(sqrt_product):
            sqrt_product = np.real(sqrt_product)
            print(M1)
            print(M2)
            
        # Calculate the Bures-Wasserstein distance
        trace_diff = np.trace(M1) + np.trace(M2) - 2 * np.trace(sqrt_product)
        trace_diff = max(0, trace_diff)  # Avoid negative values under sqrt
        
        BW_distance = np.sqrt(trace_diff)        
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
