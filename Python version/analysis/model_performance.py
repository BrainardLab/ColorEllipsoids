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
from scipy.stats import special_ortho_group
from dataclasses import dataclass
from typing import Tuple, List
sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids')
from analysis.ellipses_tools import ellParams_to_covMat, rotAngle_to_eigenvectors, ellParamsQ_to_covMat 

@dataclass
class PltBWDSettings:
    figsize: Tuple[float, float] = (3.6, 4)
    dpi: int = 1024
    y_upper_bound: float = 0.2
    flag_visualize_baseline: bool = False
    baseline_lw: float = 2
    baseline_alpha: float = 0.8
    errorbar_cs: float = 0
    errorbar_c: str = 'k'
    errorbar_m: str = 'o'
    errorbar_ms: float = 12
    errorbar_lw: float = 2
    axis_grid_alpha: float = 0.3
    x_tick_rot: int = 30
    dashed_ls: str = '--'
    dashed_lc: str = 'k'
    dashed_lw: float = 0.5
    x_label: str = 'Total number of trial'
    y_label: str = 'Bures-Wasserstein distance'

#%%
class ModelPerformance():
    def __init__(self, color_dimension, gt_ellParams):
        """
        Initializes the ModelPerformance object with the necessary data for 
        evaluating model performance against ground-truth ellipses or ellipsoids.
        
        Parameters:
        - color_dimension: int, dimensionality of the color space (2 for ellipses, 3 for ellipsoids).
        = gt_ellParams: list, ground truth ellipses
            2D size: len(gt_ellParams) = 7, len(gt_ellParams[0]) = 7, len(gt_ellParams[0],[0]) = 5 parameters 
            
        """
        self.ndims        = color_dimension 
        self.ellParams_gt = gt_ellParams
        if self.ndims == 2:
            self.ref_size = (len(gt_ellParams),len(gt_ellParams[0]))
        elif self.ndims == 3:
            self.ref_size = (len(gt_ellParams),len(gt_ellParams[0]), len(gt_ellParams[0][0]))
        
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
        self.BW_distance           = np.full((self.nLevels,)+ self.ref_size, np.nan)
        self.BW_distance_maxEigval = np.full(self.ref_size, np.nan)
        self.BW_distance_minEigval = np.full(self.ref_size, np.nan)
        
        self.LU_distance           = np.full((self.nLevels,) + self.ref_size, np.nan)
        self.LU_distance_maxEigval = np.full(self.ref_size, np.nan)
        self.LU_distance_minEigval = np.full(self.ref_size, np.nan)
        
        #ground truth covariance matrices and model predictions
        self.covMat_gt             = np.full(self.ref_size+(self.ndims, self.ndims,), np.nan)
        self.covMat_modelPred      = np.full((self.nLevels,)+self.ref_size+(self.ndims, self.ndims,), np.nan)
    
    def load_modelPreds_ellParams(self, ellParams_set, verbose = False):
        """
        Loads model predictions (ellipsoid or ellipse parameters) from the Wishart model
        and converts them to covariance matrices.
        
        Scales the radii by 1/2 because the Wishart model operates in the W space [-1, 1],
        while the ground-truth ellipses/ellipsoids are in the N space [0,1].
        """
        if self.ndims == 2:
            try:
                for l in range(self.nLevels): 
                    ellParams_l = ellParams_set[l]
                    for ii, jj in np.ndindex(self.ref_size):
                        # Retrieve predicted ellipse parameters (rotation → eigenvectors)
                        eigVec_jiijj = rotAngle_to_eigenvectors(ellParams_l[ii][jj][-1])
                        radii_jiijj = np.array(ellParams_l[ii][jj][2:4])
                        
                        # Sort radii and eigenvectors (enforce major/minor axis ordering)
                        radii_jiijj, eigVec_jiijj = ModelPerformance.sort_eig(radii_jiijj, eigVec_jiijj)
                
                        # Convert sorted ellipse params into covariance matrix
                        self.covMat_modelPred[l, ii, jj] = ellParams_to_covMat(radii_jiijj, eigVec_jiijj)
                
                        # If verbose mode is enabled, print radius comparison for first level
                        if verbose and l == 0:
                            _, radii_gt = self._convert_ellParams_to_covMat(self.ellParams_gt[ii][jj])
                            print(f"[i,j] = [{ii}, {jj}]")
                            print(f"Ground truths: {np.sort(radii_gt)}")
                            print(f"W Model preds: {np.sort(radii_jiijj)}")
            except:
                print('Cannot find ell parameters.')
        else:
            try:
                for l in range(self.nLevels):
                    ellParams_l = ellParams_set[l]
                    for ii, jj, kk in np.ndindex(self.ref_size):
                        eigVec_jiijjkk = ellParams_l[ii][jj][kk]['evecs']
                        radii_jiijjkk = ellParams_l[ii][jj][kk]['radii']
                        radii_jiijjkk, eigVec_jiijjkk = ModelPerformance.sort_eig(radii_jiijjkk, eigVec_jiijjkk)
                        
                        self.covMat_modelPred[l, ii, jj, kk] = ellParams_to_covMat(radii_jiijjkk, eigVec_jiijjkk)
                
                        if verbose and l == 0:
                            _, radii_gt = self._convert_ellParams_to_covMat(self.ellParams_gt[ii][jj][kk])
                            print(f"[i,j,k] = [{ii}, {jj}, {kk}]")
                            print(f"Ground truths: {np.sort(radii_gt)}")
                            print(f"W Model preds: {np.sort(radii_jiijjkk)}")
            except:
                print('Cannot find ell parameters.')
                                
    def _convert_ellParams_to_covMat(self, ellParams):
        """
        Converts ellipse or ellipsoid parameters into a covariance matrix.
        
        Scales the radii by a specified factor (default is 5) and sorts the radii and
        corresponding eigenvectors in descending order.
        """
        if self.ndims == 2:
            _, _, a, b, R = ellParams
            radii = np.array([a, b])
            eigvecs = rotAngle_to_eigenvectors(R)
        else:
            radii = ellParams['radii']
            eigvecs = ellParams['evecs']
        
        # Sort radii and eigenvectors in descending order.
        radii, eigvecs = ModelPerformance.sort_eig(radii, eigvecs)
        
        # Convert to covariance matrix.
        covMat = ellParams_to_covMat(radii, eigvecs)
        return covMat, radii
                
    def compare_with_extreme_ell(self, ell1Params):  
        """
        Compares the ground-truth ellipse/ellipsoid to extreme cases (largest and smallest eigenvalue),
        generating covariance matrices for bounding spheres and computing performance metrics.
        
        Returns covariance matrix, Bures-Wasserstein and Log-Euclidean distances for both extreme cases.
        """

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
            BW_distance_corner[m] = self.compute_Bures_Wasserstein_distance(covMat_corner[m],covMat_gt)
            LU_distance_corner[m] = self.log_operator_norm_distance(covMat_corner[m],covMat_gt)
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

    def evaluate_model_performance(self, gt_ellParams_set, covMat_corner = None):
        """
        Evaluates the overall performance of the model by comparing the ground truth 
        with model predictions using both Bures-Wasserstein and Log-Euclidean distances.
        
        Optionally compares with ellipsoids at selected corner locations.
        """
        # Initialize arrays for storing model performance results.
        self.nLevels = len(gt_ellParams_set)
        self._initialize()
        
        # Load model predictions and convert them to covariance matrices.
        self.load_modelPreds_ellParams(gt_ellParams_set)
        
        # If corner ellipsoids are provided, initialize arrays for storing corner distances.
        if covMat_corner is not None:
            self.nCorners = len(covMat_corner)
            self.BW_distance_corner = np.full((self.nCorners,) + self.BW_distance_maxEigval.shape, np.nan)
            self.LU_distance_corner = np.full(self.BW_distance_corner.shape, np.nan)
        
        for idx in np.ndindex(self.ref_size):
            # Unpack indices
            if self.ndims == 2:
                ii, jj = idx
                try:
                    self.covMat_gt[ii,jj], self.BW_distance_minEigval[ii,jj], \
                        self.BW_distance_maxEigval[ii,jj], self.LU_distance_minEigval[ii,jj], \
                        self.LU_distance_maxEigval[ii,jj] = self.compare_with_extreme_ell(
                            self.ellParams_gt[ii][jj])
                    
                    if covMat_corner is not None:
                        self.BW_distance_corner[:,ii,jj], self.LU_distance_corner[:,ii,jj] = \
                            self.compare_with_corner_ell(self.covMat_gt[ii,jj], covMat_corner)
                    
                    self.BW_distance[:,ii,jj], self.LU_distance[:,ii,jj] = \
                        self.compare_gt_model_pred_one_instance(self.covMat_gt[ii,jj], 
                                                               self.covMat_modelPred[:,ii,jj])
                except:
                    print('Cannot find ell parameters.')
            else:
                ii, jj, kk = idx
                try:
                    self.covMat_gt[ii,jj,kk], self.BW_distance_minEigval[ii,jj,kk], \
                        self.BW_distance_maxEigval[ii,jj,kk], self.LU_distance_minEigval[ii,jj,kk], \
                        self.LU_distance_maxEigval[ii,jj,kk] = self.compare_with_extreme_ell(
                            self.ellParams_gt[ii][jj][kk])
                    
                    if covMat_corner is not None:
                        self.BW_distance_corner[:,ii,jj,kk], self.LU_distance_corner[:,ii,jj,kk] = \
                            self.compare_with_corner_ell(self.covMat_gt[ii,jj,kk], covMat_corner)
                    
                    self.BW_distance[:,ii,jj,kk], self.LU_distance[:,ii,jj,kk] = \
                        self.compare_gt_model_pred_one_instance(self.covMat_gt[ii,jj,kk], 
                                                               self.covMat_modelPred[:,ii,jj,kk])       
                except:
                    print('Cannot find ell parameters.')
        
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
            ax.hist(similarity_score[j].flatten(), bins = bin_edges,
                    color = cmap_l, alpha = pltP['alpha'], edgecolor = 'w',
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
            'ls':[],
            'ls_median': [],
            'jitter':np.zeros((nSets)),
            'lw': 1,
            } 
        pltP.update(kwargs)
        
        #fig, ax = plt.subplots(1, 1, figsize=(5,4.5))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        for m in range(nSets):
            if len(pltP['cmap']) == 0: cmap_m = np.random.rand(1,3)
            else: cmap_m = pltP['cmap'][m];
            if len(pltP['ls']) == 0: ls = '--';
            else: ls = pltP['ls'][m]
            if len(pltP['ls_median']) == 0: ls_m = '-';
            else: ls_m = pltP['ls_median'][m]            
            median_m = np.median(similarity_score[m].flatten())
            counts_m,_ = np.histogram(similarity_score[m].flatten(), bins=bin_edges)
            ax.plot(bin_centers+pltP['jitter'][m], counts_m,  color = cmap_m, ls = ls, lw = pltP['lw'])
            ax.plot([median_m, median_m], [0, 80], ls = ls_m, color = cmap_m, lw = pltP['lw'])
        ax.grid(True, alpha=0.3)

    @staticmethod
    def generate_ellipses_within_BWdistance(ellipse_gt, target_bw_dist, min_axis_len, max_axis_len,
                         max_trials=10000, tol=1e-4, seed=None, num_ellipses=1):
        """
        Generate ellipses whose Bures-Wasserstein distance to the ground truth ellipse
        is close to the target distance.

        Parameters:
            ellipse_gt: tuple (center, (a_gt, b_gt), theta_gt)
            target_bw_dist: float
            min_axis_len: float
            max_axis_len: float
            max_trials: int
            tol: float
            seed: int or None
            num_ellipses: int (number of ellipses to generate)

        Returns:
            ellipses: list of tuples [(center, (a, b), theta), ...]
            distances: list of corresponding BW distances
        """
        if seed is not None:
            np.random.seed(seed)
        
        a_gt, b_gt, theta_gt, center_x_gt, center_y_gt = ellipse_gt
        cov_gt = ellParamsQ_to_covMat(a_gt, b_gt, theta_gt)
        
        ellipses = []
        distances = []
        
        attempts = 0
        while len(ellipses) < num_ellipses and attempts < max_trials:
            # Randomly sample axis lengths and angle within bounds
            a = np.random.uniform(min_axis_len, max_axis_len)
            b = np.random.uniform(min_axis_len, max_axis_len)
            theta = np.random.uniform(0, 180)
            
            cov_sim = ellParamsQ_to_covMat(a, b, theta)
            
            bw_dist = ModelPerformance.compute_Bures_Wasserstein_distance(cov_gt, cov_sim)
            
            if np.isclose(bw_dist, target_bw_dist, atol=tol):
                ellipses.append((center_x_gt,center_y_gt, a, b, theta))
                distances.append(bw_dist)
            
            attempts += 1
        
        if len(ellipses) < num_ellipses:
            print(f"Only found {len(ellipses)} ellipse(s) within {max_trials} trials.")
        
        return ellipses, distances
    
    @staticmethod
    def generate_ellipsoids_within_BWdistance(gt_ellipsoid, target_bw_dist, min_axis_len, max_axis_len,
                                              max_trials=10000, tol=1e-4, seed=None, num_ellipsoids=1):
        """
        Generate ellipsoids in 3D whose BW distance to the ground truth ellipsoid is close to the target.

        Parameters:
            gt_ellipsoid: dict with 'radii', 'evecs', and 'center'
            target_bw_dist: float
            min_axis_len: float
            max_axis_len: float
            max_trials: int
            tol: float
            seed: int or None
            num_ellipsoids: int

        Returns:
            ellipsoids: list of dicts with 'radii', 'evecs', 'center'
            distances: list of BW distances
        """
        if seed is not None:
            np.random.seed(seed)

        # Ground truth covariance matrix: Σ = R * diag(radii^2) * R^T
        radii_gt = gt_ellipsoid['radii']
        evecs_gt = gt_ellipsoid['evecs']
        center_gt = gt_ellipsoid['center'].flatten()

        cov_gt = evecs_gt @ np.diag(radii_gt**2) @ evecs_gt.T

        ellipsoids = []
        distances = []

        attempts = 0
        while len(ellipsoids) < num_ellipsoids and attempts < max_trials:
            # Sample radii and a random rotation matrix
            radii = np.random.uniform(min_axis_len, max_axis_len, size=3)
            evecs = special_ortho_group.rvs(3)  # Random orthonormal 3x3 matrix

            cov_sim = evecs @ np.diag(radii**2) @ evecs.T
            bw_dist = ModelPerformance.compute_Bures_Wasserstein_distance(cov_gt, cov_sim)

            if np.isclose(bw_dist, target_bw_dist, atol=tol):
                ellipsoids.append({
                    'radii': radii,
                    'evecs': evecs,
                    'center': center_gt.copy()
                })
                distances.append(bw_dist)

            attempts += 1

        if len(ellipsoids) < num_ellipsoids:
            print(f"Only found {len(ellipsoids)} ellipsoid(s) within {max_trials} trials.")

        return ellipsoids, distances