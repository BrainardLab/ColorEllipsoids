#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 18:18:20 2024

@author: fangfang
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import dill as pickle
import sys
import numpy as np
import os
#sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids")
sys.path.append("/Users/fh862-adm/Documents/GitHub/ellipsoids/ellipsoids")
from analysis.color_thres import color_thresholds
from analysis.ellipses_tools import PointsOnEllipseQ, fit_2d_isothreshold_contour
from plotting.wishart_predictions_plotting import WishartPredictionsVisualization
# Import functions and classes from your project
from core.probability_surface import IndividualProbSurfaceModel, optimize_nloglikelihood
#sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/Python version')
sys.path.append("/Users/fh862-adm/Documents/GitHub/ColorEllipsoids/Python version")
from analysis.utils_load import select_file_and_get_path
from data_reorg import group_trials_by_grid

baseDir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'

#'/Users/fh862-adm/Aguirre-Brainard Lab Dropbox/Fangfang Hong/META_analysis/ModelFitting_DataFiles/2dTask/CIE/sub2/subset6000'
#'Fitted_byWishart_Isoluminant plane_49interleaved2DExpt_10_10_10_330_AEPsychSampling_EAVC_sub2.pkl'
fits_path, file_name = select_file_and_get_path()
Wishart_full_path = os.path.join(fits_path, file_name)

with open(Wishart_full_path, 'rb') as f:  
    vars_dict = pickle.load(f)
nRefs = vars_dict['NCONFIGS']
NUM_GRID_PTS = int(np.sqrt(nRefs))
data_AEPsych_fullset = vars_dict['data_AEPsych_fullset']
color_thres_data = vars_dict['color_thres_data']
grid = vars_dict['grid']

size_subset_indvEll = 6027
nSims = size_subset_indvEll//nRefs

data_AEPsych_subset_indvEll = (data_AEPsych_fullset[0][:size_subset_indvEll],
                               data_AEPsych_fullset[1][:size_subset_indvEll],
                               data_AEPsych_fullset[2][:size_subset_indvEll])

data_AEPsych_subset_indvEll_orgGrid, data_AEPsych_subset_indvEll_orgFlatten = \
    group_trials_by_grid(grid, *data_AEPsych_subset_indvEll)

#%% load ground truth
#'/Users/fh862-adm/Aguirre-Brainard Lab Dropbox/Fangfang Hong/ELPS_analysis/ModelFitting_DataFiles/2D_oddity_task/Isoluminant plane'
#'Fitted_isothreshold_Isoluminant plane_CIE1994_sim18000total_samplingNearContour_jitter0.3_seed0_bandwidth0.005_decay0.4_oddity.pkl'
gt_fits_path, gt_file_name = select_file_and_get_path()
gt_full_path = os.path.join(gt_fits_path, gt_file_name)
with open(gt_full_path, 'rb') as f:  
    gt_vars_dict = pickle.load(f)
sim = gt_vars_dict['sim']
gt_Wishart = gt_vars_dict['model_pred_Wishart_grid7']

#%% three variables we need to define for loading the data

for rr in range(10):
    #%-------------------------------
    # Constants describing simulation
    # -------------------------------
    # Initialize an instance of IndividualProbSurfaceModel
    model_indvEll = IndividualProbSurfaceModel(NUM_GRID_PTS, 
                                            [1e-2,0.3],  # Bounds for radii
                                            [0, 2*jnp.pi],  # Bounds for angle in radians
                                            [0.5, 5],  # Bounds for Weibull parameter 'a'
                                            [0.1, 5])  # Bounds for Weibull parameter 'b'
    
    # -----------------------------
    # Fit W by maximizing posterior
    # -----------------------------
    # Set Weibull parameters a and b (these control threshold and slope)
    weibull_params = jnp.array([sim['alpha'], sim['beta']])  # Store Weibull parameters in an array
    #a = 3.189; b = 1.505
    
    nReps = 10
    KEY_list = list(range(nReps))  
    objhist = np.full((20000,), 1)
    for k in KEY_list:    
        print(f'Reptition {k}:')                                 
        KEY_k = jax.random.PRNGKey(k)  # Initialize random key for reproducibility
        init_params = model_indvEll.sample_params_prior(KEY_k)  # Sample initial parameters for the model
        
        # Run optimization to recover the best-fit parameters
        params_recover_k, iters, objhist_k = optimize_nloglikelihood(
            init_params, data_AEPsych_subset_indvEll, 
            total_steps=20000,           # Number of total optimization steps
            save_every=10,               # Save the objective value every 10 steps
            fixed_weibull_params=weibull_params,  # Fix the Weibull parameters during optimization
            bds_radii = model_indvEll.bds_radii,
            bds_angle = model_indvEll.bds_angle,
            learning_rate = 1e-1,
            show_progress=True           # Show progress using tqdm
        )
        if objhist_k[-1] < objhist[-1]: 
            objhist = objhist_k
            params_recover = params_recover_k
    
    # Plot the optimization history (objective value vs iterations)
    fig, ax = plt.subplots(1, 1)
    ax.plot(iters, objhist)  # Plot iterations vs objective history
    fig.tight_layout()
    
    #%
    # -----------------------------
    # Compute model predictions
    # -----------------------------
    # Recover ellipses from the optimized parameters
    nTheta = 200
    xy_recover = np.full((nRefs, 2, nTheta), np.nan)  # Initialize array to store recovered ellipses
    for i in range(nRefs):
        # Reconstruct the recovered ellipses using the optimized parameters
        x_recover, y_recover = PointsOnEllipseQ(*params_recover[i,0:2],  # Semi-major and semi-minor axes
                                       jnp.rad2deg(params_recover[i,2]),  # Convert angle back to degrees
                                       *xref_jnp[i,0], nTheta= nTheta)       # Center of ellipse, number of points
        
        xy_recover[i] = jnp.stack((x_recover, y_recover))  # Store recovered ellipse points
    fitEll_unscaled = np.reshape(xy_recover,(NUM_GRID_PTS, NUM_GRID_PTS,2, nTheta))    
    
    #fit an ellipse, get paramQ
    grid_1d = jnp.linspace(-0.6, 0.6, NUM_GRID_PTS)
    grid = jnp.stack(jnp.meshgrid(*[grid_1d for _ in range(2)]), axis=-1)
    
    #initialize
    ell_paramsQ = []
    fitEll_scaled = np.full(fitEll_unscaled.shape, np.nan)
    for i in range(NUM_GRID_PTS):
        ell_paramsQ_i = []
        for j in range(NUM_GRID_PTS):
            rgb_comp_ij = fitEll_unscaled[i,j]
            fitEll_scaled_ij, _, _, _, ellP_ij = \
                fit_2d_isothreshold_contour(grid[i,j], [], rgb_comp_ij)
            fitEll_scaled[i,j] = fitEll_scaled_ij
            ell_paramsQ_i.append(ellP_ij)
        ell_paramsQ.append(ell_paramsQ_i)
        
    #class for model prediction
    class model_pred:
        def __init__(self, M, fitEll_unscaled, fitEll_scaled, params_ell, target_pC = 2/3):
            self.fitM = M
            self.fitEll_unscaled = fitEll_unscaled
            self.fitEll_scaled = fitEll_scaled
            self.params_ell = params_ell
            self.target_pC = target_pC
    model_pred_indvEll = model_pred(params_recover, fitEll_unscaled, fitEll_scaled, ell_paramsQ)
            
    #%
    # -----------------------------
    # Visualize model predictions
    # -----------------------------
    wishart_pred_vis = WishartPredictionsVisualization(data_AEPsych_subset_indvEll,
                                                       model_indvEll, 
                                                       model_pred_indvEll, 
                                                       color_thres_data,
                                                       fig_dir = output_figDir_fits, 
                                                       save_fig = True)
    #specify figure name and path
    fig_name_part1 = 'Fitted' + file_sim[4:-4]
    
    wishart_pred_vis.plot_2D(
        grid, 
        gt_Wishart.fitEll_scaled, 
        visualize_samples= True,
        visualize_gt = False,
        visualize_model_estimatedCov = False,
        samples_alpha = 1,
        samples_s = 1,
        plane_2D = color_thres_data.plane_2D,
        modelpred_ls = '-',
        modelpred_lc = [0.3,0.3,0.3],
        modelpred_lw = 2,
        modelpred_alpha = 0.8,
        gt_lw= 0.5,
        gt_lc =[0.1,0.1,0.1],
        fig_name = fig_name_part1+'_indvEll_withSamples.pdf') 
            
    #save data
    
    # variable_names = ['plane_2D', 'sim_jitter','nSims', 'data','x1_raw',
    #                   'xref_raw','data_new','weibull_params','sim_trial_by_CIE',
    #                   'grid_1d', 'grid', 'iters', 'objhist','model_indvEll',
    #                   'model_pred_indvEll', 
    #                   'gt_covMat_CIE']
    # vars_dict = {}
    # for i in variable_names: vars_dict[i] = eval(i)
    
    # # Write the list of dictionaries to a file using pickle
    # with open(full_path, 'wb') as f:
    #     pickle.dump(vars_dict, f)
