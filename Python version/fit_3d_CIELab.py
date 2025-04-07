#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 17:01:45 2024

@author: fangfang
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import dill as pickled
import sys
import os
import numpy as np
from dataclasses import replace
#sys.path.append("/Users/fh862-adm/Documents/GitHub/ellipsoids/ellipsoids")
sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids')
from core import oddity_task, model_predictions, optim
from core.wishart_process import WishartProcessModel
from core.model_predictions import wishart_model_pred
from analysis.color_thres import color_thresholds
from plotting.wishart_plotting import PlotSettingsBase 
from plotting.wishart_predictions_plotting import WishartPredictionsVisualization, Plot3DPredSettings
#sys.path.append("/Users/fh862-adm/Documents/GitHub/ColorEllipsoids/Python version")
sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/Python version')
from data_reorg import organize_data_3d_fixed_grid, visualize_data_3d_fixed_grid,\
    derive_gt_slice_2d_ellipse_CIE
from analysis.cross_validation import expt_data

#%% three variables we need to define for loading the data
for rr in range(1):
    rnd_seed  = rr
    nSims     = 240
    jitter    = 0.3
    colordiff_alg = 'CIE1994'
    
    #base_dir = '/Users/fh862-adm/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'
    base_dir = '/Volumes/T9/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'
    output_figDir_fits = os.path.join(base_dir,'ELPS_analysis','ModelFitting_FigFiles',
                                      'Python_version','3D_oddity_task', colordiff_alg)
    output_fileDir = os.path.join(base_dir,'ELPS_analysis','ModelFitting_DataFiles',
                                  '3D_oddity_task', colordiff_alg)
    os.makedirs(output_fileDir, exist_ok= True)
    os.makedirs(output_figDir_fits, exist_ok = True)
    
    #% ------------------------------------------
    # Load data simulated using CIELab
    # ------------------------------------------
    # Create an instance of the class
    color_thres_data = color_thresholds(3, base_dir)
    # Load Wishart model fits
    color_thres_data.load_CIE_data(CIE_version= colordiff_alg)  
    stim3D = color_thres_data.get_data('stim3D', dataset='CIE_data')
    results3D = color_thres_data.get_data('results3D', dataset='CIE_data')
    
    #simulation files
    path_str = os.path.join(base_dir, 'ELPS_analysis','Simulation_DataFiles',
                            'ellipsoids', colordiff_alg)
    colordiff_alg_str = '' if colordiff_alg == 'CIE1976' else f'_{colordiff_alg}'
    file_sim = f'Sims_isothreshold_ellipsoids_sim{nSims}perCond_samplingNearContour_'+\
                f'jitter{jitter}_seed{rnd_seed}{colordiff_alg_str}.pkl'
    full_path = os.path.join(path_str, file_sim)
    with open(full_path, 'rb') as f: 
        data_load = pickled.load(f)
    sim = data_load['sim_trial'].sim
    
    """
    Fitting would be easier if we first scale things up, and then scale the model 
    predictions back down
    """
    #we take 5 samples from each color dimension
    #but sometimes we don't want to sample that finely. Instead, we might just pick
    #2 or 3 samples from each color dimension, and see how well the model can 
    #interpolate between samples
    idx_trim = list(range(5))
    #x1_raw is unscaled
    data_temp, x1_raw, xref_raw = organize_data_3d_fixed_grid(sim, slc_idx = idx_trim)
    # if we run oddity task with the reference stimulus fixed at the top
    y_jnp, xref_jnp, x0_jnp, x1_jnp = data_temp
    # if we run oddity task with all three stimuli shuffled
    data = (y_jnp, xref_jnp, x1_jnp)
    
    #% Visualize the simulated data again
    visualize_data_3d_fixed_grid(data_load['sim_trial'], fixed_val = 0.5)
    
    #%
    # -------------------------------
    # Constants describing simulation
    # -------------------------------
    model = WishartProcessModel(
        5,     # Degree of the polynomial basis functions #5
        3,     # Number of stimulus dimensions
        1,     # Number of extra inner dimensions in `U`. #1
        3e-4,  # Scale parameter for prior on `W`.
        0.4,   # Geometric decay rate on `W`.  #0.4
        0,     # Diagonal term setting minimum variance for the ellipsoids.
    )
    
    NUM_GRID_PTS = 5           # Number of grid points over stimulus space.
    MC_SAMPLES   = 2000        # Number of simulated trials to compute likelihood.
    BANDWIDTH    = 5e-3        # Bandwidth for logistic density function. #5e-3
    
    # Random number generator seeds
    W_INIT_KEY   = jax.random.PRNGKey(322)  # Key to initialize `W_est`. 
    DATA_KEY     = jax.random.PRNGKey(333)  # Key to generate datatset.
    OPT_KEY      = jax.random.PRNGKey(444)  # Key passed to optimizer.
    
    # -----------------------------
    # Fit W by maximum a posteriori
    # -----------------------------
    # Fit model, initialized at random W
    W_init = 1e-1*model.sample_W_prior(W_INIT_KEY)  
    
    opt_params = {
        "learning_rate": 1e-4, 
        "momentum": 0.2,
        "mc_samples": MC_SAMPLES,
        "bandwidth": BANDWIDTH,
    }
    W_est, iters, objhist = optim.optimize_posterior(
        W_init, data, model, OPT_KEY,
        opt_params,
        oddity_task.simulate_oddity,
        total_steps=1000,
        save_every=1,
        show_progress=True
    )
    
    fig, ax = plt.subplots(1, 1)
    ax.plot(iters, objhist)
    fig.tight_layout()
    plt.show()
    
    #% -----------------------------
    # Rocover covariance matrices
    # -----------------------------
    # Specify grid over stimulus space
    grid_1d = jnp.linspace(jnp.min(xref_jnp), jnp.max(xref_jnp), NUM_GRID_PTS)
    grid = jnp.stack(jnp.meshgrid(*[grid_1d for _ in range(model.num_dims)]), axis=-1)
    grid_trans = np.transpose(grid,(1,0,2,3))
    Sigmas_est_grid = model.compute_Sigmas(model.compute_U(W_est, grid))
    
    # -----------------------------
    # Compute model predictions
    # -----------------------------
    model_pred_Wishart = wishart_model_pred(model, opt_params, W_INIT_KEY,
                                            OPT_KEY, W_init, 
                                            W_est, Sigmas_est_grid, 
                                            color_thres_data,
                                            target_pC= 2/3,
                                            ngrid_bruteforce = 1000,
                                            bds_bruteforce = [0.0005, 0.3])
    
    model_pred_Wishart.convert_Sig_Threshold_oddity_batch(grid_trans)
    
    #% derive 2D slices: compute the 2D ellipse slices from the 3D covariance matrices 
    #for both ground truth and predictions
    gt_slice_2d_ellipse_CIE, gt_covMat_CIE = derive_gt_slice_2d_ellipse_CIE(NUM_GRID_PTS, 
                                             results3D, flag_convert_to_W= True)
    
    # compute the slice of model-estimated cov matrix
    model_pred_slice_2d_ellipse = model_predictions.covMat3D_to_2DsurfaceSlice(\
                                            model_pred_Wishart.Sigmas_recover_grid)
    model_pred_Wishart.Sigmas_recover_grid_slice_2d = np.transpose(model_pred_slice_2d_ellipse,(1,0,2,3,4,5))
        
    # ---------------------------------------------
    # plot figures and save them as png and gif
    # ---------------------------------------------
    pltSettings_base = PlotSettingsBase(fig_dir=output_figDir_fits, fontsize = 10)
    pred3D_settings = replace(Plot3DPredSettings(), **pltSettings_base.__dict__)
    fig_name = f'Fitted{file_sim[4:-4]}' #+'_maxDeg' + str(model.degree)

    sim_trial_by_CIE  = expt_data(xref_jnp, x1_jnp, y_jnp, None)
    wishart_pred_vis = WishartPredictionsVisualization(sim_trial_by_CIE,
                                                       model, 
                                                       model_pred_Wishart, 
                                                       color_thres_data,
                                                       settings = pltSettings_base,
                                                       save_fig = True)
    pred3D_settings = replace(pred3D_settings,
                              visualize_samples = True,
                              samples_s = 1,
                              samples_alpha = 0.2,
                              gt_ls = '--',
                              gt_lw = 1,
                              gt_lc = 'k',
                              gt_alpha = 0.85,#0.85
                              modelpred_alpha = 0.55,
                              modelpred_lc = None,
                              fig_name = fig_name)
            
    wishart_pred_vis.plot_3D(grid_trans, 
                             gt_covMat=gt_covMat_CIE, 
                             gt_slice_2d_ellipse=gt_slice_2d_ellipse_CIE, 
                             settings = pred3D_settings) 
        
    #% save data
    output_file = f"Fitted{file_sim[4:-4]}_bandwidth{BANDWIDTH}_oddity.pkl"
    full_path4 = os.path.join(output_fileDir, output_file)
    
    variable_names = ['color_thres_data','file_sim','data_temp','data', 
                      'sim_trial_by_CIE', 'grid_1d', 'grid','grid_trans','iters', 
                      'objhist','model_pred_Wishart','gt_covMat_CIE',
                      'gt_slice_2d_ellipse_CIE']
    vars_dict = {}
    for i in variable_names: vars_dict[i] = eval(i)
    
    # Write the list of dictionaries to a file using pickle
    with open(full_path4, 'wb') as f:
        pickled.dump(vars_dict, f)
    
