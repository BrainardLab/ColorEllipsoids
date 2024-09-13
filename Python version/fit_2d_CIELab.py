#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 11:52:44 2024

@author: fangfang

This fits a Wishart Process model to the simulated data using the CIELab color space. 

"""

#%% import modules
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import dill as pickle
import sys
import numpy as np

sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids")
from core import optim, oddity_task
from core.wishart_process import WishartProcessModel
from analysis.color_thres import color_thresholds
from core.model_predictions import wishart_model_pred
from plotting.wishart_predictions_plotting import WishartPredictionsVisualization
sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/Python version')
from data_reorg import organize_data

#%% three variables we need to define for loading the data
plane_2D      = 'RB plane'
plane_2D_dict = {'GB plane': 0, 'RB plane': 1, 'RG plane': 2}
plane_2D_idx  = plane_2D_dict[plane_2D]
sim_jitter    = '0.3'
nSims         = 80 #number of simulations: 240 trials for each ref stimulus

for rr in range(2,10):
    rnd_seed      = rr
    
    baseDir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'
    output_figDir_fits = baseDir + 'ELPS_analysis/ModelFitting_FigFiles/Python_version/2D_oddity_task/'
    output_fileDir = baseDir + 'ELPS_analysis/ModelFitting_DataFiles/2D_oddity_task/'
    
    #%%
    # -----------------------------------------------------------
    # Load data simulated using CIELab and organize data
    # -----------------------------------------------------------
    #file 1
    path_str      = baseDir + 'ELPS_analysis/Simulation_DataFiles/'
    # Create an instance of the class
    color_thres_data = color_thresholds(2, baseDir + 'ELPS_analysis/',
                                        plane_2D = plane_2D)
    # Load Wishart model fits
    color_thres_data.load_CIE_data()
    results = color_thres_data.get_data('results2D', dataset = 'CIE_data')  
    
    #file 2
    file_sim      = f"Sims_isothreshold_{plane_2D}_sim{nSims}perCond_"+\
                    f"samplingNearContour_jitter{sim_jitter}_seed{rnd_seed}.pkl"
    full_path     = f"{path_str}{file_sim}"   
    with open(full_path, 'rb') as f:  data_load = pickle.load(f)
    sim = data_load[0]
    
    """
    Fitting would be easier if we first scale things up, and then scale the model 
    predictions back down
    """
    scaler_x1  = 5
    data, x1_raw, xref_raw = organize_data(2, sim, scaler_x1,
                                           visualize_samples = True,
                                           plane_2D = plane_2D)
    ref_size_dim1, ref_size_dim2 = x1_raw.shape[0:2]
    y_jnp, xref_jnp, x0_jnp, x1_jnp = data 
    data_new = (y_jnp, xref_jnp, x1_jnp)
    
    #%% -------------------------------
    # Constants describing simulation
    # -------------------------------
    model = WishartProcessModel(
        5,     # Degree of the polynomial basis functions
        2,     # Number of stimulus dimensions
        1,     # Number of extra inner dimensions in `U`.
        3e-4,  # Scale parameter for prior on `W`.
        0.4,   # Geometric decay rate on `W`. 
        0,     # Diagonal term setting minimum variance for the ellipsoids.
    )
    
    NUM_GRID_PTS = 5      # Number of grid points over stimulus space.
    MC_SAMPLES   = 2000        # Number of simulated trials to compute likelihood.
    BANDWIDTH    = 5e-3        # Bandwidth for logistic density function.
    
    # Random number generator seeds
    W_INIT_KEY   = jax.random.PRNGKey(227)  # Key to initialize `W_est`. 227
    DATA_KEY     = jax.random.PRNGKey(333)  # Key to generate datatset.
    OPT_KEY      = jax.random.PRNGKey(444)  # Key passed to optimizer.
    
    # -----------------------------
    # Fit W by maximizing posterior
    # -----------------------------
    # Fit model, initialized at random W
    W_init = 1e-1*model.sample_W_prior(W_INIT_KEY) 
    
    opt_params = {
        "learning_rate": 1e-3,#1e-2
        "momentum": 0.2,
        "mc_samples": MC_SAMPLES,
        "bandwidth": BANDWIDTH,
    }
    W_est, iters, objhist = optim.optimize_posterior(
        W_init, data_new, model, OPT_KEY,
        opt_params,
        oddity_task.simulate_oddity, #oddity_task.simulate_oddity or oddity_task.simulate_oddity_reference
        total_steps=500,
        save_every=1,
        show_progress=True
    )
    
    fig, ax = plt.subplots(1, 1)
    ax.plot(iters, objhist)
    fig.tight_layout()
    
    #%%
    # -----------------------------
    # Rocover covariance matrices
    # -----------------------------
    # Specify grid over stimulus space
    grid_1d = jnp.linspace(jnp.min(xref_jnp), jnp.max(xref_jnp), NUM_GRID_PTS)
    grid = jnp.stack(jnp.meshgrid(*[grid_1d for _ in range(model.num_dims)]), axis=-1)
    Sigmas_est_grid = model.compute_Sigmas(model.compute_U(W_est, grid))
    
    # -----------------------------
    # Compute model predictions
    # -----------------------------
    model_pred_Wishart = wishart_model_pred(model, opt_params, NUM_GRID_PTS, 
                                            MC_SAMPLES, BANDWIDTH, W_INIT_KEY,
                                            DATA_KEY, OPT_KEY, W_init, 
                                            W_est, Sigmas_est_grid, 
                                            color_thres_data,
                                            target_pC=2/3,
                                            scaler_x1 = scaler_x1,
                                            ngrid_bruteforce = 500,
                                            bds_bruteforce = [0.01, 0.25])
    
    grid_trans = np.transpose(grid,(2,0,1))
    model_pred_Wishart.convert_Sig_Threshold_oddity_batch(grid_trans)
    
            
    #%%
    # -----------------------------
    # Visualize model predictions
    # -----------------------------
    #ground truth ellipses
    gt_covMat_CIE = color_thresholds.N_unit_to_W_unit(results['fitEllipse_scaled'][plane_2D_idx])
    #specify figure name and path
    fig_name_part1 = 'Fitted' + file_sim[4:-4] + '_bandwidth' + str(BANDWIDTH)
    
    class sim_data:
        def __init__(self, xref_all, x1_all):
            self.xref_all = xref_all
            self.x1_all = x1_all
    sim_trial_by_CIE = sim_data(xref_jnp, x1_jnp)
    
    wishart_pred_vis = WishartPredictionsVisualization(sim_trial_by_CIE,
                                                         model, 
                                                         model_pred_Wishart, 
                                                         color_thres_data,
                                                         fig_dir = output_figDir_fits, 
                                                         save_fig = False)
            
    wishart_pred_vis.plot_2D(
        grid, 
        grid,
        gt_covMat_CIE, 
        visualize_samples= True,
        visualize_gt = False,
        visualize_model_estimatedCov = True,
        samples_alpha = 1,
        samples_s = 1,
        plane_2D = plane_2D,
        modelpred_ls = '-',
        modelpred_lc = [0.3,0.3,0.3],
        modelpred_lw = 2,
        mdoelpred_alpha = 1,
        gt_lw= 0.5,
        gt_lc =[0.1,0.1,0.1],
        fig_name = fig_name_part1+'_withSamples.pdf') 
    
    wishart_pred_vis.plot_2D(
        grid, 
        grid,
        gt_covMat_CIE, 
        visualize_samples= False,
        visualize_model_estimatedCov= False,
        visualize_gt = True,
        gt_lw = 1, 
        gt_alpha = 0.5, 
        modelpred_lw = 2, 
        modelpred_alpha = 0.5,
        fig_name = fig_name_part1 + '.pdf') 
    
            
    #%% save data
    output_file = fig_name_part1 + "_oddity.pkl"
    full_path = f"{output_fileDir}{output_file}"
    
    variable_names = ['plane_2D', 'sim_jitter','nSims', 'data','x1_raw',
                      'xref_raw','sim_trial_by_CIE', 'grid_1d', 'grid','grid_trans',
                      'iters', 'objhist','model','model_pred_Wishart', 'gt_covMat_CIE']
    vars_dict = {}
    for i in variable_names: vars_dict[i] = eval(i)
    
    # Write the list of dictionaries to a file using pickle
    with open(full_path, 'wb') as f:
        pickle.dump(vars_dict, f)
    
