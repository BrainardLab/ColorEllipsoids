#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 16:05:29 2024

@author: fangfang
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import dill as pickled
import sys
import numpy as np

#load functions from other modules
sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids")
from core import optim, oddity_task
from core.model_predictions import wishart_model_pred
from analysis.color_thres import color_thresholds
from core.wishart_process import WishartProcessModel
from analysis.ellipses_tools import covMat_to_ellParamsQ, PointsOnEllipseQ
from plotting.adaptive_sampling_plotting import SamplingRefCompPairVisualization
from plotting.wishart_predictions_plotting import WishartPredictionsVisualization
sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/Python version/')
from analysis.trial_placement import TrialPlacementWithoutAdaptiveSampling

base_dir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'
output_figDir1 = base_dir + 'ELPS_analysis/Simulation_FigFiles/Python_version/2d_oddity_task/'
output_figDir2 = base_dir + 'ELPS_analysis/ModelFitting_FigFiles/Python_version/2D_oddity_task/random_ref/'
output_fileDir = base_dir + 'ELPS_analysis/ModelFitting_DataFiles/2D_oddity_task/random_ref/'
    
#%% -----------------------------------------------------------
# Load data simulated using CIELab and organize data
# -----------------------------------------------------------
#rng_seed = 0
for rng_seed in range(1):
    np.random.seed(rng_seed)  # Set the seed for numpy
    #set stimulus info
    color_dimension = 2
    plane_2D        = 'GB plane' 
    nSims           = 60 # Number of simulations or trials per reference stimulus.
    
    #file 1
    path_str   = base_dir + 'ELPS_analysis/Simulation_DataFiles/'
    # Create an instance of the class
    color_thres_data = color_thresholds(color_dimension, 
                                        base_dir + 'ELPS_analysis/',
                                        plane_2D = plane_2D)
    # Retrieve specific data from Wishart_data
    color_thres_data.load_model_fits()
    gt_Wishart  = color_thres_data.get_data('model_pred_Wishart', dataset = 'Wishart_data')
    gt_model = gt_Wishart.model
    gt_W = gt_Wishart.W_est
    gt_opt_params = gt_Wishart.opt_params
    
    #% simulate data
    # Define the boundaries for the reference stimuli
    xref_bds         = [-0.9, 0.9] # Square boundary limits for the stimuli
    # Amount of jitter added to comparison stimulus sampling
    sim_jitter       = 0.3
    # Total number of simulations to perform
    nSims_total      = int(nSims * gt_Wishart.num_grid_pts**color_dimension)
    # Draw random reference stimuli within the specified boundary
    xrand            = np.array(np.random.rand(nSims_total,2)*(xref_bds[-1]-xref_bds[0]) + xref_bds[0])
    # Compute the covariance matrices for each reference stimulus based on a model
    Sigmas_est_grid  = gt_model.compute_Sigmas(gt_model.compute_U(gt_W, xrand))
    
    # Initialize arrays to hold the comparison stimuli, predicted probabilities, and responses
    x1rand           = np.full(xrand.shape, jnp.nan)
    pX1              = np.full((nSims_total,), jnp.nan)
    resp             = np.full((nSims_total,), jnp.nan)
    paramEllipse_scaled = np.full((nSims_total, 5), jnp.nan)
    
    sim_trial = TrialPlacementWithoutAdaptiveSampling(skip_sim=True)
    sim_trial.sim = {'random_jitter': sim_jitter, 
                     'nSims': 1, 
                     'varying_RGBplane': color_thres_data.varying_RGBplane,
                     'slc_RGBplane': color_thres_data.fixed_color_dim,
                     'slc_fixedVal': color_thres_data.fixed_value}
    target_pC     = 2/3
    # Scaler used to adjust covariance matrices to approximate 66.7% correct responses
    # scaler_radii, pC_approx = sim_trial.compute_radii_scaler_to_reach_targetPC(target_pC,
    #                                                                            nsteps=100,
    #                                                                            visualize= True)
    scaler_radii = 2.56
    
    # Process each randomly sampled reference stimulus
    for i in range(nSims_total):
        # Scale the estimated covariance matrix for the current stimulus
        Sigmas_est_grid_i = Sigmas_est_grid[i]
        # Convert the covariance matrix to ellipse parameters (semi-major axis, 
        # semi-minor axis, rotation angle)
        _, _, ab_i, theta_i = covMat_to_ellParamsQ(Sigmas_est_grid_i)
        # scale radii
        ab_scaled_i = scaler_radii*ab_i
        # Pack the center of the ellipse with its parameters
        paramEllipse_scaled[i] = np.array([*xrand[i], *ab_scaled_i, theta_i])
        # Sample a comparison stimulus near the contour of the reference stimulus, applying jitter
        x1rand_temp,_,_,_ = sim_trial.sample_rgb_comp_2DNearContour(xrand[i],
                                                                    paramEllipse_scaled[i])
        # Reshape and clip the sampled comparison stimulus to fit within the [-1, 1] bounds        
        x1rand_reshape = x1rand_temp[color_thres_data.varying_RGBplane].T
        x1rand[i] = np.clip(x1rand_reshape,-1,1)
        
    xrand  = jnp.array(xrand)
    x1rand = jnp.array(x1rand)
    # compute weighted sum of basis function at rgb_ref 
    Uref   = gt_model.compute_U(gt_W, xrand)
    # compute weighted sum of basis function at rgb_comp
    U1     = gt_model.compute_U(gt_W, x1rand)
    # Predict the probability of choosing the comparison stimulus over the reference
    pX1    = oddity_task.oddity_prediction((xrand, x1rand, Uref, U1),
                      jax.random.split(gt_Wishart.opt_key, num = nSims_total),
                      gt_opt_params['mc_samples'], 
                      gt_opt_params['bandwidth'],
                      gt_model.diag_term, 
                      oddity_task.simulate_oddity)
    # Simulate a response based on the predicted probability
    randNum   = np.random.rand(*pX1.shape) 
    resp      = jnp.array((randNum < pX1).astype(int))
    
    # Package the processed data into a tuple for further use
    data_rand = (resp, xrand, x1rand)
    print(np.mean(resp))
    
    #% ------------------------------------------------
    # SECTION 3: Visualize trial placement
    # --------------------------------------------------
    sampling_vis = SamplingRefCompPairVisualization(color_dimension,
                                                    color_thres_data,
                                                    fig_dir = output_figDir1,
                                                    save_fig = False)
    
    # This array defines the opacity of markers in the plots, decreasing with more trials.
    marker_alpha = np.linspace(0.3, 1, 12)[::-1]
    # Define specific slices of data points to be visualized, ranging from very few to many.
    slc_datapoints_to_show = np.clip([2**i for i in range(12)],0,nSims_total)
    
    # Loop over the selected data points to generate and visualize each corresponding figure.
    for i,n in enumerate(slc_datapoints_to_show):
        # Construct a filename for each figure based on the plane and number of experiments.
        fig_name_i = f"Sims_isothreshold_{plane_2D}_im{n:04}total_samplingRandom_wFittedW_jitter{sim_jitter}"
        
        _, ax_i = plt.subplots(1, 1, figsize = (3,3.5), dpi= 256)
        if i == 5 or i == 6:
            for j in range(n):
                ell_i_x, ell_i_y = PointsOnEllipseQ(*paramEllipse_scaled[j][2:],
                                                    *paramEllipse_scaled[j][0:2])
                ax_i.plot(ell_i_x, ell_i_y, c= 'k',lw=0.5,alpha = 0.5)
        # Visualize the trials up to the nth data point with specified marker transparency.
        sampling_vis.plot_sampling(xrand[:n],   # Reference points up to the nth data point
                                   x1rand[:n],    # Comparison points up to the nth data point
                                   linealpha = marker_alpha[i],        # Line transparency for this subset of data
                                   comp_markeralpha = marker_alpha[i], # Marker transparency for this subset of data
                                   fig_name = fig_name_i + '.pdf',              # Filename under which the figure will be saved
                                   bounds = xref_bds,
                                   ref_markersize = 10,
                                   plane_2D = plane_2D,
                                   ax = ax_i)
        plt.show()
    
    #% -------------------------------
    # Constants describing simulation
    # -------------------------------
    model_test = WishartProcessModel(
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
    W_init = 1e-1*model_test.sample_W_prior(W_INIT_KEY) 
    
    opt_params = {
        "learning_rate": 1e-4,
        "momentum": 0.2,
        "mc_samples": MC_SAMPLES,
        "bandwidth": BANDWIDTH,
    }
    
    W_recover_wRandx, iters, objhist = optim.optimize_posterior(
        W_init, data_rand, model_test, OPT_KEY,
        opt_params,
        oddity_task.simulate_oddity, #oddity_task.simulate_oddity or oddity_task.simulate_oddity_reference
        total_steps=500,
        save_every=1,
        show_progress=True
    )
    
    fig, ax = plt.subplots(1, 1)
    ax.plot(iters, objhist)
    fig.tight_layout()
    
    #%
    # -----------------------------
    # Rocover covariance matrices
    # -----------------------------
    # Specify grid over stimulus space
    grid_1d = jnp.linspace(-0.6, 0.6, NUM_GRID_PTS)
    grid = jnp.stack(jnp.meshgrid(*[grid_1d for _ in range(model_test.num_dims)]), axis=-1)
    Sigmas_recover_wRandx = model_test.compute_Sigmas(model_test.compute_U(W_recover_wRandx, grid))
    
    # -----------------------------
    # Compute model predictions
    # -----------------------------
    model_pred_Wishart_wRandx = wishart_model_pred(model_test, opt_params, 
                                            NUM_GRID_PTS, MC_SAMPLES, 
                                            BANDWIDTH, W_INIT_KEY,
                                            DATA_KEY, OPT_KEY, W_init, 
                                            W_recover_wRandx, Sigmas_recover_wRandx, 
                                            color_thres_data,
                                            target_pC= target_pC,
                                            scaler_x1 = 5,
                                            ngrid_bruteforce = 500,
                                            bds_bruteforce = [0.01, 0.25])
    
    grid_trans = np.transpose(grid,(2,0,1))
    model_pred_Wishart_wRandx.convert_Sig_Threshold_oddity_batch(grid_trans)
    
    #%
    grid_1d_s = jnp.linspace(-0.75, 0.75, NUM_GRID_PTS+1)
    grid_s = jnp.stack(jnp.meshgrid(*[grid_1d_s for _ in range(gt_model.num_dims)]), axis=-1)
    Sigmas_recover_wRandx_s = model_test.compute_Sigmas(model_test.compute_U(W_recover_wRandx, grid_s))
    
    # -----------------------------
    # Compute model predictions
    # -----------------------------
    model_pred_Wishart_wRandx_s = wishart_model_pred(model_test, opt_params, 
                                            NUM_GRID_PTS+1,  MC_SAMPLES, 
                                            BANDWIDTH, W_INIT_KEY,
                                            DATA_KEY, OPT_KEY, W_init, 
                                            W_recover_wRandx, Sigmas_recover_wRandx_s, 
                                            color_thres_data,
                                            target_pC= target_pC,
                                            scaler_x1 = 5,
                                            ngrid_bruteforce = 500,
                                            bds_bruteforce = [0.01, 0.25])
    
    grid_trans_s = np.transpose(grid_s,(2,0,1))
    model_pred_Wishart_wRandx_s.convert_Sig_Threshold_oddity_batch(grid_trans_s)
    
    #% -----------------------------
    # Visualize model predictions
    # -----------------------------
    class sim_data:
        def __init__(self, xref_all, x1_all):
            self.xref_all = xref_all
            self.x1_all = x1_all
    sim_trial_by_Wishart = sim_data(xrand, x1rand)
    
    wishart_pred_vis = WishartPredictionsVisualization(sim_trial_by_Wishart,
                                                       model_test, 
                                                       model_pred_Wishart_wRandx, 
                                                       color_thres_data,
                                                       fig_dir = output_figDir2, 
                                                       save_fig = False)
    
    fig, ax = plt.subplots(1, 1, figsize = (3.1,3.6), dpi= 256)
    for i in range(NUM_GRID_PTS+1):
       for j in range(NUM_GRID_PTS+1):
           c_ij = np.array(color_thres_data.W_unit_to_N_unit(grid_trans_s[:,i,j]))
           c_ij = np.insert(c_ij, color_thres_data.fixed_color_dim, color_thres_data.fixed_value)
           # Plot the model predictions as lines.
           ax.plot(model_pred_Wishart_wRandx_s.fitEll_unscaled[i,j,0], 
                   model_pred_Wishart_wRandx_s.fitEll_unscaled[i,j,1],
                   c = c_ij,
                   lw =1,
                   alpha = 0.5, 
                   ls = '-')
    fig_name_fits = f"Fitted_isothreshold_{plane_2D}_sim{nSims_total}total_"+\
        f"samplingNearContour_noFixedRef_jitter{sim_jitter}_seed{rng_seed}_bandwidth{BANDWIDTH}"
    wishart_pred_vis.plot_2D(
        grid, 
        grid,
        gt_Wishart.fitEll_unscaled, 
        ax = ax,
        visualize_samples= False,
        visualize_gt = True,
        visualize_model_estimatedCov = False,
        samples_alpha = 1,
        samples_s = 1,
        plane_2D = plane_2D,
        modelpred_ls = '-',
        modelpred_lw = 2.5,
        modelpred_lc = None,
        mdoelpred_alpha = 1,
        gt_lw= 0.5,
        gt_lc =[0.1,0.1,0.1],
        fontsize = 8.5,
        fig_name = fig_name_fits + ".pdf") 
        
    #% save data
    output_file = fig_name_fits+'_oddity.pkl'
    full_path = f"{output_fileDir}{output_file}"
    
    variable_names = ['rng_seed','plane_2D', 'sim_jitter','nSims',
                      'nSims_total','color_thres_data', 'gt_Wishart', 'data_rand',
                      'pX1', 'sim_trial_by_Wishart', 'model_test', 'grid_1d',
                      'grid','grid_trans','grid_1d_s', 'grid_s',
                      'grid_trans_s', 'model_test','model_pred_Wishart_wRandx',
                      'model_pred_Wishart_wRandx_s', 'iters', 'objhist']
    
    vars_dict = {}
    for i in variable_names: vars_dict[i] = eval(i)
    
    # Write the list of dictionaries to a file using pickle
    with open(full_path, 'wb') as f:
        pickled.dump(vars_dict, f)

