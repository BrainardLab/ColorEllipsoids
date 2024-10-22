#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 18:18:20 2024

@author: fangfang
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
from analysis.color_thres import color_thresholds
from analysis.ellipses_tools import PointsOnEllipseQ, fit_2d_isothreshold_contour
from plotting.wishart_predictions_plotting import WishartPredictionsVisualization
# Import functions and classes from your project
from core.probability_surface import IndividualProbSurfaceModel, optimize_nloglikelihood
sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/Python version')
from data_reorg import organize_data


#%% three variables we need to define for loading the data
plane_2D      = 'RG plane'
plane_2D_dict = {'GB plane': 0, 'RB plane': 1, 'RG plane': 2}
plane_2D_idx  = plane_2D_dict[plane_2D]
sim_jitter    = '0.3'
nSims         = 480 #number of simulations: 240 trials for each ref stimulus

for rr in range(10):
    rnd_seed = rr
    
    baseDir = '/Volumes/T9/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'
    output_figDir_fits = baseDir + 'ELPS_analysis/ModelFitting_FigFiles/Python_version/2D_oddity_task_indvEll/'
    output_fileDir = baseDir + 'ELPS_analysis/ModelFitting_DataFiles/2D_oddity_task_indvEll/'
    
    #%
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
    y_jnp_flat, xref_jnp_flat, x0_jnp_flat, x1_jnp_flat = data 
    nRefs    = y_jnp_flat.shape[0]//nSims
    y_jnp    = jnp.reshape(y_jnp_flat, (nRefs, nSims))
    xref_jnp = jnp.reshape(xref_jnp_flat, (nRefs, nSims, 2))
    x1_jnp   = jnp.reshape(x1_jnp_flat, (nRefs, nSims,2))
    data_new = (y_jnp, xref_jnp[:,0,:], x1_jnp)
    
    #%-------------------------------
    # Constants describing simulation
    # -------------------------------
    NUM_GRID_PTS = int(np.sqrt(nRefs))
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
    #a = 1.17; b = 2.33    
    
    nReps = 10
    KEY_list = list(range(nReps))  
    objhist = np.full((20000,), 1)
    for k in KEY_list:    
        print(f'Reptition {k}:')                                 
        KEY_k = jax.random.PRNGKey(k)  # Initialize random key for reproducibility
        init_params = model_indvEll.sample_params_prior(KEY_k)  # Sample initial parameters for the model
        
        # Run optimization to recover the best-fit parameters
        params_recover_k, iters, objhist_k = optimize_nloglikelihood(
            init_params, data_new, 
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
                fit_2d_isothreshold_contour(grid[i,j], [], rgb_comp_ij, 
                                            scaler_x1= 1/scaler_x1)
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
    # Retrieve ground truth
    # -----------------------------
    #ground truth ellipses
    gt_covMat_CIE = color_thresholds.N_unit_to_W_unit(results['fitEllipse_scaled'][plane_2D_idx])
    
    class sim_data:
        def __init__(self, xref_all, x1_all):
            self.xref_all = xref_all
            self.x1_all = x1_all
    sim_trial_by_CIE = sim_data(xref_jnp_flat, x1_jnp_flat)
    
    #%
    # -----------------------------
    # Visualize model predictions
    # -----------------------------
    wishart_pred_vis = WishartPredictionsVisualization(sim_trial_by_CIE,
                                                         model_indvEll, 
                                                         model_pred_indvEll, 
                                                         color_thres_data,
                                                         fig_dir = output_figDir_fits, 
                                                         save_fig = True)
    #specify figure name and path
    fig_name_part1 = 'Fitted' + file_sim[4:-4]
    
    wishart_pred_vis.plot_2D(
        grid, 
        grid,
        gt_covMat_CIE, 
        visualize_samples= True,
        visualize_gt = False,
        visualize_model_estimatedCov = False,
        samples_alpha = 1,
        samples_s = 1,
        plane_2D = plane_2D,
        modelpred_ls = '-',
        modelpred_lc = [0.3,0.3,0.3],
        modelpred_lw = 2,
        modelpred_alpha = 0.8,
        gt_lw= 0.5,
        gt_lc =[0.1,0.1,0.1],
        fig_name = fig_name_part1+'_indvEll_withSamples.pdf') 
            
    #% save data
    output_file = fig_name_part1 + "_oddity_indvEll.pkl"
    full_path = f"{output_fileDir}{output_file}"
    
    variable_names = ['plane_2D', 'sim_jitter','nSims', 'data','x1_raw',
                      'xref_raw','data_new','weibull_params','sim_trial_by_CIE',
                      'grid_1d', 'grid', 'iters', 'objhist','model_indvEll',
                      'model_pred_indvEll', 
                      'gt_covMat_CIE']
    vars_dict = {}
    for i in variable_names: vars_dict[i] = eval(i)
    
    # Write the list of dictionaries to a file using pickle
    with open(full_path, 'wb') as f:
        pickle.dump(vars_dict, f)
