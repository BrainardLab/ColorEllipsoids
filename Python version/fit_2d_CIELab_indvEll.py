#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 18:18:20 2024

@author: fangfang

This script fits independent thresholds at each reference location using a reduced
Wishart Process Model, where only the first basis function is retained—corresponding 
to a uniform performance field.

Main steps:
    1. Load a .pkl file containing the full dataset.
    2. Optionally select a subset of the dataset for analysis.
    3. Choose whether to fit the independent-threshold model to:
        - the full dataset,
        - the selected subset, or
        - bootstrapped versions of the dataset.

Note:
    Unlike 'visualize_CI_indvEll_grid_3d.py', which separates plotting into a different script,
    this script includes plotting code directly, as the fitting process is relatively quick.

"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import dill as pickled
import sys
import copy
import numpy as np
from tqdm import trange
import os
from dataclasses import replace
#sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids")
sys.path.append("/Users/fh862-adm/Documents/GitHub/ellipsoids/ellipsoids")
from core import optim, oddity_task
from core.model_predictions import wishart_model_pred
from plotting.wishart_plotting import PlotSettingsBase 
from plotting.wishart_predictions_plotting import WishartPredictionsVisualization, Plot2DPredSettings
from analysis.ellipses_tools import find_inner_outer_contours
#sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/Python version')
sys.path.append("/Users/fh862-adm/Documents/GitHub/ColorEllipsoids/Python version")
from analysis.utils_load import load_4D_expt_data
from analysis.utils_load import select_file_and_get_path
from data_reorg import group_trials_by_grid

#define output directory for output files and figures
baseDir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'
COLOR_DIMENSION = 2

#%% 
# ----------------------------------------------
# SECTION 1: Load the data
# ----------------------------------------------
#'META_analysis/ModelFitting_DataFiles/2dTask/CIE/sub2/subset4000'
#'Fitted_byWishart_Isoluminant plane_49interleaved2DExpt_10_10_10_330_AEPsychSampling_EAVC_sub2_subset4000.pkl'
# It takes the file xxx_subset4000.pkl but we extract the full dataset from the pickle
fits_path, file_name = select_file_and_get_path()
Wishart_full_path = os.path.join(fits_path, file_name)

with open(Wishart_full_path, 'rb') as f:  
    vars_dict = pickled.load(f)
    
# Extract variables from loaded dictionary
NUM_GRID_PTS         = len(vars_dict['grid_ref'])        # Number of grid points per dimension
nRefs                = NUM_GRID_PTS ** COLOR_DIMENSION   # Total number of reference colors
data_AEPsych_fullset = vars_dict['data_AEPsych_fullset'] # Full dataset (e.g., ~18,000 trials)
color_thres_data     = vars_dict['color_thres_data']     # Color threshold data object
gt_Wishart           = vars_dict['gt_Wishart']           # Ground truth Wishart model (joint fit)
grid                 = vars_dict['grid']                 # 2D grid of reference points (shape: N x N x 2)
grid_flatten         = np.reshape(grid, (nRefs, -1))     # Flattened grid (shape: N^2 x 2)
NTRIALS_STRAT        = vars_dict['NTRIALS_STRAT']        # Trials per sampling strategy per reference (e.g., [10, 10, 10, 330])
NTRIALS              = sum(NTRIALS_STRAT)                # Total trials per reference
size_fullset         = NTRIALS * nRefs                   # Total number of trials across all references
model_pred_existing  = vars_dict['model_pred_Wishart']   # Existing model predictions
model_existing       = model_pred_existing.model         # Wishart model object
opt_params_existing  = model_pred_existing.opt_params    # Optimization parameters

#%% Define the subset size for fitting independent threshold models
# ----------------------------------------------
# SECTION 2: Select subset or the full dataset
# ----------------------------------------------
size_subset = 6027  # Example options: [6027, 10045, 14014, 17640]
str_ext_s = f'subset{size_subset}'

# Ensure the subset is evenly distributed across reference locations
assert size_subset % nRefs == 0, 'Selected trial number must be evenly divisible by number of reference locations.'

# Extract the subset of trials
data_AEPsych_subset = tuple(arr[:size_subset] for arr in data_AEPsych_fullset)
# Group trials by grid → returns (nRefs, N_perRef, 2) arrays
_, data_AEPsych_subset_flatten = group_trials_by_grid(grid,
                                                      *data_AEPsych_subset,
                                                      ndims = COLOR_DIMENSION)

# Generate output file name
output_file_name = f"Indv{file_name.split('subset')[0]}subset{size_subset}.pkl"

# Variables to save initially (before appending later model predictions)
variable_names = [
    'nRefs', 'NUM_GRID_PTS', 'data_AEPsych_fullset','data_AEPsych_subset_flatten',
    'color_thres_data','gt_Wishart', 'grid', 'grid_flatten', 'NTRIALS_STRAT', 'NTRIALS',
    'model_pred_existing', 'model_existing', 'opt_params_existing', 'size_subset'
]

# Define output directory and file path
output_fits_path = os.path.join(
    fits_path.replace('2dTask', '2dTask_indvEll').rpartition('/subset')[0],
    f'subset{size_subset}'
)
os.makedirs(output_fits_path, exist_ok=True)
output_full_path = os.path.join(output_fits_path, output_file_name)

#check if the file exists    
if os.path.exists(output_full_path):
    # File exists → load it
    with open(output_full_path, 'rb') as f:
        vars_dict_subset = pickled.load(f)
    flag_load_previous = True
    print(f"Loaded existing file: {output_full_path}")
else:
    # Save initial variables to pickle file
    vars_dict_subset = {var_name: eval(var_name) for var_name in variable_names}

    # File does not exist → write it
    with open(output_full_path, 'wb') as f:
        pickled.dump(vars_dict_subset, f)
    flag_load_previous = False
    print(f"Saved new file: {output_full_path}")
    
#%% some figure settings
#whether we load the data from a previous file or not, we need to set the directory for 
#output figures
# Create output directory for figures
output_figDir_fits = output_fits_path.replace('DataFiles', 'FigFiles')
os.makedirs(output_figDir_fits, exist_ok=True)

# Set plotting parameters for visualization
pltSettings_base = PlotSettingsBase(fig_dir=output_figDir_fits, fontsize=8)
predM_settings = replace(Plot2DPredSettings(), **pltSettings_base.__dict__)
predM_settings = replace(predM_settings, fig_dir=fits_path)

# Initialize the visualization object for model predictions
wishart_pred_vis = WishartPredictionsVisualization(
    data_AEPsych_subset, gt_Wishart.model, gt_Wishart, color_thres_data,
    settings=predM_settings, save_fig=False
)

# Refine plotting settings for individual fits
predM_settings = replace(
    predM_settings,
    visualize_samples=False,
    visualize_model_estimatedCov=False,
    visualize_gt=False,
    flag_rescale_axes_label=False,
    modelpred_lc='k',
    modelpred_ls='--',
    modelpred_lw=0.5,
    modelpred_alpha=1,
    modelpred_label='Ground truth (CIE 1994)',
    ticks=np.linspace(-0.7, 0.7, 5))

#%%
# -----------------------------------------
# SECTION 3: Fit individual ellipses
# -----------------------------------------
nBtst = 10
if not flag_load_previous:
    # Create a mask for the weight matrix W: only the first element is free; the rest are fixed.
    # This effectively reduces the Wishart model to an independent-threshold model: we assume
    # the threshold is constant everywhere within each reference location.
    base_shape = (model_existing.degree,) * COLOR_DIMENSION + (COLOR_DIMENSION,)
    W_mask = np.zeros(base_shape + (COLOR_DIMENSION + model_existing.extra_dims,), dtype=bool)
    W_mask[(0,) * COLOR_DIMENSION] = True  # only allow the first coefficient to vary
    
    # Set optimization hyperparameters
    nSteps   = 20   # number of gradient descent steps (fitting only 1 ellipse → doesn’t need many steps)
    nRepeats = 30   # number of initializations to avoid local minima
    
    # Variables to append to the pickle output for each bootstrap iteration
    variable_names_append = [
        'W_mask', 'nSteps', 'nRepeats', 'data_AEPsych', 'random_seeds', 'W_INIT_KEY',
        'OPT_KEY', 'bestfit_seed', 'W_init', 'W_est', 'Sigmas_est_grid', 'objhist',
        'fitEll', 'model_pred_Wishart_indv_ell','fitEll_params', 'fitEll_indv_org'
    ]
    
    #% Define bootstrap settings
    btst_seed = [None] + list(range(nBtst))#[None]  # add additional seeds if needed
    flag_btst = [False] + [True]*nBtst #[False]  # set to True to activate bootstrap
    
    for flag_btst_AEPsych, ll in zip(flag_btst, btst_seed):
        str_ext = str_ext_s
        if flag_btst_AEPsych:
            str_ext += f'_btst_AEPsych[{ll}]'
    
        # Bootstrap the data subset if requested
        # we cannot just bootstrap the 
        if flag_btst_AEPsych:
            y_btst, xref_btst, x1_btst = [],[],[]
            for n in range(nRefs):
                xref_n, x1_n, y_n, _ = load_4D_expt_data.bootstrap_AEPsych_data(
                    data_AEPsych_subset_flatten[1][n], data_AEPsych_subset_flatten[2][n], 
                    data_AEPsych_subset_flatten[0][n], trials_split=[sum(NTRIALS_STRAT[:-1])], 
                    seed=ll*100 + n
                )
                y_btst.append(y_n)
                xref_btst.append(xref_n)
                x1_btst.append(x1_n)
            y_jnp_org = jnp.stack(y_btst, axis = 0)
            xref_jnp_org = jnp.stack(xref_btst, axis = 0)
            x1_jnp_org = jnp.stack(x1_btst, axis = 0)
        else:
            y_jnp_org, xref_jnp_org, x1_jnp_org = data_AEPsych_subset_flatten
    
        # Group trials by grid → returns (nRefs, N_perRef, 2) arrays
        data_AEPsych = (y_jnp_org, xref_jnp_org, x1_jnp_org)
    
        # Generate random seeds for optimization initializations
        random_seeds = np.random.randint(0, 2**12, size=(nRefs, nRepeats, 2))
    
        # Initialize arrays to store fitting results
        W_INIT_KEY      = np.full((nRefs, 2), np.nan, dtype=np.uint32)
        OPT_KEY         = np.full((nRefs, 2), np.nan, dtype=np.uint32)
        bestfit_seed    = np.full((nRefs, 2), np.nan)
        W_init          = np.full((nRefs,) + base_shape + (COLOR_DIMENSION + model_existing.extra_dims,), np.nan)
        W_est           = np.full(W_init.shape, np.nan)
        Sigmas_est_grid = np.full((nRefs, ) + (NUM_GRID_PTS,)*COLOR_DIMENSION + (COLOR_DIMENSION, COLOR_DIMENSION,), np.nan)
        objhist         = np.full((nRefs, nSteps), np.nan)
        fitEll          = np.full((nRefs,) + gt_Wishart.fitEll_unscaled.shape[-2:], np.nan)
        fitEll_params   = np.full((nRefs, 5), np.nan)
        model_pred_Wishart_indv_ell = []
    
        # Loop over each reference location → fit an independent ellipse
        for n in trange(nRefs):
            model      = copy.deepcopy(model_existing)
            opt_params = copy.deepcopy(opt_params_existing)
    
            objhist_end = 1e3  # initialize to large value → any valid fit will be smaller
            ref_n = grid_flatten[n]
            data_n = (y_jnp_org[n], xref_jnp_org[n], x1_jnp_org[n])
    
            # Multiple random initializations for this reference location
            for nn in range(nRepeats):
                W_INIT_KEY_nn = jax.random.PRNGKey(random_seeds[n, nn, 0]) 
                OPT_KEY_nn = jax.random.PRNGKey(random_seeds[n, nn, 1])
    
                W_init_nn = model.sample_W_prior(W_INIT_KEY_nn)
    
                # Optimize model parameters
                W_est_nn, _, objhist_nn = optim.optimize_posterior(
                    W_init_nn, data_n, model, OPT_KEY_nn, opt_params,
                    oddity_task.simulate_oddity, total_steps=nSteps,
                    save_every=1, show_progress=False,
                    mask=W_mask, use_prior=False
                )
    
                # Reduce learning rate with each repeat (refinement)
                opt_params['learning_rate'] = 10 ** (-nn * 0.5 - 1)
    
                # Keep best fit (lowest final loss)
                if objhist_nn[-1] < objhist_end:
                    objhist_end = objhist_nn[-1]
                    W_INIT_KEY[n], OPT_KEY[n] = W_INIT_KEY_nn, OPT_KEY_nn
                    W_init[n], W_est[n] = W_init_nn, W_est_nn
                    objhist[n], bestfit_seed[n] = objhist_nn, random_seeds[n, nn]
    
            # Compute estimated covariance matrix for this reference
            Sigmas_est_grid[n] = model.compute_Sigmas(model.compute_U(W_est[n], grid))
    
            # Generate model predictions for this reference
            model_pred_Wishart_n = wishart_model_pred(model, opt_params,
                                                      W_INIT_KEY[n], 
                                                      OPT_KEY[n], 
                                                      W_init[n], W_est[n],
                                                      Sigmas_est_grid[n], 
                                                      color_thres_data,
                                                      target_pC=0.667, 
                                                      ngrid_bruteforce=1000, 
                                                      bds_bruteforce=[0.0005, 0.25]
            )    
            # Convert covariance to threshold contour
            model_pred_Wishart_n.convert_Sig_Threshold_oddity_batch(ref_n.reshape(model.num_dims, 1, 1))
            model_pred_Wishart_indv_ell.append(model_pred_Wishart_n)
            
            # Store the fitted ellipse
            fitEll[n] = model_pred_Wishart_n.fitEll_unscaled[0, 0]
            fitEll_params[n] = model_pred_Wishart_n.params_ell[0][0]
    
        # Reshape ellipse fits to original grid shape
        fitEll_indv_org = np.reshape(fitEll, (NUM_GRID_PTS, NUM_GRID_PTS, COLOR_DIMENSION) + gt_Wishart.fitEll_unscaled.shape[-1:])
    
        #---------------------------------------------------------------------------
        # Save results for this iteration
        vars_dict_subset_ll = {var_name: eval(var_name) for var_name in variable_names_append}
        key_name_ll = f'model_pred_indv_{str_ext}'
        
        # Check if the key already exists in the dictionary
        if key_name_ll in vars_dict_subset.keys():
            # Prompt the user to confirm overwriting the existing variable
            flag_overwrite = input("The variable name already exists in this file. Enter 'y' to confirm overwrite: ")

        # If the user confirms, update the dictionary and save it
        if flag_overwrite:
            vars_dict_subset[key_name_ll] = vars_dict_subset_ll  
            # Write updated dictionary to file
            with open(output_full_path, 'wb') as f:
                pickled.dump(vars_dict_subset, f)
            print(f"Saved updated vars_dict_subset to '{output_full_path}'.")
        else:
            # Do not update the dictionary if user didn't confirm
            print("The variable was not added to the pickle file.")

        #---------------------------------------------------------------------------
        # Visualize predictions: compare joint fit and individual fits    
        fig, ax = plt.subplots(1, 1, figsize=predM_settings.fig_size, dpi=predM_settings.dpi)
        wishart_pred_vis.plot_2D(grid, ax=ax, settings=predM_settings)
        for n in range(NUM_GRID_PTS):
            for m in range(NUM_GRID_PTS):
                cm_nm = color_thres_data.M_2DWToRGB @ np.insert(grid[n, m], 2, 1)
                ax.plot(fitEll_indv_org[n, m, 0], fitEll_indv_org[n, m, 1], alpha=0.7, c=cm_nm)
        ax.set_title('Isoluminant plane');
        fig.savefig(os.path.join(output_figDir_fits, f"{output_file_name[:-4]}_{str_ext}.pdf"))    

#%% visualize the model predictions with bootstrapped confidence intervals
flag_done = [True if f'model_pred_indv_subset{size_subset}_btst_AEPsych[{i}]' in vars_dict_subset else False for i in range(nBtst)]
if flag_load_previous or all(flag_done):
    fig2, ax2 = plt.subplots(1, 1, figsize=predM_settings.fig_size, dpi=predM_settings.dpi)
    #loop through each reference color
    for n in range(NUM_GRID_PTS):
        for m in range(NUM_GRID_PTS):
            #initialize
            fitEll_params_nm = np.full((nBtst, 5), np.nan) #5 means 5 free parameters characterizing an ellipse
            #extract
            for ll in range(nBtst):
                fitEll_params_org = vars_dict_subset[f'model_pred_indv_{str_ext_s}_btst_AEPsych{[ll]}']['fitEll_params']
                fitEll_params_reshape = np.reshape(fitEll_params_org, (NUM_GRID_PTS, NUM_GRID_PTS, 5))
                fitEll_params_nm[ll] = fitEll_params_reshape[n,m]
            #Computes the confidence intervals for the model-predicted ellipses at each grid point.
            xu, yu, xi, yi = find_inner_outer_contours(fitEll_params_nm)
            fitEll_min = np.vstack((xi, yi))
            fitEll_max = np.vstack((xu, yu))
            cm_nm = color_thres_data.M_2DWToRGB @ np.insert(grid[n, m], 2, 1)
            
            if n == 0 and m == 0:
                lbl = '100% bootstrap CI (10 datasets)' 
            else:
                lbl = None
            wishart_pred_vis.add_CI_ellipses(fitEll_min, fitEll_max,
                                             ax=ax2, cm=cm_nm, label=lbl, lw_outer = 0,
                                             alpha = 0.7)
    wishart_pred_vis.plot_2D(grid, ax=ax2, settings=predM_settings)
    ax2.set_title('Isoluminant plane');
    #fig2.savefig(os.path.join(output_figDir_fits, f"{output_file_name[:-4]}_wCI.pdf"))    













