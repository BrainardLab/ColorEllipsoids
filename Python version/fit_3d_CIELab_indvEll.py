#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 16:54:24 2024

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

To visualize the results, use: visualize_CI_indvEll_grid_3d.py
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
sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids")
#sys.path.append("/Users/fh862-adm/Documents/GitHub/ellipsoids/ellipsoids")
from core import optim, oddity_task
from core.model_predictions import wishart_model_pred
sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/Python version')
#sys.path.append("/Users/fh862-adm/Documents/GitHub/ColorEllipsoids/Python version")
from analysis.utils_load import load_4D_expt_data
from analysis.utils_load import select_file_and_get_path
from data_reorg import group_trials_by_grid

#define output directory for output files and figures
#baseDir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'
base_dir = '/Volumes/T9/Aguirre-Brainard Lab Dropbox/Fangfang Hong/ELPS_analysis/'
COLOR_DIMENSION = 3

#%% load data
# ----------------------------------------------
# SECTION 1: load data
# ----------------------------------------------
#'META_analysis/ModelFitting_DataFiles/3dTask/CIE'
#'Fitted_byWishart_ellipsoids_3DExpt_30_30_30_550_AEPsychSampling_EAVC_decayRate0.4_nBasisDeg5_sub1_subset30000.pkl'
# It takes the file xxx_subset30000.pkl but we extract the full dataset from the pickle
fits_path, file_name = select_file_and_get_path()
Wishart_full_path = os.path.join(fits_path, file_name)

with open(Wishart_full_path, 'rb') as f:  
    vars_dict = pickled.load(f)
    
# Extract variables from loaded dictionary
NUM_GRID_PTS         = len(vars_dict['grid_ref'])                 # Number of grid points per dimension
nRefs                = NUM_GRID_PTS ** COLOR_DIMENSION            # Total number of reference colors
data_AEPsych_fullset = vars_dict['data_AEPsych_fullset']          # Full dataset (e.g., ~18,000 trials)
color_thres_data     = vars_dict['color_thres_data']              # Color threshold data object
gt_Wishart           = vars_dict['gt_Wishart']                    # Ground truth Wishart model (joint fit)
grid                 = vars_dict['grid']                          # 2D grid of reference points (shape: N x N x 2)
grid_flatten         = np.reshape(grid, (nRefs, -1))              # Flattened grid (shape: N^2 x 2)
NTRIALS_STRAT        = vars_dict['NTRIALS_STRAT']                 # Trials per sampling strategy per reference (e.g., [10, 10, 10, 330])
NTRIALS              = sum(NTRIALS_STRAT)                         # Total trials per reference
size_fullset         = NTRIALS * nRefs                            # Total number of trials across all references
model_pred_existing  = vars_dict['model_pred_Wishart']            # Existing model predictions
model_existing       = model_pred_existing.model                  # Wishart model object
opt_params_existing  = model_pred_existing.opt_params             # Optimization parameters

#%% Define the subset size for fitting independent threshold models
# ----------------------------------------------
# SECTION 2: Select subset or the full dataset
# ----------------------------------------------
size_subset = 45000  # Example options: [30000, 45000, 60000, 80000]
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
    'NUM_GRID_PTS', 'nRefs', 'data_AEPsych_fullset', 'color_thres_data','gt_Wishart', 
    'grid', 'grid_flatten', 'NTRIALS_STRAT', 'NTRIALS', 'model_pred_existing',
    'model_existing', 'opt_params_existing', 'size_subset','data_AEPsych_subset_flatten'
]

# Define output directory and file path
output_fits_path = os.path.join(
    fits_path.replace(f'{COLOR_DIMENSION}dTask', f'{COLOR_DIMENSION}dTask_indvEll').rpartition('/subset')[0],
    f'subset{size_subset}'
)
os.makedirs(output_fits_path, exist_ok=True)
output_full_path = os.path.join(output_fits_path, output_file_name)

#check if the file exists    
if os.path.exists(output_full_path):
    # File exists → load it
    with open(output_full_path, 'rb') as f:
        vars_dict_subset = pickled.load(f)
    print(f"Loaded existing file: {output_full_path}")
else:
    # Save initial variables to pickle file
    vars_dict_subset = {var_name: eval(var_name) for var_name in variable_names}

    # File does not exist → write it
    with open(output_full_path, 'wb') as f:
        pickled.dump(vars_dict_subset, f)
    print(f"Saved new file: {output_full_path}")
    print(f"This pickle contains {vars_dict_subset.keys()}")

#%% three variables we need to define for loading the data
# -----------------------------------------
# SECTION 3: Fit individual ellipses
# -----------------------------------------
# Create a mask for the weight matrix W: only the first element is free; the rest are fixed.
# This effectively reduces the Wishart model to an independent-threshold model: we assume
# the threshold is constant everywhere within each reference location.
base_shape = (model_existing.degree,) * COLOR_DIMENSION + (COLOR_DIMENSION,)
W_mask = np.zeros(base_shape + (COLOR_DIMENSION + model_existing.extra_dims,), dtype=bool)
W_mask[(0,) * COLOR_DIMENSION] = True  # only allow the first coefficient to vary

# Set optimization hyperparameters
nSteps   = 20   # number of gradient descent steps (fitting only 1 ellipse → doesn’t need many steps)
nRepeats = 20   # number of initializations to avoid local minima

# Variables to append to the pickle output for each bootstrap iteration
variable_names_append = [
    'W_mask', 'nSteps', 'nRepeats', 'data_AEPsych', 'random_seeds', 'W_INIT_KEY',
    'OPT_KEY', 'bestfit_seed', 'W_init', 'W_est', 'Sigmas_est_grid', 'objhist',
    'fitEll', 'model_pred_Wishart_indv_ell','fitEll_params', 'fitEll_indv_org'
]

#% Define bootstrap settings
nBtst = 1
btst_seed = [None] + list(range(nBtst))  #[None]  # add additional seeds if needed
flag_btst = [False] + [True]*nBtst       #[False] # set to True to activate bootstrap
flag_plot_debug = True

for flag_btst_AEPsych, ll in zip(flag_btst, btst_seed):
    str_ext = str_ext_s
    if flag_btst_AEPsych:
        str_ext += f'_btst_AEPsych[{ll}]'

    # Bootstrap the data subset if requested
    if flag_btst_AEPsych:
        #we need to bootstrap for each reference to make sure each one has the same
        #trial number
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
        #after we do that for each reference location, stack the bootstrapped
        #trials together for each reference
        y_jnp_org = jnp.stack(y_btst, axis = 0)
        xref_jnp_org = jnp.stack(xref_btst, axis = 0)
        x1_jnp_org = jnp.stack(x1_btst, axis = 0)
    else:
        #if we do not want to bootstrap, just use the original dataset
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
    fitEll_params   = []
    model_pred_Wishart_indv_ell = []

    # Loop over each reference location → fit an independent ellipse
    for n in trange(nRefs):
        #copy the Wishart model fits
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
            W_est_nn, iters_nn, objhist_nn = optim.optimize_posterior(
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
        model_pred_Wishart_n.convert_Sig_Threshold_oddity_batch(ref_n[np.newaxis, np.newaxis, np.newaxis,:])
        model_pred_Wishart_indv_ell.append(model_pred_Wishart_n)

        # Store the fitted ellipse
        fitEll[n] = model_pred_Wishart_n.fitEll_unscaled[(0,) * COLOR_DIMENSION]
        fitEll_params.append(model_pred_Wishart_n.params_ell[0][0][0])
        
        #debug plotting to show the ellipsoid
        if flag_plot_debug:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(fitEll[n][0], fitEll[n][1], fitEll[n][2], s=1, alpha = 0.5)  # s=1 makes dots small
            ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
            ax.set_aspect('equal'); plt.show()

    # Reshape ellipse fits to original grid shape
    fitEll_indv_org = np.reshape(fitEll, (NUM_GRID_PTS,)*COLOR_DIMENSION +\
                                 (COLOR_DIMENSION,) + gt_Wishart.fitEll_unscaled.shape[-1:])

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
        vars_dict_subset[key_name_ll] = vars_dict_subset_ll  # Overwrite or add the variable
        with open(output_full_path, 'wb') as f:
            pickled.dump(vars_dict_subset, f)  # Save the updated dictionary to file
        print(f"Saved updated vars_dict_subset to '{output_full_path}'.")
    else:
        # Do not update the dictionary if user didn't confirm
        print("The variable was not added to the pickle file.")


