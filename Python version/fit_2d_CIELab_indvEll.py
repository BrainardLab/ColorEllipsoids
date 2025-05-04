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
#sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/Python version')
sys.path.append("/Users/fh862-adm/Documents/GitHub/ColorEllipsoids/Python version")
from analysis.utils_load import load_4D_expt_data
from analysis.utils_load import select_file_and_get_path
from data_reorg import group_trials_by_grid

#define output directory for output files and figures
baseDir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'
COLOR_DIMENSION = 2

#%% load data
#'/Users/fh862-adm/Aguirre-Brainard Lab Dropbox/Fangfang Hong/META_analysis/ModelFitting_DataFiles/2dTask/CIE/sub2/subset4000'
#'Fitted_byWishart_Isoluminant plane_49interleaved2DExpt_10_10_10_330_AEPsychSampling_EAVC_sub2.pkl'
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

# Define the subset size for fitting independent threshold models
size_subset = 6027  # Example options: [6027, 10045, 14014, 17640]
str_ext_s = f'_subset{size_subset}' if size_subset < size_fullset else ''

# Ensure the subset is evenly distributed across reference locations
assert size_subset % nRefs == 0, 'Selected trial number must be evenly divisible by number of reference locations.'

# Extract the subset of trials
data_AEPsych_subset = tuple(arr[:size_subset] for arr in data_AEPsych_fullset)

# Generate output file name
output_file_name = f"Indv{file_name.split('subset')[0]}subset{size_subset}.pkl"

# Variables to save initially (before appending later model predictions)
variable_names = [
    'nRefs', 'NUM_GRID_PTS', 'data_AEPsych_fullset', 'color_thres_data',
    'gt_Wishart', 'grid', 'grid_flatten', 'NTRIALS_STRAT', 'NTRIALS',
    'model_pred_existing', 'model_existing', 'opt_params_existing', 'size_subset'
]

# Define output directory and file path
output_fits_path = os.path.join(
    fits_path.replace('2dTask', '2dTask_indvEll').rpartition('/subset')[0],
    f'subset{size_subset}'
)
os.makedirs(output_fits_path, exist_ok=True)
output_full_path = os.path.join(output_fits_path, output_file_name)

# Save initial variables to pickle file
vars_dict_subset = {var_name: eval(var_name) for var_name in variable_names}
with open(output_full_path, 'wb') as f:
    pickled.dump(vars_dict_subset, f)
    
#%% three variables we need to define for loading the data
# -----------------------------------------
# SECTION 2: Fit individual ellipses
# -----------------------------------------
# Initialize a mask for the weight matrix W, where only the first element is unconstrained.
# by doing so, we reduce the Wishart model to be the independent-threshold model. we assume
# the threshold is the same everywhere and thus we just create one model instance for each
# reference color, and fit the model to the data at that specific ref location
base_shape = (model_existing.degree, model_existing.degree, 2)
W_mask = np.zeros(base_shape + (COLOR_DIMENSION + model_existing.extra_dims,), dtype=bool)
W_mask[0,0] = True

# Define the number of repeats and steps for the optimization and initialize arrays to 
# store results for each reference location
nSteps   = 20  #number of iterations. We are fitting only 1 ellipse, so we do not need that many iterations
nRepeats = 30  #start from different initializations to avoid being stuck at the local minima

#variables that we append to the pickle
variable_names_append = ['W_mask', 'nSteps', 'nRepeats', 'xref_jnp', 'x1_jnp',
                         'y_jnp', 'y_jnp_org', 'xref_jnp_org', 'x1_jnp_org',
                         'data_AEPsych', 'random_seeds', 'W_INIT_KEY ','OPT_KEY',
                         'bestfit_seed', 'W_init', 'W_est', 'Sigmas_est_grid', 
                         'objhist','fitEll', 'model_pred_Wishart_indv_ell','fitEll_indv_org']

# Set up output directory for saving figures
output_figDir_fits = fits_path.replace('DataFiles', 'FigFiles')
os.makedirs(output_figDir_fits, exist_ok=True)

pltSettings_base = PlotSettingsBase(fig_dir=fits_path, fontsize = 8)
predM_settings = replace(Plot2DPredSettings(), **pltSettings_base.__dict__)
predM_settings = replace(predM_settings, fig_dir = fits_path)
wishart_pred_vis = WishartPredictionsVisualization(data_AEPsych_subset,
                                                   gt_Wishart.model, 
                                                   gt_Wishart, 
                                                   color_thres_data,
                                                   settings = predM_settings,
                                                   save_fig = True)
predM_settings = replace(predM_settings,
                         visualize_samples= False,
                         visualize_model_estimatedCov= False,
                         visualize_gt = False,
                         flag_rescale_axes_label = False,
                         modelpred_lc = 'k',
                         modelpred_ls = '--',
                         modelpred_lw = 0.5, 
                         modelpred_alpha = 1,
                         ticks = np.linspace(-0.7, 0.7, 5))
    
#%%
btst_seed = [None] #+ list(range(10))
flag_btst = [False] #+ [True]*10

for flag_btst_AEPsych, ll in zip(flag_btst, btst_seed):
    str_ext = str_ext_s
    if flag_btst_AEPsych: str_ext += f'_btst_AEPsych[{ll}]'
    
    # bootstrap the data subset
    if flag_btst_AEPsych:
        #bootstrap AEPsych trials
        xref_jnp, x1_jnp, y_jnp, _ = load_4D_expt_data.bootstrap_AEPsych_data(\
            data_AEPsych_subset[1], data_AEPsych_subset[2], data_AEPsych_subset[0], 
            trials_split=[sum(NTRIALS_STRAT[:-1])*nRefs], seed=ll)
    else:
        y_jnp, xref_jnp, x1_jnp = data_AEPsych_subset
    
    #the original data format of xref_jnp is N x 2 (i.e., all the references are stacked)
    #the function group_trials_by_grid returns xref_jnp as nRefs x N_perRef x 2
    _, (y_jnp_org, xref_jnp_org, x1_jnp_org) = group_trials_by_grid(grid, *data_AEPsych_subset)
    data_AEPsych = (y_jnp_org, xref_jnp_org, x1_jnp_org)
    
    #Generate a matrix of random seeds for each initialization
    random_seeds = np.random.randint(0, 2**32, size = (nRefs, nRepeats, 2))
    
    #initialize
    W_INIT_KEY       = np.full((nRefs,2), np.nan)
    OPT_KEY          = np.full((nRefs,2), np.nan)
    bestfit_seed     = np.full((nRefs,2), np.nan)
    W_init           = np.full((nRefs,) + base_shape + (COLOR_DIMENSION+1,), np.nan)        
    W_est            = np.full(W_init.shape, np.nan)
    Sigmas_est_grid  = np.full((nRefs,) + base_shape + (COLOR_DIMENSION,), np.nan)
    objhist          = np.full((nRefs, nSteps), np.nan)
    fitEll           = np.full((nRefs,) + gt_Wishart.fitEll_unscaled.shape[2:], np.nan)
    model_pred_Wishart_indv_ell = []
    
    # Loop over each reference location to fit an individual ellipse.
    for n in trange(nRefs):
        model      = copy.deepcopy(model_existing)
        opt_params = copy.deepcopy(opt_params_existing)
        
        # Initialize a high upper bound for negative log-likelihood (nLL) to track the best fit
        objhist_end = 1e3  # Start with a large number to ensure any valid fit is better
        # Extract data corresponding to the current reference location.
        ref_n = grid_flatten[n]
        data_n = (y_jnp_org[n], xref_jnp_org[n], x1_jnp_org[n])
    
        # Repeat the optimization multiple times to avoid local minima.
        for nn in range(nRepeats):
            W_INIT_KEY_nn = jax.random.PRNGKey(random_seeds[n,nn,0])
            OPT_KEY_nn = jax.random.PRNGKey(random_seeds[n,nn,1])
            
            W_init_nn = model.sample_W_prior(W_INIT_KEY_nn)
            # Perform optimization to maximize the log-posterior probability of W.
            W_est_nn, _, objhist_nn = optim.optimize_posterior(
                W_init_nn, data_n, model, OPT_KEY_nn,
                opt_params,
                oddity_task.simulate_oddity, 
                total_steps=nSteps,
                save_every=1,
                show_progress=False,
                mask = W_mask,
                use_prior = False
            )
            
            # Decrease the learning rate with each repeat to refine the optimization.
            opt_params['learning_rate'] = 10**(-nn*0.5-1)
            
            # Select the best result from the repeats based on the objective history.
            if objhist_nn[-1] < objhist_end: 
                objhist_end = objhist_nn[-1]
                # Store the best optimization result for this reference location.
                W_INIT_KEY[n], OPT_KEY[n] = W_INIT_KEY_nn, OPT_KEY_nn
                W_init[n], W_est[n] = W_init_nn, W_est_nn
                objhist[n], bestfit_seed[n] = objhist_nn, random_seeds[n,nn]
        
        # Compute the covariance matrices for the current reference location.
        Sigmas_est_grid[n] = model.compute_Sigmas(model.compute_U(W_est[n], grid))
        
        # Initialize the Wishart model predictions for the current reference location.
        model_pred_Wishart_n = wishart_model_pred(model, opt_params,
                                                  W_INIT_KEY[n], OPT_KEY[n], 
                                                  W_init[n], W_est[n], 
                                                  Sigmas_est_grid[n],
                                                  color_thres_data, 
                                                  target_pC=0.667,
                                                  ngrid_bruteforce = 1000,
                                                  bds_bruteforce = [0.0005, 0.2])
        model_pred_Wishart_indv_ell.append(model_pred_Wishart_n)
    
        # Batch compute the threshold contour based on the estimated weight matrix.
        model_pred_Wishart_n.convert_Sig_Threshold_oddity_batch(np.reshape(ref_n,(model.num_dims,1,1)))
        # Store the fitted ellipse for the current reference location.
        fitEll[n] = model_pred_Wishart_n.fitEll_unscaled[0,0]
    
    # Reshape the individual ellipse fits to match the original shape.
    fitEll_indv_org = np.reshape(fitEll, base_shape+ gt_Wishart.fitEll_unscaled.shape[-1:])
    
    # append data
    vars_dict_subset_ll = {}
    for var_name in variable_names_append:
        vars_dict_subset_ll[var_name] = eval(var_name)
        
    #append the dictionary to the existing output file
    #give it a name that's contingent on bootstrap counter
    key_name_ll = f'model_pred_indv{str_ext}'
    # Save the new dictionary under the selected key
    vars_dict_subset[key_name_ll] = vars_dict_subset_ll
    
    # 4. Write the updated dictionary back to file
    with open(output_full_path, 'wb') as f:
        pickled.dump(vars_dict_subset, f)
    
    print(f"Saved updated vars_dict_subset to '{output_full_path}'.")
    
    # visualize model predictions by joint fit and by individual fits
    predM_settings = replace(predM_settings, fig_name = f"{file_name[:-4]}{str_ext}.pdf")
    fig, ax = plt.subplots(1, 1, figsize = predM_settings.fig_size, dpi= predM_settings.dpi)
    #a cleaner way to visualize it
    wishart_pred_vis.plot_2D(grid, ax = ax, settings = predM_settings)
    for n in range(NUM_GRID_PTS):
        for m in range(NUM_GRID_PTS):
            cm_nm = color_thres_data.M_2DWToRGB @ np.insert(grid[n,m], 2, 1)
            ax.plot(fitEll_indv_org[n,m,0], fitEll_indv_org[n,m,1],alpha = 0.7, c = cm_nm)
    ax.set_title('Isoluminant plane');









