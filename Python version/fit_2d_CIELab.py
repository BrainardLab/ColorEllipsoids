#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 11:52:44 2024

@author: fangfang

This fits a Wishart Process model to the simulated data using the CIELab color space. 

"""

#import modules
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import dill as pickle
import sys
import os
import numpy as np
from dataclasses import replace
sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids")
from core import optim, oddity_task
from core.wishart_process import WishartProcessModel
from analysis.color_thres import color_thresholds
from core.model_predictions import wishart_model_pred
from plotting.wishart_predictions_plotting import WishartPredictionsVisualization, Plot2DPredSettings
from plotting.wishart_plotting import PlotSettingsBase 
sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/Python version')
from data_reorg import organize_data
from analysis.cross_validation import expt_data

#%% three variables we need to define for loading the data
#plane_2D      = 'RG plane'
#plane_2D_dict = {'GB plane': 0, 'RB plane': 1, 'RG plane': 2}
#plane_2D_idx  = plane_2D_dict[plane_2D]
plane_2D      ='Isoluminant plane'
plane_2D_idx  = 2
sim_jitter    = 0.3
nSims         = 6000 #number of simulations: 
rnd_seed      = 0

baseDir = '/Volumes/T9/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'
output_figDir_fits = os.path.join(baseDir, 'ELPS_analysis','ModelFitting_FigFiles',
                                  'Python_version','2D_oddity_task', plane_2D)
output_fileDir = os.path.join(baseDir, 'ELPS_analysis','ModelFitting_DataFiles',
                              '2D_oddity_task', plane_2D)
pltSettings_base = PlotSettingsBase(fig_dir=output_figDir_fits, fontsize = 8)

#%%
# -----------------------------------------------------------
# Load data simulated using CIELab and organize data
# -----------------------------------------------------------
#file 1
path_str = os.path.join(baseDir, 'ELPS_analysis','Simulation_DataFiles')
if plane_2D == 'Isoluminant plane':
    colordiff_alg = 'CIE1994'
    flag_convert_to_W = False
else:
    colordiff_alg = ''
    flag_convert_to_W = True 
file_sim = f"Sims_isothreshold_{plane_2D}_{colordiff_alg}_sim{nSims}total_"+\
            f"samplingNearContour_jitter{sim_jitter}_seed{rnd_seed}.pkl"
full_path = f"{path_str}/{plane_2D}/{file_sim}"   
with open(full_path, 'rb') as f:  
    data_load = pickle.load(f)
sim = data_load['sim_trial'].sim

# Create an instance of the class
color_thres_data = color_thresholds(2, baseDir + 'ELPS_analysis/',
                                    plane_2D = plane_2D)
# Load Wishart model fits
color_thres_data.load_CIE_data(CIE_version = colordiff_alg)
color_thres_data.load_transformation_matrix()

# #organize simulated data to jnp
# data_temp, x1_raw, xref_raw = organize_data(2, sim, scaler_x1,
#                                        visualize_samples = True,
#                                        flag_convert_to_W = flag_convert_to_W,
#                                        plane_2D = plane_2D,
#                                        M_2DWToRGB = color_thres_data.M_2DWToRGB)
# y_jnp, xref_jnp, x0_jnp, x1_jnp = data_temp

y_jnp, xref_jnp, x1_jnp = jnp.array(sim['resp_binary']), jnp.array(sim['ref_points'][:2,:].T), jnp.array(sim['rgb_comp'][:2,:].T)
data = (y_jnp, xref_jnp, x1_jnp)

#%% 
# -------------------------------
# Constants describing simulation
# -------------------------------
model = WishartProcessModel(
    6,     # Degree of the polynomial basis functions default = 5
    2,     # Number of stimulus dimensions
    1,     # Number of extra inner dimensions in `U`.
    3e-4,  # Scale parameter for prior on `W`.
    0.4,   # Geometric decay rate on `W`. default= 0.4
    0,     # Diagonal term setting minimum variance for the ellipsoids.
)

NUM_GRID_PTS = 8      # Number of grid points over stimulus space.
MC_SAMPLES   = 2000        # Number of simulated trials to compute likelihood.
BANDWIDTH    = 5e-3        # Bandwidth for logistic density function.
opt_params = {
    "learning_rate": 1e-4, 
    "momentum": 0.2,
    "mc_samples": MC_SAMPLES,
    "bandwidth": BANDWIDTH,
}

# -----------------------------
# Fit W by maximizing posterior
# -----------------------------
# Search for the best-fit parameters from three random initializations to avoid
# getting stuck at local minimum
nRepeats = 3
#Generate a matrix of random seeds for each initialization
random_seeds = np.random.randint(0, 2**32, size = (nRepeats, 2))
# Initialize a high upper bound for negative log-likelihood (nLL) to track the best fit
objhist_end = 1e3  # Start with a large number to ensure any valid fit is better
# Loop through each random initialization
for i in range(nRepeats):
    # Generate random keys for initializing parameters, data, and optimizer
    W_INIT_KEY_i   = jax.random.PRNGKey(random_seeds[i,0])  # Key to initialize `W_est`. 
    OPT_KEY_i      = jax.random.PRNGKey(random_seeds[i,1])  # Key passed to optimizer.
    
    # Fit model, initialized at a random W sampled from the prior distribution
    W_init_i = model.sample_W_prior(W_INIT_KEY_i) 
    
    W_est_i, iters_i, objhist_i = optim.optimize_posterior(
        W_init_i, data, model, OPT_KEY_i,
        opt_params,
        oddity_task.simulate_oddity, 
        total_steps=1000,
        save_every=1,
        show_progress=True
    )
    
    # Update the best-fit model if the current fit improves the objective (lower nLL)
    if objhist_i[-1] < objhist_end:
        objhist_end = objhist_i[-1]
        W_init, W_est, iters, objhist = W_init_i, W_est_i, iters_i, objhist_i
        W_INIT_KEY, OPT_KEY = W_INIT_KEY_i, OPT_KEY_i
        bestfit_seed = random_seeds[i]
        
fig, ax = plt.subplots(1, 1)
ax.plot(iters, objhist)
fig.tight_layout(); plt.show()

#%%
# -----------------------------
# Compute model predictions
# -----------------------------
# Specify grid over stimulus space
grid = jnp.stack(jnp.meshgrid(*[jnp.linspace(-0.6, 0.6,
                    NUM_GRID_PTS) for _ in range(model.num_dims)]), axis=-1)
# Compute the covariance matrices ('Sigmas') at each point in the grid using 
# the model's compute_U function. 
Sigmas_est_grid = model.compute_Sigmas(model.compute_U(W_est, grid))

target_pC = 0.667
model_pred_Wishart = wishart_model_pred(model, opt_params, W_INIT_KEY,
                                        OPT_KEY, W_init, 
                                        W_est, Sigmas_est_grid, 
                                        color_thres_data,
                                        target_pC=target_pC,
                                        colordiff_alg = colordiff_alg,
                                        ngrid_bruteforce = 1000,
                                        bds_bruteforce = [0.0005, 0.25])

# Transpose the grid to match the expected input format of the model's prediction functions.
# The transposition is dependent on the color dimension to ensure the correct orientation of the data.
grid_trans = np.transpose(grid,(2,0,1))
# batch compute 66.7% threshold contour based on estimated weight matrix
model_pred_Wishart.convert_Sig_Threshold_oddity_batch(grid_trans)
        
#%%
# -----------------------------
# Visualize model predictions
# -----------------------------
#ground truth ellipses
results = color_thres_data.get_data('results2D', dataset = 'CIE_data')  
if not flag_convert_to_W:
    gt_covMat_CIE = results['fitEllipse_scaled'][plane_2D_idx]
else:
    gt_covMat_CIE = color_thresholds.N_unit_to_W_unit(results['fitEllipse_scaled'][plane_2D_idx])
#specify figure name and path
fig_name_part1 = f'Fitted{file_sim[4:-4]}_bandwidth{BANDWIDTH}_decay{model.decay_rate}'
pred2D_settings = replace(Plot2DPredSettings(), **pltSettings_base.__dict__)
pred2D_settings1 = replace(pred2D_settings, 
                           visualize_samples= True,
                           visualize_gt = False,
                           visualize_model_estimatedCov = True,
                           flag_rescale_axes_label = False,
                           samples_alpha = 0.75,
                           samples_s = 1,
                           plane_2D = plane_2D,
                           modelpred_ls = '-',
                           modelpred_lc = [0,0,0],
                           modelpred_lw = 0.5,
                           modelpred_alpha = 1,
                           fontsize = 9,
                           fig_name = f"{fig_name_part1}_withSamples")
pred2D_settings2 = replace(pred2D_settings, 
                           visualize_samples= False,
                           visualize_model_estimatedCov= False,
                           visualize_gt = True,
                           gt_lw = 1, 
                           gt_alpha = 0.5, 
                           modelpred_lc = 'g',
                           modelpred_lw = 1.5, 
                           modelpred_alpha = 0.5,
                           fontsize = 9,
                           fig_name = f"{fig_name_part1}")
sim_trial_by_CIE = expt_data(xref_jnp, x1_jnp, y_jnp, None)
wishart_pred_vis = WishartPredictionsVisualization(sim_trial_by_CIE,
                                                   model, 
                                                   model_pred_Wishart, 
                                                   color_thres_data,
                                                   settings = pltSettings_base, 
                                                   save_fig = False)
#version 1 with samples
wishart_pred_vis.plot_2D(grid, grid, settings = pred2D_settings1, gt_ellipses=gt_covMat_CIE) 
#version 2 without samples
wishart_pred_vis.plot_2D(grid, grid, settings = pred2D_settings2, gt_ellipses=gt_covMat_CIE) 

        
#%% save data
output_file = f"{fig_name_part1}_oddity.pkl"
full_path = f"{output_fileDir}/{output_file}"

variable_names = ['plane_2D','sim_jitter','nSims','rnd_seed','colordiff_alg',
                  'scaler_x1', 'file_sim', 'sim', 'color_thres_data', 'data',
                  'x1_raw', 'xref_raw','model','opt_params','nRepeats',
                  'random_seeds', 'objhist_end', 'bestfit_seed', 
                  'model_pred_Wishart', 'sim_trial_by_CIE', 
                  'grid','grid_trans','gt_covMat_CIE']
vars_dict = {}
for var_name in variable_names:
    try:
        # Check if the variable exists in the global scope
        vars_dict[var_name] = eval(var_name)
    except NameError:
        # If the variable does not exist, assign None and print a message
        vars_dict[var_name] = None
        print(f"Variable '{var_name}' does not exist. Assigned as None.")

# Write the list of dictionaries to a file using pickle
with open(full_path, 'wb') as f:
    pickle.dump(vars_dict, f)

