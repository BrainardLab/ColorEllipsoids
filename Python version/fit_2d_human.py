#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 11:52:44 2024

@author: fangfang

This fits a Wishart Process model to the pilot human data. Trial placement
was guided by AEPsych. 

"""
#%% import modules
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
import dill as pickled
import sys
import numpy as np

sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids")
from core import optim, oddity_task
from core.wishart_process import WishartProcessModel
from core.model_predictions import wishart_model_pred
from analysis.color_thres import color_thresholds
from plotting.wishart_predictions_plotting import WishartPredictionsVisualization
from plotting.AEPsych_predictions_plotting import AEPsych_predictions_visualization

# Add the parent directory of the aepsych package to sys.path
path_str = '/Users/fangfang/Documents/MATLAB/toolboxes/aepsych-main'
sys.path.append(path_str)
from aepsych.server import AEPsychServer

#data path
baseDir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'
path_str  = baseDir+ 'Meta_analysis/Pilot_DataFiles/Pilot_FH/'
            
#specify the file name
subN = 1
plane_2D  = 'GB plane' 
file_name = f'unity_color_discrimination_2d_oddity_task_{plane_2D[:2]}plane_sub{subN}.db'
full_path = f"{path_str}{file_name}"
os.chdir(path_str)

# Define the color plane for the simulations
COLOR_DIMENSION = 2
color_thres_data = color_thresholds(COLOR_DIMENSION, baseDir + 'ELPS_analysis/', plane_2D = plane_2D)

#specify figure name and path
output_figDir_fits = baseDir+'META_analysis/ModelFitting_FigFiles/2dTask/pilot/'
output_fileDir_fits = baseDir+ 'META_analysis/ModelFitting_DataFiles/2dTask/pilot/'

#%%
# -----------------------------------------------------------
# Load data simulated using CIELab and organize data
# -----------------------------------------------------------
# Initialize the AEPsych server to interact with the database.
serv             = AEPsychServer(database_path = file_name)
exp_ids          = [rec.experiment_id for rec in serv.db.get_master_records()]
nSims            = 360 # Number of simulations or trials per reference stimulus.
nRefs = 9#len(exp_ids)
expt_indices = list(range(nRefs))#list(range(len(exp_ids)))
expt_idx_skip = []#22
#expt_indices.remove(expt_idx_skip)

# Initialize arrays to store reference and comparison stimuli, and observer responses.
refStimulus      = np.full((nRefs,nSims,COLOR_DIMENSION), np.nan)
compStimulus     = np.full(refStimulus.shape, np.nan)
#1: the observer successfully identified the odd stimulus
#0: the observer failed to identify the odd stimulus
responses        = np.full((nRefs,nSims), np.nan) 
# Initialize arrays to store the modified lower and upper bounds for each configuration.
lb_all     = np.full((nRefs, COLOR_DIMENSION), np.nan)
ub_all     = np.full(lb_all.shape, np.nan)

# Iterate over each reference stimulus.
for idx, s in enumerate(expt_indices):
    # Replay the experiment to load the data without redoing computations.
    serv.replay(exp_ids[s], skip_computations = True)

    # Retrieve strategy and calculate stimuli.
    strat             = serv._strats[idx]
    # retrieve lower and upper bounds in rgb unit (0-255)
    lb_all[idx]       = np.array(strat.lb)
    ub_all[idx]       = np.array(strat.ub)
    # the ref stimulus we intend to display
    xref_raw_intend   = (strat.ub - strat.lb)/2 + strat.lb
    # unity rounds up to the closest integer
    xref_raw_actual   = color_thres_data.rgb_to_interger_rgb(xref_raw_intend)
    # convert the RGB value unity actually displays to W unit
    # The original range of RGB is within [0,1], but Chebyshev polynomial ranges from 
    # [-1,1], so we have to rescale the RGB values to fit in the new box 
    xref_raw_s        = color_thres_data.rgb_to_W_unit(xref_raw_actual)
    refStimulus[idx]  = np.tile(xref_raw_s, (nSims,1))
    # Do the same for the comparison stimulus
    x1_raw_intend     = strat.x[:nSims]
    x1_raw_actual     = color_thres_data.rgb_to_interger_rgb(x1_raw_intend)
    compStimulus[idx] = color_thres_data.rgb_to_W_unit(x1_raw_actual)
    # Retrieve the observer responses for the stimuli.
    responses[idx]    = strat.y[:nSims]

# Reshape the arrays for fitting.
xref_jnp = jnp.reshape(refStimulus, (nRefs*nSims,2))
x0_jnp   = jnp.reshape(refStimulus, (nRefs*nSims,2))
x1_jnp   = jnp.reshape(compStimulus, (nRefs*nSims,2))
y_jnp    = jnp.reshape(responses, (nRefs*nSims))

# Pack the processed data into a tuple for further use.
#data = (y_jnp, xref_jnp, x0_jnp, x1_jnp)
data = (y_jnp, xref_jnp, x1_jnp)

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
W_INIT_KEY   = jax.random.PRNGKey(225)  # Key to initialize `W_est`. 
DATA_KEY     = jax.random.PRNGKey(333)  # Key to generate datatset.
OPT_KEY      = jax.random.PRNGKey(444)  # Key passed to optimizer.

# -----------------------------
# Fit W by maximizing posterior
# -----------------------------
# Fit model, initialized at random W
W_init = 1e-1*model.sample_W_prior(W_INIT_KEY) #1e-1*

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

#%%
# -----------------------------
# Rocover covariance matrices
# -----------------------------
# Generate a multidimensional grid based on the number of color dimensions
grid = jnp.stack(jnp.meshgrid(*[jnp.linspace(jnp.min(xref_jnp),\
                                              jnp.max(xref_jnp),\
                    NUM_GRID_PTS) for _ in range(model.num_dims)]), axis=-1)
# Compute the covariance matrices ('Sigmas') at each point in the grid using 
# the model's compute_U function. 
Sigmas_est_grid = model.compute_Sigmas(model.compute_U(W_est, grid))

# Initialize the Wishart model prediction using various parameters.
model_pred_Wishart = wishart_model_pred(model, opt_params, NUM_GRID_PTS, 
                                        MC_SAMPLES, BANDWIDTH, W_INIT_KEY,
                                        DATA_KEY, OPT_KEY, W_init,
                                        W_est, Sigmas_est_grid,
                                        color_thres_data, 
                                        target_pC=0.67,
                                        ngrid_bruteforce = 1000,
                                        bds_bruteforce = [0.0005, 0.2])

# Transpose the grid to match the expected input format of the model's prediction functions.
# The transposition is dependent on the color dimension to ensure the correct orientation of the data.
grid_trans = np.transpose(grid,(2,0,1))
# batch compute 78% threshold contour based on estimated weight matrix
model_pred_Wishart.convert_Sig_Threshold_oddity_batch(grid_trans)
        
#%%
# -----------------------------
# Visualize model predictions
# -----------------------------
#specify figure name and path
fig_name_part1 = f'Fitted_isothreshold_{plane_2D}_sim{nSims}perRef_{nRefs}refs_'+\
                f'AEPsychSampling_bandwidth{BANDWIDTH}'
    
class expt_data:
    def __init__(self, xref_all, x1_all, y_all, pseudo_order):
        self.xref_all = xref_all
        self.x1_all = x1_all
        self.y_all = y_all
        self.pseudo_order = pseudo_order
expt_trial = expt_data(xref_jnp, x1_jnp, y_jnp, np.reshape(np.array([20,0,14,22,10,24,12,4,2]),(nRefs,1)))
#expt_trial = expt_data(xref_jnp, x1_jnp, y_jnp, np.reshape(np.array([20,0,14,22,10,24,12,4,2,6,18,16,8,7,13,11,17,1,19,5,23,3,21,15,9]),(nRefs,1)))
wishart_pred_vis = WishartPredictionsVisualization(expt_trial,
                                                   model, 
                                                   model_pred_Wishart, 
                                                   color_thres_data,
                                                   fig_dir = output_figDir_fits, 
                                                   save_fig = True)
wishart_pred_vis.pltP['dpi'] = 1024
grid_samples_temp = color_thres_data.N_unit_to_W_unit(np.linspace(0.2, 0.8,5))
grid_samples = jnp.stack(jnp.meshgrid(*[grid_samples_temp for _ in range(2)]), axis=-1)

#visualize samples and model-estimated cov matrices
wishart_pred_vis.plot_2D(
    grid, 
    grid_samples,
    visualize_samples= True,
    visualize_gt = False,
    samples_alpha = 0.5,
    samples_s = 2,
    sigma_lw = 0.5,
    sigma_alpha = 1,
    modelpred_alpha = 1,
    modelpred_lw = 0.5,
    modelpred_lc = 'k',
    modelpred_ls = '--',
    samples_label = 'Experimental data',
    fig_name = fig_name_part1)

wishart_pred_vis.plot_2D(
    grid, 
    grid_samples,
    visualize_samples= True,
    visualize_gt = False,
    samples_alpha = 0.5,
    samples_s = 1,
    sigma_lw = 0.5,
    sigma_alpha = 1,
    modelpred_alpha = 1,
    modelpred_lw = 0.5,
    modelpred_lc = 'k',
    modelpred_ls = '--',
    samples_label = 'Experimental data',
    samples_colorcoded_resp = True,
    fig_name = fig_name_part1 +'_wYesNo')

#%% Fit individual ellipses
# Unroll the grid into a 2D array where each column is a flattened reference location.
idx_slc = [0,2,4]
grid_unroll = np.reshape(grid_trans[:,idx_slc][:,:,idx_slc], (model.num_dims, nRefs))

# Initialize a mask for the weight matrix W, where only the first element is unconstrained.
W_mask = np.zeros(W_est.shape, dtype=bool)
W_mask[0,0] = True

# Set the optimization parameters for individual ellipse fitting.
opt_params_indv_ell = opt_params

# Define the number of repeats and steps for the optimization and initialize arrays to 
# store results for each reference location
nRepeats = 10
nSteps   = 20
W_est_indv_ell           = np.full((nRefs,) + W_est.shape, np.nan)
Sigmas_est_grid_indv_ell = np.full((nRefs,) + Sigmas_est_grid.shape, np.nan)
objhist_indv_ell         = np.full((nRefs, nSteps), np.nan)
fitEll_indv              = np.full((nRefs,) + model_pred_Wishart.fitEll_unscaled.shape[2:], np.nan)
model_pred_Wishart_indv_ell = []
# Loop over each reference location to fit an individual ellipse.
for n in range(nRefs):
    print(n)
    # Extract data corresponding to the current reference location.
    ref_n = np.reshape(grid_unroll[:,n], (1, model.num_dims))
    idx_n = np.where(np.all(np.abs(xref_jnp - ref_n) < 5e-2, axis=1))[0]
    if len(idx_n) != nSims: print('Some error occurred!')
    data_n = (y_jnp[idx_n], xref_jnp[idx_n], x1_jnp[idx_n])

    # Repeat the optimization multiple times to avoid local minima.
    for nn in range(nRepeats):
        W_INIT_KEY_nn   = jax.random.PRNGKey(222+nn)
        W_init_nn = model.sample_W_prior(W_INIT_KEY_nn)
        # Perform optimization to maximize the log-posterior probability of W.
        W_est_indv_ell_nn, iters_indv_ell, objhist_indv_ell_nn = optim.optimize_posterior(
            W_init_nn, data_n, model, OPT_KEY,
            opt_params_indv_ell,
            oddity_task.simulate_oddity, 
            total_steps=nSteps,
            save_every=1,
            show_progress=False,
            mask = W_mask,
            use_prior = False
        )
        
        # Decrease the learning rate with each repeat to refine the optimization.
        opt_params_indv_ell['learning_rate'] = 10**(-nn*0.5-1)
        
        # Select the best result from the repeats based on the objective history.
        if nn == 0: 
            W_est_indv_ell_n = W_est_indv_ell_nn
            objhist_indv_ell_n = objhist_indv_ell_nn
        else:
            if objhist_indv_ell_nn[-1] < objhist_indv_ell_n[-1]:
                W_est_indv_ell_n = W_est_indv_ell_nn
                objhist_indv_ell_n = objhist_indv_ell_nn
        
    # Store the best optimization result for this reference location.
    W_est_indv_ell[n]= W_est_indv_ell_n
    objhist_indv_ell[n] = objhist_indv_ell_n
    
    # Check if optimization failed by comparing the start and end of the objective history.
    if objhist_indv_ell[n,-1] > objhist_indv_ell[n,0]:
        print('Optimization failed!')
    
    # Compute the covariance matrices for the current reference location.
    Sigmas_est_grid_indv_ell[n] = model.compute_Sigmas(model.compute_U(W_est_indv_ell[n], grid))
    
    # Initialize the Wishart model predictions for the current reference location.
    model_pred_Wishart_n = wishart_model_pred(model, opt_params_indv_ell, 1, 
                                            MC_SAMPLES, BANDWIDTH, W_INIT_KEY,
                                            DATA_KEY, OPT_KEY, W_init,
                                            W_est_indv_ell[n], Sigmas_est_grid_indv_ell[n],
                                            color_thres_data, 
                                            target_pC=0.67,
                                            ngrid_bruteforce = 1000,
                                            bds_bruteforce = [0.0005, 0.2])
    model_pred_Wishart_indv_ell.append(model_pred_Wishart_n)

    # Batch compute the threshold contour based on the estimated weight matrix.
    model_pred_Wishart_n.convert_Sig_Threshold_oddity_batch(np.reshape(ref_n,(model.num_dims,1,1)))
    # Store the fitted ellipse for the current reference location.
    fitEll_indv[n] = model_pred_Wishart_n.fitEll_unscaled[0,0]

# Reshape the individual ellipse fits to match the original shape.
fitEll_indv_org = np.reshape(fitEll_indv, model_pred_Wishart.fitEll_unscaled.shape)
    
#%% visualize model predictions by joint fit and by individual fits
wishart_pred_vis.plot_2D(
    grid, 
    grid_samples, 
    gt_ellipses=fitEll_indv_org,
    visualize_samples= True,
    visualize_gt = True,
    visualize_model_estimatedCov = False,
    modelpred_alpha = 1,
    modelpred_lw = 0.5,
    modelpred_lc = 'k',
    modelpred_ls = '--',
    gt_lw = 1,
    gt_lc = np.array([255,140,0])/255,
    gt_ls = '-',
    gt_alpha = 0.5,
    gt_label = 'Indivisual-ellipse fits',
    samples_s = 0.5,
    samples_label = 'Experimental data',
    samples_colorcoded_resp = True,
    fig_name = fig_name_part1+'_wIndvFits')   

#%% ----------------------------------------------------
# Visualize AEPsych-estimated thresholds
# ------------------------------------------------------
# Initialize the visualization class for AEPsych predictions
AEPsych_pred_vis = AEPsych_predictions_visualization(expt_trial,
                                                     serv, 
                                                     color_thres_data,
                                                     fig_dir = output_figDir_fits, #make sure this folder exists
                                                     save_fig = False)
refStimulus_unique = refStimulus[:,0,:]
fitEll_indv_N_unit = np.transpose(fitEll_indv_org,(1,0,2,3)) 
idx_order = np.argsort(np.reshape(expt_trial.pseudo_order,(nRefs)))
# This function dynamically chooses between 2D and 3D plotting based on the dimensionality of the data.
AEPsych_pred_vis.plot_predictions(refStimulus_unique[idx_order],         # Reference points used in the visualizations
                                fitEll_indv_N_unit, # Ground truth data for ellipses or ellipsoids
                                np.array(list(range(5))),   # Indices indicating which reference points from ground truth to use
                                lb_all[idx_order],        # Lower bounds for each dimension in the plot
                                ub_all[idx_order],        # Upper bounds for each dimension in the plot
                                skip_expt = [expt_idx_skip],
                                grid_size = 200,
                                fig_name = 'Fitted_byAEPsych_temp')             
        
#%% save data
output_file = fig_name_part1 + f'_pilot_sub{subN}.pkl'
full_path = f"{output_fileDir_fits}{output_file}"

variable_names = ['plane_2D', 'nSims', 'expt_idx_skip','expt_indices',
                  'color_thres_data','data','model_pred_Wishart',
                  'grid', 'grid_trans','grid_samples', 'grid_unroll',
                  'W_mask', 'W_est_indv_ell', 'Sigmas_est_grid_indv_ell',
                  'objhist_indv_ell', 'fitEll_indv', 'fitEll_indv_org',
                  'model_pred_Wishart_indv_ell']
vars_dict = {}
for i in variable_names: vars_dict[i] = eval(i)

# Write the list of dictionaries to a file using pickle
with open(full_path, 'wb') as f:
    pickled.dump(vars_dict, f)

