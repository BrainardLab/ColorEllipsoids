#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 21:08:44 2024

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
from plotting.wishart_predictions_plotting import WishartPredictionsVisualization
sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/Python version/')

base_dir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'
output_figDir = base_dir + 'ELPS_analysis/ModelFitting_FigFiles/Python_version/2D_oddity_task/MOCS/'
output_fileDir = base_dir + 'ELPS_analysis/ModelFitting_DataFiles/2D_oddity_task/MOCS/'
    
#%% -----------------------------------------------------------
# Load data simulated using CIELab and organize data
# -----------------------------------------------------------
# Set the seed for numpy
rng_seed = 0
np.random.seed(rng_seed)  
#set stimulus info
color_dimension = 2
plane_2D        = 'RG plane' 
#chromatic direction 
nTheta          = 8
# sample theta from 0 to 2pi uniformaly; get rid of the last one, which is the same as the first one
theta           = np.linspace(0, 2*np.pi, nTheta + 1)[:-1]
# convert angle to x, y coordinates
cDir_xy         = np.vstack((np.cos(theta), np.sin(theta)))
# number of vector lengths
nLevels         = 8
# evenly sample vector length from a range
vecLength       = np.linspace(0.01, 0.3, nLevels)
#number of trials per chromatic direction and vector length
nSims_l         = 10
# Number of simulations or trials per reference stimulus
nSims           = nTheta * nLevels * nSims_l 

#file 1
path_str   = base_dir + 'ELPS_analysis/Simulation_DataFiles/'
# Create an instance of the class
color_thres_data = color_thresholds(color_dimension, 
                                    base_dir + 'ELPS_analysis/',
                                    plane_2D = plane_2D)
# Retrieve specific data from Wishart_data
color_thres_data.load_model_fits()
gt_Wishart    = color_thres_data.get_data('model_pred_Wishart', dataset = 'Wishart_data')
gt_model      = gt_Wishart.model
gt_W          = gt_Wishart.W_est
gt_opt_params = gt_Wishart.opt_params

# retrieve 2D grid
gt_grid = color_thres_data.get_data('grid', dataset = 'Wishart_data') #in W unit
# Total number of simulations to perform
nSims_total   = int(nSims * gt_Wishart.num_grid_pts**color_dimension)

#%% simulate data
# Repeat the grid of reference stimuli. Each reference stimulus is repeated nTheta x nLevels x nSims_l times.
grid_rep = np.transpose(np.tile(gt_grid, (nSims,1,1,1)), (3,1,2,0))
# Reshape the grid_rep array into a new shape so that xMOCS has the shape (2 x nSims_total).
xMOCS = jnp.array(np.transpose(np.reshape(grid_rep, (color_dimension, nSims_total)),(1,0)))

# Construct the comparison stimulus x1 carefully:
# First, repeat the vector length for each trial (nSims_l times) across the number of simulations.
vecLength_rep        = np.tile(vecLength[:,np.newaxis], (1, nSims_l))
vecLength_reshape    = np.reshape(vecLength_rep, (nSims_l*nLevels))
# Now repeat the reshaped vector length by the number of chromatic directions (nTheta times).
vecLength_rep2       = np.tile(vecLength_reshape, (nTheta)) 
# Repeat the chromatic directions for nSims_l * nLevels times, matching the length of vecLength_rep2.
cDir_xy_rep          = np.tile(cDir_xy[:,:,np.newaxis], (1, 1, nSims_l*nLevels))
# Reshape the repeated chromatic directions to have the same number of columns as vecLength_rep2.
cDir_xy_reshape      = np.reshape(cDir_xy_rep, (color_dimension, nSims))
# Multiply the reshaped chromatic directions by the vector length to get the final comparison stimulus directions.
cDir_xy_1ref         = cDir_xy_reshape * vecLength_rep2[np.newaxis,:]

# Repeat the chromatic directions by the number of reference stimuli to match the grid of reference stimuli.
cDir_xy_1ref_rep     = np.tile(cDir_xy_1ref[:,np.newaxis,:], (1,gt_Wishart.num_grid_pts**color_dimension,1))
# Reshape the repeated chromatic directions so the final size is (2 x nSims_total).
cDir_xy_1ref_reshape = np.reshape(cDir_xy_1ref_rep, (color_dimension, nSims_total))
# Add the chromatic directions to the grid of reference stimuli (xMOCS) to get the final comparison stimulus (x1MOCS).
x1MOCS               = jnp.array(xMOCS + np.transpose(cDir_xy_1ref_reshape, (1,0)))

# compute weighted sum of basis function at rgb_ref 
Uref   = gt_model.compute_U(gt_W, xMOCS)
# compute weighted sum of basis function at rgb_comp
U1     = gt_model.compute_U(gt_W, x1MOCS)
# Predict the probability of choosing the comparison stimulus over the reference
pX1    = oddity_task.oddity_prediction((xMOCS, x1MOCS, Uref, U1),
                  jax.random.split(gt_Wishart.opt_key, num = nSims_total),
                  gt_opt_params['mc_samples'], 
                  gt_opt_params['bandwidth'],
                  gt_model.diag_term, 
                  oddity_task.simulate_oddity)
# Simulate a response based on the predicted probability
randNum   = np.random.rand(*pX1.shape) 
resp      = jnp.array((randNum < pX1).astype(int))

# Package the processed data into a tuple for further use
data_MOCS = (resp, xMOCS, x1MOCS)
print(np.mean(resp))
#plt.scatter(x1MOCS[:,0] - xMOCS[:,0], x1MOCS[:,1] - xMOCS[:,1])

#%% -------------------------------
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
    W_init, data_MOCS, model_test, OPT_KEY,
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
grid_1d = jnp.linspace(-0.6, 0.6, NUM_GRID_PTS)
grid = jnp.stack(jnp.meshgrid(*[grid_1d for _ in range(model_test.num_dims)]), axis=-1)
Sigmas_recover_wRandx = model_test.compute_Sigmas(model_test.compute_U(W_recover_wRandx, grid))

# -----------------------------
# Compute model predictions
# -----------------------------
target_pC = 2/3
model_pred_Wishart_MOCS = wishart_model_pred(model_test, opt_params, 
                                        W_INIT_KEY, DATA_KEY, OPT_KEY, W_init, 
                                        W_recover_wRandx, Sigmas_recover_wRandx, 
                                        color_thres_data,
                                        target_pC= target_pC,
                                        scaler_x1 = 5,
                                        ngrid_bruteforce = 500,
                                        bds_bruteforce = [0.01, 0.25])

grid_trans = np.transpose(grid,(2,0,1))
model_pred_Wishart_MOCS.convert_Sig_Threshold_oddity_batch(grid_trans)

#%% -----------------------------
# Visualize model predictions
# -----------------------------
class sim_data:
    def __init__(self, xref_all, x1_all):
        self.xref_all = xref_all
        self.x1_all = x1_all
sim_trial_by_Wishart = sim_data(xMOCS, x1MOCS)

wishart_pred_vis = WishartPredictionsVisualization(sim_trial_by_Wishart,
                                                   model_test, 
                                                   model_pred_Wishart_MOCS, 
                                                   color_thres_data,
                                                   fig_dir = output_figDir, 
                                                   save_fig = True)

fig_name_fits = f"Fitted_isothreshold_{plane_2D}_sim{nSims_total}total_{nTheta}cDir_"+\
    f"{nLevels}nLevels_{nSims_l}simsPerCond_MOCS_seed{rng_seed}_bandwidth{BANDWIDTH}"
wishart_pred_vis.plot_2D(
    grid, 
    grid,
    gt_Wishart.fitEll_unscaled, 
    visualize_samples= True,
    visualize_gt = True,
    visualize_model_estimatedCov = False,
    samples_alpha = 1,
    samples_s = 1,
    plane_2D = plane_2D,
    modelpred_ls = '-',
    modelpred_lw = 2,
    modelpred_lc = None,
    modelpred_alpha = 0.8,
    samples_label = 'Simulated data based on estimated W',
    gt_lw= 0.5,
    gt_lc =[0.1,0.1,0.1],
    fontsize = 8.5,
    fig_name = fig_name_fits +'.pdf') 
    
#%% save data
output_file = fig_name_fits+'_oddity.pkl'
full_path = f"{output_fileDir}{output_file}"

variable_names = ['rng_seed','plane_2D', 'nTheta','cDir_xy','nLevels',
                  'vecLength', 'nSims_l', 'nSims', 'nSims_total',
                  'color_thres_data', 'gt_Wishart', 'data_MOCS', 'pX1', 
                  'model_test', 'grid_1d', 'grid','grid_trans',
                  'model_pred_Wishart_MOCS','iters', 'objhist']

vars_dict = {}
for i in variable_names: vars_dict[i] = eval(i)

# Write the list of dictionaries to a file using pickle
with open(full_path, 'wb') as f:
    pickled.dump(vars_dict, f)
