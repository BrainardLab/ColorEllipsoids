#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 11:52:44 2024

@author: fangfang
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from core import chebyshev, viz, utils, oddity_task, optim
from core.wishart_process import WishartProcessModel

# -------------------------------
# Constants describing simulation
# -------------------------------
model = WishartProcessModel(
    5,     # Degree of the polynomial basis functions
    2,     # Number of stimulus dimensions
    1,     # Number of extra inner dimensions in `U`.
    4e-3,  # Scale parameter for prior on `W`.
    0.2,   # Geometric decay rate on `W`.
    1e-3,  # Diagonal term setting minimum variance for the ellipsoids.
)

NUM_GRID_PTS = 5      # Number of grid points over stimulus space.
MC_SAMPLES = 1000        # Number of simulated trials to compute likelihood.
NUM_TRIALS = 4000      # Number of trials in simulated dataset.
MIN_LR = -7
MAX_LR = -3

# Random number generator seeds
W_TRUE_KEY = jax.random.PRNGKey(111)  # Key to generate `W`.
W_INIT_KEY = jax.random.PRNGKey(223)  # Key to initialize `W_est`. 
DATA_KEY = jax.random.PRNGKey(333)    # Key to generate datatset.
OPT_KEY = jax.random.PRNGKey(444)     # Key passed to optimizer.

# ------------------------------------------
# Load data simulated using CIELab
# ------------------------------------------
import os
import pickle

plane_2D = 'RG plane'
file_name = 'Sims_isothreshold_'+plane_2D+'_sim240perCond_samplingNearContour_jitter0.3.pkl'
path_str  = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
            'ELPS_analysis/Simulation_DataFiles/'
full_path = f"{path_str}{file_name}"
os.chdir(path_str)

#load data    
with open(full_path, 'rb') as f:
    # Load the object from the file
    data_load = pickle.load(f)
sim = data_load[0]
#comparisom stimulus; size: (5 x 5 x 3 x 240)
#the comparison stimulus was sampled around 5 x 5 different ref stimuli
x1_temp = sim['rgb_comp'][:,:,sim['varying_RGBplane'],:]
#the original data ranges from 0 to 1
#here we scale it so that it fits the [-1, 1] cube
x1_temp = x1_temp * 2 -1
#number of simulations: 240 trials for each ref stimulus
nSims = (x1_temp.shape)[-1] 
#reshape the array so the final size is (2000, 3)
x1_tmep_transpose = np.transpose(x1_temp, (0,1,3,2))
x1_repeated_col1 = x1_tmep_transpose[:,:,:,0].ravel()
x1_repeated_col2 = x1_tmep_transpose[:,:,:,1].ravel()
x1_temp_reshaped = np.stack((x1_repeated_col1,x1_repeated_col2), axis=1)
#convert it to jnp
x1_jnp = jnp.array(x1_temp_reshaped, dtype=jnp.float64)

#reference stimulus
xref_temp = sim['ref_points'][sim['varying_RGBplane'],:,:]
xref_temp_transpose = np.transpose(xref_temp, (1,2,0))
xref_temp_scaled = xref_temp_transpose * 2 -1
xref_temp_expanded = np.expand_dims(xref_temp_scaled, axis=-1)
xref_repeated = np.tile(xref_temp_expanded, (1, 1, 1, nSims))
xref_repeated_col1 = xref_repeated[:,:,0,:].ravel()
xref_repeated_col2 = xref_repeated[:,:,1,:].ravel()
xref_temp_reshaped = np.stack((xref_repeated_col1,xref_repeated_col2), axis=1)
xref_jnp = jnp.array(xref_temp_reshaped, dtype = jnp.float64)

#copy the reference stimulus
x0_jnp = jnp.copy(xref_jnp)

#binary responses 
y_temp = jnp.array(sim['resp_binary'], dtype = jnp.float64)
y_jnp = y_temp.ravel()

# Specify grid over stimulus space
xgrid = jnp.stack(
    jnp.meshgrid(
        *[jnp.linspace(-0.6, 0.6, NUM_GRID_PTS) for _ in range(model.num_dims)]
    ), axis=-1
)

# Package data
data = (y_jnp, xref_jnp, x0_jnp, x1_jnp)
print("Proportion of correct trials:", jnp.mean(y_jnp))
    

#%%
file_name2 = 'Isothreshold_contour_CIELABderived.pkl'
path_str = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
        'ELPS_analysis/Simulation_DataFiles/'
full_path = f"{path_str}{file_name2}"
os.chdir(path_str)

#Here is what we do if we want to load the data
with open(full_path, 'rb') as f:
    # Load the object from the file
    data_load = pickle.load(f)
param, stim, results, plt_specifics = data_load[0], data_load[1], data_load[2], data_load[3]
    
#import functions from the other script
func_path = '/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/Python version/'
os.chdir(func_path)
from Simulate_probCorrectResp import plot_2D_sampledComp

#%% visualize the simulated data
plot_2D_sampledComp(stim['grid_ref'], stim['grid_ref'], sim['rgb_comp'],\
                    sim['varying_RGBplane'], sim['method_sampling'], \
                    responses = sim['resp_binary'], \
                    groundTruth = results['fitEllipse_unscaled'][sim['slc_RGBplane']],\
                    x_label = plt_specifics['subTitles'][sim['slc_RGBplane']][0],\
                    y_label=plt_specifics['subTitles'][sim['slc_RGBplane']][1])  
    
fig, ax = plt.subplots(1,3,figsize = (12,4))
for i in range(NUM_GRID_PTS*NUM_GRID_PTS):
    i_lb = i*nSims
    i_ub = (i+1)*nSims
    #figure 1: comparison stimulus
    #figure 2: reference stimulus
    #figure 3: comparison - reference, which should be scattered around 0
    if plane_2D == 'GB plane':
        cm = (np.array([0,xref_jnp[i_lb,0],xref_jnp[i_lb,1]])+1)/2
    elif plane_2D == 'RB plane':
        cm = (np.array([xref_jnp[i_lb,0],0,xref_jnp[i_lb,1]])+1)/2
    else:
        cm = (np.array([xref_jnp[i_lb,0],xref_jnp[i_lb,1], 0])+1)/2
    ax[0].scatter(x1_jnp[i_lb:i_ub,0], x1_jnp[i_lb:i_ub,1], c = cm,s = 1, alpha = 0.1)
    ax[1].scatter(xref_jnp[i_lb:i_ub,0], xref_jnp[i_lb:i_ub,1],\
                  c = cm,s = 50,marker = '+')
    ax[2].scatter(x1_jnp[i_lb:i_ub,0] - xref_jnp[i_lb:i_ub,0],\
                x1_jnp[i_lb:i_ub,1] - xref_jnp[i_lb:i_ub,1],\
                c = cm, s = 2, alpha = 0.5)
    if i in [0,1]:
        ax[i].set_xticks(np.unique(xgrid))
        ax[i].set_yticks(np.unique(xgrid))
        ax[i].set_xlim([-0.8, 0.8]);
        ax[i].set_ylim([-0.8, 0.8])
ax[0].set_title('x1')
ax[1].set_title('xref = x0')
ax[2].set_title('x1 - xref')
plt.tight_layout()
plt.show()
full_path = os.path.join('/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
        'ELPS_analysis/SanityChecks_FigFiles/', 'scaledData_'+plane_2D+'.png')
fig.savefig(full_path)

#%% -----------------------------
# Fit W by maximizing posterior
# -----------------------------

# Fit model, initialized at random W
W_init = 1e-3 * model.sample_W_prior(W_INIT_KEY)

# W_init = model.sample_W_prior(W_INIT_KEY)

# # Run optimization.
# W_est, iters, objhist = optim.autotuned_map_estimate(
#     W_init, data, model, OPT_KEY,
#     mc_samples=MC_SAMPLES,
#     save_every=1
# )
opt_params = {
    "learning_rate": 1e-2,
    "momentum": 0.2,
    "mc_samples": 1000,
    "bandwidth": 1e-4,
}
W_est, iters, objhist = optim.optimize_posterior(
    W_init, data, model, OPT_KEY,
    opt_params,
    total_steps=5000,
    save_every=1,
    show_progress=True
)

#%%
fig, ax = plt.subplots(1, 1)
plt.rcParams['figure.dpi'] = 250 
ax.plot(iters, objhist)
fig.tight_layout()

Sigmas_init_grid = model.compute_Sigmas(
    model.compute_U(W_init, xgrid)
)
Sigmas_est_grid = model.compute_Sigmas(
    model.compute_U(W_est, xgrid)
)

if plane_2D == 'GB plane':
    gt_sigma = results['fitEllipse_scaled'][0]
elif plane_2D == 'RB plane':
    gt_sigma = results['fitEllipse_scaled'][1]
else:
    gt_sigma = results['fitEllipse_scaled'][2]
fig, axes = plt.subplots(1, 2, figsize = (6,3))
for i in range(NUM_GRID_PTS):
    for j in range(NUM_GRID_PTS):
        if plane_2D == 'GB plane':
            cm = (np.array([0,xgrid[i, j,0],xgrid[i, j,1]])+1)/2
        elif plane_2D == 'RB plane':
            cm = (np.array([xgrid[i, j,0],0,xgrid[i, j,1]])+1)/2
        else:
            cm = (np.array([xgrid[i, j,0],xgrid[i, j,1],0])+1)/2
        axes[0].plot(gt_sigma[i,j,0,:] - xref_temp[0,i,j] + xgrid[i, j,0],\
                     gt_sigma[i,j,1,:] - xref_temp[1,i,j] + xgrid[i, j,1], color = cm)
        axes[1].plot(gt_sigma[i,j,0,:] - xref_temp[0,i,j] + xgrid[i, j,0],\
                     gt_sigma[i,j,1,:] - xref_temp[1,i,j] + xgrid[i, j,1], color = cm)
        viz.plot_ellipse(
            axes[0], xgrid[i, j,:], 0.5 * Sigmas_init_grid[i, j,:,:], color="k", alpha=.5, lw=1
        )
        viz.plot_ellipse(
            axes[1], xgrid[i, j,:], 0.5 * Sigmas_est_grid[i, j,:,:], color="k", alpha=.5, lw=1
        )
axes[0].set_xticks(np.unique(xgrid))
axes[0].set_yticks(np.unique(xgrid))
axes[0].set_xlim([-0.8, 0.8]); axes[0].set_ylim([-0.8, 0.8])
axes[1].set_xticks(np.unique(xgrid))
axes[1].set_yticks(np.unique(xgrid))
axes[1].set_xlim([-0.8, 0.8]); axes[1].set_ylim([-0.8, 0.8])
axes[0].set_title("before training")
axes[1].set_title("after training")
fig.tight_layout()
plt.show()
fig_outputDir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                        'ELPS_analysis/ModelFitting_FigFiles/'
fig_name = 'Fitted' + file_name[4:-4]
full_path = os.path.join(fig_outputDir,fig_name+'.png')
fig.savefig(full_path)    

#%% save data
outputDir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                        'ELPS_analysis/ModelFitting_DataFiles/'
full_path = f"{outputDir}{fig_name}"

variable_names = ['data','W_init','opt_params', 'W_est', 'iters', 'objhist',\
                  'Sigmas_init_grid', 'Sigmas_est_grid',]
vars_dict = {}
for i in variable_names: vars_dict[i] = eval(i)

# Write the list of dictionaries to a file using pickle
with open(full_path, 'wb') as f:
    pickle.dump(vars_dict, f)









