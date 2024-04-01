#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 17:01:45 2024

@author: fangfang
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt

import sys
sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids')
from core import viz, utils, oddity_task, optim
from core.wishart_process import WishartProcessModel

# -------------------------------
# Constants describing simulation
# -------------------------------
model = WishartProcessModel(
    5,     # Degree of the polynomial basis functions
    3,     # Number of stimulus dimensions
    1,     # Number of extra inner dimensions in `U`.
    4e-3,  # Scale parameter for prior on `W`.
    0.3,   # Geometric decay rate on `W`.
    1e-3,  # Diagonal term setting minimum variance for the ellipsoids.
)

NUM_GRID_PTS = 100      # Number of grid points over stimulus space.
NUM_TRIALS = 40000     # Number of trials in simulated dataset.

# Random number generator seeds
W_TRUE_KEY = jax.random.PRNGKey(113)  # Key to generate `W`.
W_INIT_KEY = jax.random.PRNGKey(222)  # Key to initialize `W_est`. 
DATA_KEY = jax.random.PRNGKey(333)    # Key to generate datatset.
OPT_KEY = jax.random.PRNGKey(444)     # Key passed to optimizer.

# ------------------------------------------
# Load data simulated using CIELab
# ------------------------------------------
import os
import pickle
import numpy as np

file_name = 'Sims_isothreshold_ellipsoids_sim240perCond_samplingGaussian_covMatrixScaler0.25.pkl'
path_str  = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
            'ELPS_analysis/Simulation_DataFiles/'
full_path = f"{path_str}{file_name}"
os.chdir(path_str)

#load data    
with open(full_path, 'rb') as f:
    # Load the object from the file
    data_load = pickle.load(f)
sim = data_load[0]
#comparisom stimulus; size: (5 x 5 x 5 x 3 x 240)
#the comparison stimulus was sampled around 5 x 5 x 5 different ref stimuli
x1_temp = sim['rgb_comp']
#the original data ranges from 0 to 1
#here we scale it so that it fits the [-1, 1] cube
x1_temp = x1_temp * 2 -1
#number of simulations: 240 trials for each ref stimulus
nSims = (x1_temp.shape)[-1] 
#reshape the array so the final size is (30000, 3)
x1_tmep_transpose = np.transpose(x1_temp, (0,1,2,4,3))
x1_temp_reshaped = np.reshape(x1_tmep_transpose, (-1, 3))
#convert it to jnp
x1_jnp = jnp.array(x1_temp_reshaped, dtype=jnp.float64)

#reference stimulus
xref_temp = sim['ref_points']
xref_temp = xref_temp * 2 -1
xref_temp_expanded = np.expand_dims(xref_temp, axis=-1)
xref_repeated = np.tile(xref_temp_expanded, (1, 1, 1, 1, nSims))
xref_temp_transpose = np.transpose(xref_repeated, (0,1,2,4,3))
xref_temp_reshaped = np.reshape(xref_temp_transpose, (-1,3))
xref_jnp = jnp.array(xref_temp_reshaped, dtype = jnp.float64)

#copy the reference stimulus
x0_jnp = jnp.copy(xref_jnp)

#visualize raw data
fig = plt.figure(figsize = (16,8))
ax = fig.add_subplot(1,2, 1,projection = '3d')
x1_amplified = (x1_jnp - x0_jnp)*3 + x0_jnp
ax.scatter(x1_amplified[:,0], x1_amplified[:,1], x1_amplified[:,2],s = 1, alpha = 0.2)
ax1 = fig.add_subplot(1,2, 2,projection = '3d')
ax1.scatter(xref_temp[:,:,:,0], xref_temp[:,:,:,1], xref_temp[:,:,:,2],s = 5)

#binary responses 
y_temp = jnp.array(sim['resp_binary'], dtype = jnp.float64)
y_temp_reshaped = y_temp.reshape(-1)
y_jnp = y_temp_reshaped.ravel()

# Specify grid over stimulus space
xgrid = jnp.stack(
    jnp.meshgrid(
        *[jnp.linspace(-1, 1, NUM_GRID_PTS) for _ in range(model.num_dims)]
    ), axis=-1
)

# Package data
data = (y_jnp, xref_jnp, x0_jnp, x1_jnp)
print("Proportion of correct trials:", jnp.mean(y_jnp))

#%% 
def initialize_model(data, key, num_sims=1000, degree=5, num_dims=3, extra_dims=1,
        mc_samples=1000, bandwidth=1e-6):

    k1, k2, k3, key = jax.random.split(key, 4)
    decay_rates = jax.random.uniform(k1, shape=(num_sims,))
    cov_scales = jax.random.uniform(k2, minval=-4, maxval=0, shape=(num_sims,)) ** 10
    diag_terms = jax.random.uniform(k3, minval=-8, maxval=-4, shape=(num_sims,)) ** 10
    models, vals = [], []

    for i in range(num_sims):
        model = WishartProcessModel(
            degree,      # Degree of the polynomial basis functions
            num_dims,    # Number of stimulus dimensions
            extra_dims,  # Number of extra inner dimensions in `U`.
            cov_scales[i],   # Scale parameter for prior on `W`.
            decay_rates[i],  # Geometric decay rate on `W`.
            diag_terms[i],   # Diagonal term setting minimum variance for the ellipsoids.
        )
        prior_key, loglike_key, key = jax.random.split(key, 3)
        vals.append(
            jnp.mean(jnp.array(
                [oddity_task.estimate_loglikelihood(
                    model.sample_W_prior(k),
                    model, data, loglike_key, mc_samples, bandwidth
                ) for k in jax.random.split(prior_key, 10)]
            ))
        )
        models.append(model)

    return models, jnp.array(vals)

# Fit model, initialized at random W
W_init = model.sample_W_prior(W_INIT_KEY)
#W_init = initialize_model(data, W_INIT_KEY)

#%%
# -----------------------------
# Fit W by maximum a posteriori
# -----------------------------

opt_params = {
    "learning_rate": 1e-3,
    "momentum": 0.6,
    "mc_samples": 100,
    "bandwidth": 1e-5,
}

W_est, iters, objhist = optim.optimize_posterior(
    W_init, data, model, OPT_KEY,
    opt_params,
    total_steps=10000,
    save_every=10,
    show_progress=True
)


# -----------------------------
# Plot results
# -----------------------------

fig, ax = plt.subplots(1, 1)
ax.plot(iters, objhist)
fig.tight_layout()
plt.show()

#%%
Sigmas_init_grid = model.compute_Sigmas(
    model.compute_U(W_init, xgrid)
)
Sigmas_est_grid = model.compute_Sigmas(
    model.compute_U(W_est, xgrid)
)

#%% plotting
scaling_factor = 3
k = 2; grid_slc_k = 50
grids_slc = [19, 34, 49, 64, 79]
idx_list = [[0, 1], [0, 2], [1, 2]]
for _idx in range(3):
    fig, axes = plt.subplots(1, 2)
    idx = jnp.array(idx_list[_idx])
    for i in range(len(grids_slc)):
        for j in range(len(grids_slc)):
            if _idx == 0:   ii,jj,kk = i,j,k; iii,jjj,kkk = grids_slc[i], grids_slc[j],grid_slc_k
            elif _idx == 1: ii,jj,kk = i,k,j; iii,jjj,kkk = grid_slc_k, grids_slc[i], grids_slc[j]
            elif _idx == 2: ii,jj,kk = k,i,j; iii,jjj,kkk = grids_slc[i], grid_slc_k, grids_slc[j]
            
            #scatter plot the data
            axes[0].scatter((x1_temp[ii,jj,kk,idx[0],:] - \
                             xref_temp[ii,jj,kk,idx[0]])*scaling_factor \
                + xref_temp[ii,jj,kk,idx[0]],(x1_temp[ii,jj,kk,idx[1],:] -\
                             xref_temp[ii,jj,kk,idx[1]])*scaling_factor \
                + xref_temp[ii,jj,kk,idx[1]],s=1,c = 'green', alpha=0.2)

            #visualize the fits
            
            viz.plot_ellipse(axes[1],
                xgrid[iii,jjj,kkk, idx],
                scaling_factor*Sigmas_init_grid[iii,jjj,kkk][idx][:, idx],
                color="k", alpha=.5, lw=5
            )  
            
            viz.plot_ellipse(axes[1],
                xgrid[iii,jjj,kkk, idx],
                scaling_factor*Sigmas_est_grid[iii,jjj,kkk][idx][:, idx],
                color="r", alpha=.5, lw=1
            )      
            
    axes[0].set_xlim([-1,1])
    axes[0].set_ylim([-1,1])
    axes[1].set_xlim([-1,1])
    axes[1].set_ylim([-1,1])
    axes[0].set_title("Data")
    axes[1].set_title("after training")
    fig.tight_layout()
plt.show()