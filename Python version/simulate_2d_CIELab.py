#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 20:54:44 2024

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
    3,     # Degree of the polynomial basis functions
    2,     # Number of stimulus dimensions
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

func_path = '/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/Python version/'
os.chdir(func_path)
from Simulate_probCorrectResp import plot_2D_sampledComp

file_name = 'Isothreshold_contour_CIELABderived.pkl'
path_str = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
        'ELPS_analysis/Simulation_DataFiles/'
full_path = f"{path_str}{file_name}"
os.chdir(path_str)

#Here is what we do if we want to load the data
with open(full_path, 'rb') as f:
    # Load the object from the file
    data_load = pickle.load(f)
param, stim, results, plt_specifics = data_load[0], data_load[1], data_load[2], data_load[3]
    
#%%
file_name = 'Sims_isothreshold_GB plane_sim240perCond_samplingNearContour_jitter0.1.pkl'
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
#reshape the array so the final size is (6000, 3)
x1_temp_reshaped = np.reshape(x1_temp, (-1, 3))
#convert it to jnp
x1_jnp = jnp.array(x1_temp_reshaped, dtype=jnp.float64)

#reference stimulus
xref_temp = sim['ref_points']
xref_temp = xref_temp * 2 -1
xref_temp_expanded = np.expand_dims(xref_temp, axis=-1)
xref_repeated = np.tile(xref_temp_expanded, (1, 1, 1, nSims))
xref_temp_reshaped = np.reshape(xref_repeated, (-1,3))
xref_jnp = jnp.array(xref_temp_reshaped, dtype = jnp.float64)

#copy the reference stimulus
x0_jnp = jnp.copy(xref_jnp)

#visualize raw data
plot_2D_sampledComp(stim['grid_ref'], stim['grid_ref'], sim['rgb_comp'],\
                    sim['varying_RGBplane'], sim['method_sampling'], \
                    responses = sim['resp_binary'], \
                    groundTruth = results['fitEllipse_unscaled'][sim['slc_RGBplane']],\
                    x_label = plt_specifics['subTitles'][sim['slc_RGBplane']][0],\
                    y_label=plt_specifics['subTitles'][sim['slc_RGBplane']][1])  

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

# -----------------------------
# Fit W by maximum a posteriori
# -----------------------------
W_init = 1e-3*model.sample_W_prior(W_INIT_KEY)
opt_params = {
    "learning_rate": 1e-3,
    "momentum": 0.6,#0.6
    "mc_samples": 100,
    "bandwidth": 1e-2,
}

W_est, iters, objhist = optim.optimize_posterior(
    W_init, data, model, OPT_KEY,
    opt_params,
    total_steps=1000,
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







