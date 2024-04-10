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

NUM_GRID_PTS = 5      # Number of grid points over stimulus space.
NUM_TRIALS = 1000     # Number of trials in simulated dataset.

# Random number generator seeds
W_TRUE_KEY = jax.random.PRNGKey(111)  # Key to generate `W`.
W_INIT_KEY = jax.random.PRNGKey(222)  # Key to initialize `W_est`. 
DATA_KEY = jax.random.PRNGKey(333)    # Key to generate datatset.
OPT_KEY = jax.random.PRNGKey(444)     # Key passed to optimizer.

#%% ------------------------------------------
# Load data simulated using CIELab
# ------------------------------------------
import os
import pickle
import numpy as np

file_name = 'Sims_isothreshold_ellipsoids_sim240perCond_samplingNearContour_jitter0.1.pkl'
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
x1_temp_col1 = x1_temp[:,:,:,0,:].ravel()
x1_temp_col2 = x1_temp[:,:,:,1,:].ravel()
x1_temp_col3 = x1_temp[:,:,:,2,:].ravel()
x1_temp_reshaped = np.stack((x1_temp_col1, x1_temp_col2,x1_temp_col3), axis=1)
#convert it to jnp
x1_jnp = jnp.array(x1_temp_reshaped, dtype=jnp.float64)

#reference stimulus
xref_temp = sim['ref_points']
xref_temp = xref_temp * 2 -1
xref_temp_expanded = np.expand_dims(xref_temp, axis=-1)
xref_repeated = np.tile(xref_temp_expanded, (1, 1, 1, 1, nSims))
xref_temp_col1 = xref_repeated[:,:,:,0,:].ravel()
xref_temp_col2 = xref_repeated[:,:,:,1,:].ravel()
xref_temp_col3 = xref_repeated[:,:,:,2,:].ravel()
xref_temp_reshaped = np.stack((xref_temp_col1, xref_temp_col2,xref_temp_col3), axis=1)
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
#visualize raw data
fig = plt.figure(figsize = (10,5))
plt.rcParams['figure.dpi'] = 250
ax = fig.add_subplot(1,2, 1,projection = '3d')
ax1 = fig.add_subplot(1,2, 2,projection = '3d')
x1_amplified = (x1_jnp - x0_jnp)*3 + x0_jnp
slc_grid_seq = [0,2,4,10,12,14,20,22,24]
slc_grid = slc_grid_seq + [i+NUM_GRID_PTS**2*2 for i in slc_grid_seq] +\
          [i+NUM_GRID_PTS**2*4 for i in slc_grid_seq]
for i in range(NUM_GRID_PTS**3):
    if i in slc_grid:
        i_lb = i*nSims
        i_ub = (i+1)*nSims
        cm = (np.array(xref_jnp[i_lb])+1)/2
        ax.scatter(x1_amplified[i_lb:i_ub,0], x1_amplified[i_lb:i_ub,1],\
                   x1_amplified[i_lb:i_ub,2], s = 1, alpha = 0.5, c = cm)
        ax1.scatter(xref_jnp[i_lb:i_ub,0], xref_jnp[i_lb:i_ub,1],\
                    xref_jnp[i_lb:i_ub,2],s = 100, c = cm, marker = '+')
plt.tight_layout()
ax.set_xlim([-0.8, 0.8]); ax.set_ylim([-0.8, 0.8]); ax.set_zlim([-0.8, 0.8])
ax.set_title('x1'); ax1.set_title('xref')
ax.set_xlabel('R'); ax.set_ylabel('G'); ax.set_zlabel('B');
ax1.set_xlabel('R'); ax1.set_ylabel('G'); ax1.set_zlabel('B');
ax1.set_xlim([-0.8, 0.8]); ax1.set_ylim([-0.8, 0.8]); ax1.set_zlim([-0.8, 0.8])
ticks = np.unique(xgrid)
ax.set_xticks(ticks); ax.set_yticks(ticks); ax.set_zticks(ticks);
ax1.set_xticks(ticks); ax1.set_yticks(ticks); ax1.set_zticks(ticks);

#%%
# -----------------------------
# Fit W by maximum a posteriori
# -----------------------------

# Fit model, initialized at random W
W_init = 0.001*model.sample_W_prior(W_INIT_KEY)
#W_init = initialize_model(data, W_INIT_KEY)

opt_params = {
    "learning_rate": 1e-3,
    "momentum": 0.2,
    "mc_samples": 1000,
    "bandwidth": 1e-4,
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

#compute covariance matrices
Sigmas_init_grid = model.compute_Sigmas(
    model.compute_U(W_init, xgrid)
)
Sigmas_est_grid = model.compute_Sigmas(
    model.compute_U(W_est, xgrid)
)

#%% load ground truths
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
gt_ellipses = results['fitEllipse_scaled']

#%% plotting
scaling_factor = 3
k = 2; 
idx_list = [[1, 2],[0, 2],[0, 1]]
plt.rcParams['figure.dpi'] = 250
for _idx in range(3):
    fig, axes = plt.subplots(1, 2)
    idx = jnp.array(idx_list[_idx])
    for i in range(NUM_GRID_PTS):
        for j in range(NUM_GRID_PTS):
            if _idx == 0:   ii,jj,kk = k,i,j; 
            elif _idx == 1: ii,jj,kk = i,k,j; 
            elif _idx == 2: ii,jj,kk = i,j,k; 
            
            cm = (xgrid[jj,ii,kk]+1)/2
            #scatter plot the data
            x1_ij = x1_temp[ii,jj,kk,idx]
            xref_ij = xref_temp[ii,jj,kk,idx].reshape((2,1))
            axes[0].scatter((x1_ij[0] - xref_ij[0])*scaling_factor + xref_ij[0],\
                            (x1_ij[1] - xref_ij[1])*scaling_factor + xref_ij[1],\
                            s=1,c = cm, alpha=0.1)
            
            #ground truth ellipses
            gt_ellipses_ij = gt_ellipses[_idx,j,i]
            gt_ellipses_ij_scaled = gt_ellipses_ij - (xref_ij+1)/2 + xref_ij
            axes[0].plot(gt_ellipses_ij_scaled[0], gt_ellipses_ij_scaled[1],color = np.array(cm))
            #visualize the fits
            
            viz.plot_ellipse(axes[1],
                xgrid[jj,ii,kk, idx],
                scaling_factor/4*Sigmas_init_grid[jj,ii,kk][idx][:, idx],
                color="k", alpha=.2, lw=1
            )  
            
            viz.plot_ellipse(axes[1],
                xgrid[jj,ii,kk, idx],
                scaling_factor/4*Sigmas_est_grid[jj,ii,kk][idx][:, idx],
                color="k", lw=1
            )      
    axes[0].set_aspect('equal'); axes[1].set_aspect('equal')    
    axes[0].set_xlim([-0.8,0.8]); axes[0].set_ylim([-0.8,0.8])
    axes[1].set_xlim([-0.8,0.8]); axes[1].set_ylim([-0.8,0.8])
    ticks = np.unique(xgrid)
    axes[0].set_xticks(ticks); axes[0].set_yticks(ticks); 
    axes[1].set_xticks(ticks); axes[1].set_yticks(ticks); ax1.set_zticks(ticks);
    axes[0].set_title('Ground truth: '+plt_specifics['subTitles'][_idx])
    axes[1].set_title("Model predictions")
    fig.tight_layout()
    plt.show()
    fig_outputDir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                            'ELPS_analysis/ModelFitting_FigFiles/'
    fig_name = 'Fitted' + file_name[4:-4]
    full_path = os.path.join(fig_outputDir,fig_name+'_iters1000_'+plt_specifics['subTitles'][_idx]+'.png')
    fig.savefig(full_path)    
