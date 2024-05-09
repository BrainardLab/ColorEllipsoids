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
import warnings

import sys
sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids')
from core import viz, utils, oddity_task, model_predictions, optim
from core.wishart_process import WishartProcessModel

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

scaler_x1  = 5
data, x1_raw, xref_raw = model_predictions.organize_data(sim, scaler_x1,\
                                                    visualize_samples = False)
# unpackage data
ref_size_dim1, ref_size_dim2, ref_size_dim3 = x1_raw.shape[0:3]
y_jnp, xref_jnp, x0_jnp, x1_jnp = data 

#%%
NUM_GRID_PTS = 5
#visualize raw data
fig = plt.figure(figsize = (10,5))
plt.rcParams['figure.dpi'] = 250
ax = fig.add_subplot(1,2, 1,projection = '3d')
ax1 = fig.add_subplot(1,2, 2,projection = '3d')
slc_grid_seq = [0,2,4,10,12,14,20,22,24]
slc_grid = slc_grid_seq + [i+NUM_GRID_PTS**2*2 for i in slc_grid_seq] +\
          [i+NUM_GRID_PTS**2*4 for i in slc_grid_seq]
for i in range(NUM_GRID_PTS**3):
    if i in slc_grid:
        i_lb = i*sim['nSims']
        i_ub = (i+1)*sim['nSims']
        cm = (np.array(xref_jnp[i_lb])+1)/2
        ax.scatter(x1_jnp[i_lb:i_ub,0], x1_jnp[i_lb:i_ub,1],\
                   x1_jnp[i_lb:i_ub,2], s = 1, alpha = 0.5, c = cm)
        ax1.scatter(xref_jnp[i_lb:i_ub,0], xref_jnp[i_lb:i_ub,1],\
                    xref_jnp[i_lb:i_ub,2],s = 100, c = cm, marker = '+')
plt.tight_layout()
ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])
ax.set_title('x1'); ax1.set_title('xref')
ax.set_xlabel('R'); ax.set_ylabel('G'); ax.set_zlabel('B');
ax1.set_xlabel('R'); ax1.set_ylabel('G'); ax1.set_zlabel('B');
ax1.set_xlim([-0.8, 0.8]); ax1.set_ylim([-0.8, 0.8]); ax1.set_zlim([-0.8, 0.8])
#ticks = np.unique(xgrid)
#ax.set_xticks(ticks); ax.set_yticks(ticks); ax.set_zticks(ticks);
#ax1.set_xticks(ticks); ax1.set_yticks(ticks); ax1.set_zticks(ticks);

#%%
# -------------------------------
# Constants describing simulation
# -------------------------------
model = WishartProcessModel(
    5,     # Degree of the polynomial basis functions
    3,     # Number of stimulus dimensions
    1,     # Number of extra inner dimensions in `U`.
    3e-4,  # Scale parameter for prior on `W`.
    0.4,   # Geometric decay rate on `W`. 
    0,     # Diagonal term setting minimum variance for the ellipsoids.
)

NUM_GRID_PTS = 5      # Number of grid points over stimulus space.
MC_SAMPLES   = 2000        # Number of simulated trials to compute likelihood.
BANDWIDTH    = 5e-3        # Bandwidth for logistic density function.

# Random number generator seeds
W_INIT_KEY   = jax.random.PRNGKey(223)  # Key to initialize `W_est`. 
DATA_KEY     = jax.random.PRNGKey(333)  # Key to generate datatset.
OPT_KEY      = jax.random.PRNGKey(444)  # Key passed to optimizer.

# -----------------------------
# Fit W by maximum a posteriori
# -----------------------------
# Fit model, initialized at random W
W_init = model.sample_W_prior(W_INIT_KEY) #1e-1*

opt_params = {
    "learning_rate": 1e-2,
    "momentum": 0.2,
    "mc_samples": MC_SAMPLES,
    "bandwidth": BANDWIDTH,
}
W_est, iters, objhist = optim.optimize_posterior(
    W_init, data, model, OPT_KEY,
    opt_params,
    total_steps=100,
    save_every=1,
    show_progress=True
)

#%% -----------------------------
# Plot results
# -----------------------------

fig, ax = plt.subplots(1, 1)
ax.plot(iters, objhist)
fig.tight_layout()
plt.show()

# -----------------------------
# Rocover covariance matrices
# -----------------------------
# Specify grid over stimulus space
xgrid = jnp.stack(jnp.meshgrid(*[jnp.linspace(jnp.min(xref_jnp),\
                                              jnp.max(xref_jnp),\
                    NUM_GRID_PTS) for _ in range(model.num_dims)]), axis=-1)

Sigmas_init_grid = model.compute_Sigmas(model.compute_U(W_init, xgrid))
Sigmas_est_grid = model.compute_Sigmas(model.compute_U(W_est, xgrid))



#%% load ground truths
file_name2 = 'Isothreshold_contour_CIELABderived.pkl'
path_str = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
        'ELPS_analysis/Simulation_DataFiles/'
full_path2 = f"{path_str}{file_name2}"
os.chdir(path_str)

#Here is what we do if we want to load the data
with open(full_path2, 'rb') as f:
    # Load the object from the file
    data_load = pickle.load(f)
param, stim, results, plt_specifics = data_load[0], data_load[1], data_load[2], data_load[3]
gt_ellipses = results['fitEllipse_unscaled']

file_name3 = 'Isothreshold_ellipsoid_CIELABderived.pkl'
full_path3 = f"{path_str}{file_name3}"
os.chdir(path_str)

#Here is what we do if we want to load the data
with open(full_path3, 'rb') as f:
    # Load the object from the file
    data_load = pickle.load(f)
param3D, stim3D, results3D = data_load[0], data_load[1], data_load[2]  

#%%initialize

ngrid_bruteforce = 500
nThetaEllipse    = 200
contour_scaler   = 5
recover_fitEllipse_scaled   = np.full((NUM_GRID_PTS, NUM_GRID_PTS, NUM_GRID_PTS, 2, nThetaEllipse), np.nan)
recover_fitEllipse_unscaled = np.full(recover_fitEllipse_scaled.shape, np.nan)
recover_rgb_comp_scaled     = np.full((NUM_GRID_PTS, NUM_GRID_PTS, NUM_GRID_PTS, 2, stim['grid_theta_xy'].shape[-1]), np.nan)
recover_rgb_contour_cov     = np.full((NUM_GRID_PTS, NUM_GRID_PTS, NUM_GRID_PTS, 2, 2), np.nan)
recover_rgb_comp_est        = np.full((NUM_GRID_PTS, NUM_GRID_PTS, NUM_GRID_PTS, stim['grid_theta_xy'].shape[-1], 3), np.nan)

#for each reference stimulus
for i in range(NUM_GRID_PTS):
    print(i)
    for j in range(NUM_GRID_PTS):
        for k in range(NUM_GRID_PTS):
            #first grab the reference stimulus' RGB
            rgb_ref_scaled_ijk = xref_raw[i,j,k]
            #from previous results, get the optimal vector length
            vecLength_ijk = results3D['opt_vecLen'][i,j,k]
            #use brute force to find the optimal vector length
            vecLength_test = np.linspace(min(vecLength_ijk)*scaler_x1*0.5,\
                                         max(vecLength_ijk)*scaler_x1*2.5, ngrid_bruteforce) 
            
            #fit an ellipse to the estimated comparison stimuli
            recover_fitEllipse_scaled[i,j,:,:], recover_fitEllipse_unscaled[i,j,:,:],\
                recover_rgb_comp_scaled[i,j,:,:], recover_rgb_contour_cov[i,j,:,:],\
                recover_rgb_comp_est[i,j,:,:], _ = \
                    convert_Sig_3DisothresholdEllipsoid_oddity(\
                        rgb_ref_scaled_ij, sim['varying_RGBplane'], \
                        stim['grid_theta_xy'], vecLength_test,\
                        sim['pC_given_alpha_beta'],\
                        W_est)

#%% plotting
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
            x1_ij = x1_raw[ii,jj,kk,idx]
            xref_ij = xref_raw[ii,jj,kk,idx].reshape((2,1))
            axes[0].scatter((x1_ij[0] - xref_ij[0])*scaler_x1 + xref_ij[0],\
                            (x1_ij[1] - xref_ij[1])*scaler_x1 + xref_ij[1],\
                            s=1,c = cm, alpha=0.1)
            
            #ground truth ellipses
            gt_ellipses_ij = gt_ellipses[_idx,j,i]
            gt_ellipses_ij_scaled = (gt_ellipses_ij - (xref_ij+1)/2)*scaler_x1*2 + xref_ij
            axes[0].plot(gt_ellipses_ij_scaled[0], gt_ellipses_ij_scaled[1],color = np.array(cm))
            #visualize the fits
            
            viz.plot_ellipse(axes[1],
                xgrid[jj,ii,kk, idx],
                Sigmas_est_grid[jj,ii,kk][idx][:, idx],
                color="k", lw=1
            )      
    axes[0].set_aspect('equal'); axes[1].set_aspect('equal')    
    axes[0].set_xlim([-1,1]); axes[0].set_ylim([-1,1])
    axes[1].set_xlim([-1,1]); axes[1].set_ylim([-1,1])
    ticks = np.unique(xgrid)
    axes[0].set_xticks(ticks); axes[0].set_yticks(ticks); 
    axes[1].set_xticks(ticks); axes[1].set_yticks(ticks); ax1.set_zticks(ticks);
    axes[0].set_title('Ground truth: '+plt_specifics['subTitles'][_idx])
    axes[1].set_title("Model predictions")
    fig.tight_layout()
    plt.show()
    # fig_outputDir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
    #                         'ELPS_analysis/ModelFitting_FigFiles/'
    # fig_name = 'Fitted' + file_name[4:-4]
    # full_path = os.path.join(fig_outputDir,fig_name+'_iters1000_'+plt_specifics['subTitles'][_idx]+'.png')
    # fig.savefig(full_path)    
