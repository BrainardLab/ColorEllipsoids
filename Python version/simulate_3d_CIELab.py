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
from core import viz, utils, oddity_task, optim
from core.wishart_process import WishartProcessModel
sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/Python version/")
from IsothresholdEllipsoids import fit_3d_isothreshold_ellipsoid

# -------------------------------
# Constants describing simulation
# -------------------------------
model = WishartProcessModel(
    5,     # Degree of the polynomial basis functions
    3,     # Number of stimulus dimensions
    1,     # Number of extra inner dimensions in `U`.
    4e-2,  # Scale parameter for prior on `W`.
    0.2,   # Geometric decay rate on `W`.
    0,  # Diagonal term setting minimum variance for the ellipsoids.
)

NUM_GRID_PTS = 5      # Number of grid points over stimulus space.
MC_SAMPLES = 1000     # Number of trials in simulated dataset.
BANDWIDTH = 1e-3

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
scaler_x1 = 5
x1_jnp = (x1_jnp - x0_jnp)*scaler_x1 + x0_jnp

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
slc_grid_seq = [0,2,4,10,12,14,20,22,24]
slc_grid = slc_grid_seq + [i+NUM_GRID_PTS**2*2 for i in slc_grid_seq] +\
          [i+NUM_GRID_PTS**2*4 for i in slc_grid_seq]
for i in range(NUM_GRID_PTS**3):
    if i in slc_grid:
        i_lb = i*nSims
        i_ub = (i+1)*nSims
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
ticks = np.unique(xgrid)
ax.set_xticks(ticks); ax.set_yticks(ticks); ax.set_zticks(ticks);
ax1.set_xticks(ticks); ax1.set_yticks(ticks); ax1.set_zticks(ticks);

#%%
# -----------------------------
# Fit W by maximum a posteriori
# -----------------------------

# Fit model, initialized at random W
W_init = 1e-2*model.sample_W_prior(W_INIT_KEY)

opt_params = {
    "learning_rate": 1e-3,
    "momentum": 0.2,
    "mc_samples": MC_SAMPLES,
    "bandwidth": BANDWIDTH,
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

#%%
def convert_Sig_2DisothresholdContour_oddity(rgb_ref, varying_RGBplane,
                                             grid_theta_xy, vecLength,
                                             pC_threshold, W, **kwargs):
    params = {
        'nThetaEllipse': 200,
        'contour_scaler': 5,
    }
    #update default options with any keyword arguments provided
    params.update(kwargs)
    
    #initialize
    nSteps_bruteforce = len(vecLength)
    numDirPts = grid_theta_xy.shape[1]
    recover_vecLength = np.full((numDirPts), np.nan)
    recover_rgb_comp_est = np.full((numDirPts, 3), np.nan)
    
    #grab the reference stimulus' RGB
    rgb_ref_s = jnp.array(rgb_ref[varying_RGBplane]).reshape((1,2))
    Uref = model.compute_U(W, rgb_ref_s)
    U0 = model.compute_U(W, rgb_ref_s)
    
    #for each chromatic direction
    for i in range(numDirPts):
        #determine the direction we are going
        vecDir = jnp.array(grid_theta_xy[:,i]).reshape((1,2))
    
        pChoosingX1 = np.full((nSteps_bruteforce), np.nan)
        for x in range(nSteps_bruteforce):
            rgb_comp = rgb_ref_s + vecDir * vecLength[x]
            U1 = model.compute_U(W, rgb_comp)
            #signed diff: z0_to_zref - z1_to_zref
            signed_diff = oddity_task.simulate_oddity_one_trial(\
                (rgb_ref_s[0], rgb_ref_s[0], rgb_comp[0], Uref[0], U0[0], U1[0]),\
                    OPT_KEY, MC_SAMPLES, model.diag_term)
            
            pChoosingX1[x] = oddity_task.approx_cdf_one_trial(\
                0.0, signed_diff, BANDWIDTH)
            
        # find the index that corresponds to the minimum |pChoosingX1 - pC_threshold|
        min_idx = np.argmin(np.abs(pChoosingX1 - pC_threshold))
        if min_idx in [0, nSteps_bruteforce-1]:
            warnings.warn('Expand the range for grid search!')
            return
        # find the vector length that corresponds to the minimum index
        recover_vecLength[i] = vecLength[min_idx]
        #find the comparison stimulus, which is the reference stimulus plus the
        #vector length in the direction of vecDir
        recover_rgb_comp_est[i, varying_RGBplane] = rgb_ref_s + \
            params['contour_scaler'] * vecDir * recover_vecLength[i]
    
    #find the R/G/B plane with fixed value
    fixed_RGBplane = np.setdiff1d([0,1,2], varying_RGBplane)
    #store the rgb values of the fixed plane
    recover_rgb_comp_est[:, fixed_RGBplane] = rgb_ref[fixed_RGBplane]
    
    #fit an ellipse to the estimated comparison stimuli
    fitEllipse_scaled, fitEllipse_unscaled, rgb_comp_scaled, \
        rgb_contour_cov, [xCenter, yCenter, majorAxis, minorAxis, theta] = \
            fit_3d_isothreshold_ellipsoid(rgb_ref, [], grid_theta_xy, \
                vecLength = recover_vecLength, varyingRGBplan = varying_RGBplane)

    return fitEllipse_scaled, fitEllipse_unscaled, rgb_comp_scaled, \
        rgb_contour_cov, recover_rgb_comp_est, \
            [xCenter, yCenter, majorAxis, minorAxis, theta]        

#%%initialize
# Specify grid over stimulus space
xgrid = jnp.stack(
    jnp.meshgrid(
        *[jnp.linspace(-0.6, 0.6, NUM_GRID_PTS) for _ in range(model.num_dims)]
    ), axis=-1
)

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
            rgb_ref_scaled_ijk = xref_temp[i,j,k]
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
            x1_ij = x1_temp[ii,jj,kk,idx]
            xref_ij = xref_temp[ii,jj,kk,idx].reshape((2,1))
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
                scaler_x1*Sigmas_init_grid[jj,ii,kk][idx][:, idx],
                color="k", alpha=.2, lw=1
            )  
            
            viz.plot_ellipse(axes[1],
                xgrid[jj,ii,kk, idx],
                scaler_x1*Sigmas_est_grid[jj,ii,kk][idx][:, idx],
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
