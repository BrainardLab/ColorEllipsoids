#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 21:01:22 2024

@author: fangfang
"""

#%% import modules
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
import pickle
import sys
import numpy as np

sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids")
from core import optim, model_predictions, oddity_task
from core.wishart_process import WishartProcessModel

#three variables we need to define for loading the data
plane_2D      = 'GB plane'
sim_jitter    = '0.1'
nSims         = 240 #number of simulations: 240 trials for each ref stimulus

#%%
# -----------------------------------------------------------
# Load data simulated using CIELab and organize data
# -----------------------------------------------------------
#file 1
path_str      = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                'ELPS_analysis/Simulation_DataFiles/'
os.chdir(path_str)
file_CIE      = 'Isothreshold_contour_CIELABderived.pkl'
full_path     = f"{path_str}{file_CIE}"
with open(full_path, 'rb') as f: data_load = pickle.load(f)
stim          = data_load[1]
results       = data_load[2]

#file 2
plane_2D_dict = {'GB plane': 0, 'RB plane': 1, 'RG plane': 2}
plane_2D_idx  = plane_2D_dict[plane_2D]
file_sim      = 'Sims_isothreshold_'+plane_2D+'_sim'+str(nSims)+'perCond_'+\
                'samplingNearContour_jitter'+sim_jitter+'.pkl'
full_path     = f"{path_str}{file_sim}"   
with open(full_path, 'rb') as f:  data_load = pickle.load(f)
sim = data_load[0]

"""
If we do not apply a scaler to x1
"""
scaler_x1  = 5
data, x1_raw, xref_raw = model_predictions.organize_data(sim, scaler_x1,\
                                                    visualize_samples = True,\
                                                    plane_2D = plane_2D)
ref_size_dim1, ref_size_dim2 = x1_raw.shape[0:2]
y_jnp, xref_jnp, x0_jnp, x1_jnp = data 

#%% -------------------------------
# Constants describing simulation
# -------------------------------
model = WishartProcessModel(
    5,     # Degree of the polynomial basis functions
    2,     # Number of stimulus dimensions
    1,     # Number of extra inner dimensions in `U`.
    4e-2,  # Scale parameter for prior on `W`.
    0.2,   # Geometric decay rate on `W`.
    0,     # Diagonal term setting minimum variance for the ellipsoids.
)

NUM_GRID_PTS = 5      # Number of grid points over stimulus space.
MC_SAMPLES   = 1000        # Number of simulated trials to compute likelihood.
#BANDWIDTH    = 1e-2        # Bandwidth for logistic density function.
BANDWIDTH    = 1e-3        # Bandwidth for logistic density function.

# Random number generator seeds
W_INIT_KEY   = jax.random.PRNGKey(223)  # Key to initialize `W_est`. 
DATA_KEY     = jax.random.PRNGKey(333)  # Key to generate datatset.
OPT_KEY      = jax.random.PRNGKey(444)  # Key passed to optimizer.

# -----------------------------
# Fit W by maximizing posterior
# -----------------------------
# Fit model, initialized at random W
W_init = 1e-2 * model.sample_W_prior(W_INIT_KEY)

opt_params = {
    "learning_rate": 1e-2,
    "momentum": 0.2,
    "mc_samples": MC_SAMPLES,
    "bandwidth": BANDWIDTH,
}

outputDir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                         'ELPS_analysis/ModelFitting_DataFiles/'  
output_file = 'Fitted' + file_sim[4:-4] + '_bandwidth' + str(BANDWIDTH) + '.pkl'
full_path = f"{outputDir}{output_file}"
with open(full_path, 'rb') as f:  data_load = pickle.load(f)
vars_dict = data_load
W_est = vars_dict['W_est']
iters = vars_dict['iters']
objhist = vars_dict['objhist']

#%%
# -----------------------------
# Rocover covariance matrices
# -----------------------------
# Specify grid over stimulus space
xgrid = jnp.stack(jnp.meshgrid(*[jnp.linspace(jnp.min(xref_jnp),\
                                              jnp.max(xref_jnp),\
                    NUM_GRID_PTS) for _ in range(model.num_dims)]), axis=-1)

Sigmas_init_grid = model.compute_Sigmas(model.compute_U(W_init, xgrid))
Sigmas_est_grid = model.compute_Sigmas(model.compute_U(W_est, xgrid))

# -----------------------------
# Compute model predictions
# -----------------------------
ngrid_search            = 500
bds_scaler_gridsearch   = [0.5, 3]
nTheta                  = 200
recover_fitEllipse_scaled, recover_fitEllipse_unscaled,\
    recover_rgb_comp_scaled, recover_rgb_contour_cov, params_ellipses =\
    model_predictions.convert_Sig_2DisothresholdContour_oddity_batch(xref_raw,\
    sim['varying_RGBplane'], stim['grid_theta_xy'], sim['pC_given_alpha_beta'],\
    W_est, model, results['opt_vecLen'], ngrid_bruteforce = ngrid_search,\
    scaler_bds_bruteforce = bds_scaler_gridsearch, scaler_x1 = scaler_x1,\
    nThetaEllipse = nTheta, mc_samples = MC_SAMPLES,bandwidth = BANDWIDTH,\
    opt_key = OPT_KEY)
        
#%% visualize how sampled data looks like at discrete chromatic directions
slc_ref_pts1 = 0
slc_ref_pts2 = 0
rgb_ref_s  = jnp.array(xref_raw[:,0,0])
recover_rgb_comp_scaled_slc = recover_rgb_comp_scaled[slc_ref_pts1, slc_ref_pts2]
recover_rgb_comp_unscaled_slc = (recover_rgb_comp_scaled_slc  -\
                               jnp.reshape(rgb_ref_s, (2, 1)))/scaler_x1 + jnp.reshape(rgb_ref_s, (2, 1))
numDirPts = stim['grid_theta_xy'].shape[1]
Uref      = model.compute_U(W_est, rgb_ref_s)
U0        = model.compute_U(W_est, rgb_ref_s)
Z1 =[]
Z0_to_zref = []
Z1_to_zref = []
#for each chromatic direction
for i in range(numDirPts):
    # Calculate RGB composition for current vector length
    rgb_comp = recover_rgb_comp_unscaled_slc[:,i]
    U1 = model.compute_U(W_est, rgb_comp)
        
    # Generate random draws from isotropic, standard gaussians
    keys = jax.random.split(OPT_KEY, num=6)
    nnref = jax.random.normal(keys[0], shape=(MC_SAMPLES, U1.shape[1]))
    nn0 = jax.random.normal(keys[1], shape=(MC_SAMPLES, U1.shape[1]))
    nn1 = jax.random.normal(keys[2], shape=(MC_SAMPLES, U1.shape[1]))

    # Re-scale and translate the noisy samples to have the correct mean and
    # covariance. For example, zref ~ Normal(mref, Uref @ Uref.T).
    zref = nnref @ Uref.T + rgb_ref_s[None, :]
    z0 = nn0 @ U0.T + rgb_ref_s[None, :] 
    z1 = nn1 @ U1.T + rgb_comp[None, :] 
    
    # Compute squared distance of each probe stimulus to reference
    z0_to_zref = jnp.sum((z0 - zref) ** 2, axis=1)
    z1_to_zref = jnp.sum((z1 - zref) ** 2, axis=1)
    
    Z0_to_zref.append(z0_to_zref)
    Z1_to_zref.append(z1_to_zref)
    
    Z1.append(z1)
    
#%%
default_cmap = plt.get_cmap('tab20b')
values = np.linspace(0, 1, numDirPts)
# Get the array of RGBA colors from the colormap
colors_array = default_cmap(values)
chromDir_deg = np.rad2deg(stim['grid_theta'])
fig, ax1 = plt.subplots(1,1)
plt.rcParams['figure.dpi'] = 250 
for i in range(numDirPts):
    ax1.scatter(Z1[i][:,0],Z1[i][:,1],c = colors_array[i],s = 5,\
                label = str(chromDir_deg[i]))
ax1.scatter(zref[:,0], zref[:,1],c=[0.5,0.5,0.5],s = 5)
ax1.grid(True)
ax1.set_aspect('equal')
ax1.set_xlim([-0.75,-0.45])
ax1.set_ylim([-0.75,-0.45])
ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8,ncol = 1,\
           title='Chromatic \ndirection (deg)')
plt.tight_layout()
plt.show()


fig, ax2 = plt.subplots(4,4,figsize = (10,8))
plt.rcParams['figure.dpi'] = 250 
for i in range(numDirPts):
    ax2[i//4, i%4].set_xlim([0, 0.01])
    ax2[i//4, i%4].hist(Z0_to_zref[i], bins = np.linspace(0,0.01,30),\
                        color=[0.5,0.5,0.5],alpha = 0.7)
    ax2[i//4, i%4].hist(Z1_to_zref[i], bins = np.linspace(0,0.01,30),\
                        color=colors_array[i],alpha = 0.8)
    if i%4 !=0: ax2[i//4, i%4].set_yticks([])
    if i//4 != 3: ax2[i//4, i%4].set_xticks([]); 
    else: ax2[i//4, i%4].set_xticks([0,0.01]); 
    ax2[i//4, i%4].tick_params(axis='x', labelsize=18)
    ax2[i//4, i%4].tick_params(axis='y', labelsize=18)
plt.tight_layout()
plt.show()