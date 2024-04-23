#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 11:52:44 2024

@author: fangfang

This fits a Wishart Process model to the simulated data using the CIELab color space. 

"""
#%% import modules
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import sys

sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids")
from core import optim, model_predictions
from core.wishart_process import WishartProcessModel

#three variables we need to define for loading the data
plane_2D      = 'GB plane'
sim_jitter    = '0.1'
nSims         = 80 #number of simulations: 240 trials for each ref stimulus

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
BANDWIDTH    = 1e-2        # Bandwidth for logistic density function.
#BANDWIDTH    = 1e-3        # Bandwidth for logistic density function.

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
W_est, iters, objhist = optim.optimize_posterior(
    W_init, data, model, OPT_KEY,
    opt_params,
    total_steps=1000,
    save_every=1,
    show_progress=True
)  

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

#%% 
# -----------------------------
# Compute model predictions
# -----------------------------
ngrid_bruteforce            = 500
scaler_bds_bruteforce       = [0.5, 3]
nTheta                      = 200
params_ellipses             = [[]]*NUM_GRID_PTS
recover_fitEllipse_scaled   = np.full((NUM_GRID_PTS, NUM_GRID_PTS, 2, nTheta),\
                                      np.nan)
recover_fitEllipse_unscaled = np.full(recover_fitEllipse_scaled.shape, np.nan)
recover_rgb_comp_scaled     = np.full((NUM_GRID_PTS, NUM_GRID_PTS, 2,\
                                       stim['grid_theta_xy'].shape[-1]), np.nan)
recover_rgb_contour_cov     = np.full((NUM_GRID_PTS, NUM_GRID_PTS, 2, 2), np.nan)

#for each reference stimulus
for i in range(NUM_GRID_PTS):
    print(i)
    params_ellipses[i] = [[]]*NUM_GRID_PTS
    for j in range(NUM_GRID_PTS):
        #first grab the reference stimulus' RGB
        rgb_ref_scaled_ij = xref_raw[:,i,j]
        #insert the fixed R/G/B value to the corresponding plane
        rgb_ref_scaled_ij = np.insert(rgb_ref_scaled_ij, plane_2D_idx, 0)
        #from previous results, get the optimal vector length
        vecLength_ij = results['opt_vecLen'][plane_2D_idx,i,j]
        #use brute force to find the optimal vector length
        vecLength_test = np.linspace(\
            min(vecLength_ij)*scaler_x1*scaler_bds_bruteforce[0],\
            max(vecLength_ij)*scaler_x1*scaler_bds_bruteforce[1],\
            ngrid_bruteforce) 
        
        #fit an ellipse to the estimated comparison stimuli
        recover_fitEllipse_scaled[i,j], recover_fitEllipse_unscaled[i,j],\
            recover_rgb_comp_scaled[i,j], recover_rgb_contour_cov[i,j],\
            params_ellipses[i][j] = model_predictions.convert_Sig_2DisothresholdContour_oddity(\
                    rgb_ref_scaled_ij, sim['varying_RGBplane'], \
                    stim['grid_theta_xy'], vecLength_test,\
                    sim['pC_given_alpha_beta'],\
                    W_est, model, scaler_x1, nThetaEllipse = nTheta,\
                        opt_key = OPT_KEY,
                        mc_samples = MC_SAMPLES,
                        bandwidth = BANDWIDTH)

#%%
# -----------------------------
# Visualize model predictions
# -----------------------------

fig, ax = plt.subplots(1, 1)
ax.plot(iters, objhist)
fig.tight_layout()

#ground truth ellipses
gt_sigma = results['fitEllipse_scaled'][plane_2D_idx]
gt_sigma_scaled = (gt_sigma * 2 - 1)
#specify figure name and path
fig_outputDir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                        'ELPS_analysis/ModelFitting_FigFiles/Python_version/'
fig_name_part1 = 'Fitted' + file_sim[4:-4] + '_bandwidth' + str(BANDWIDTH)

#visualize the model predictions with samples
model_predictions.plot_2D_modelPredictions_byWishart(
    xgrid, x1_jnp, [], Sigmas_est_grid,[], plane_2D_idx,\
    saveFig = False,visualize_samples= True, visualize_modelPred = False,\
    samples_alpha = 0.2, nSims = nSims, plane_2D = plane_2D,\
    figDir = fig_outputDir,fig_name = fig_name_part1 +'_withSamples')
    
#a different way to visualize it
model_predictions.plot_2D_modelPredictions_byWishart(
    xgrid, x1_jnp, gt_sigma_scaled, Sigmas_est_grid, 
    recover_fitEllipse_unscaled, plane_2D_idx,\
    visualize_samples= False, visualize_sigma = False,\
    visualize_groundTruth = True, visualize_modelPred = True,\
    gt_mc = 'r', gt_ls = '--', gt_lw = 1, gt_alpha = 0.5, modelpred_mc = 'g',\
    modelpred_ls = '-', modelpred_lw = 2, modelpred_alpha = 0.5,\
    plane_2D = plane_2D, saveFig = False, figDir = fig_outputDir,\
    fig_name = fig_name_part1)
 
        
#%% save data
outputDir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                        'ELPS_analysis/ModelFitting_DataFiles/'
output_file = fig_name_part1 + '.pkl'
full_path = f"{outputDir}{output_file}"

variable_names = ['plane_2D', 'sim_jitter','nSims', 'data','model',\
                  'NUM_GRID_PTS', 'MC_SAMPLES','BANDWIDTH', 'W_INIT_KEY',\
                  'DATA_KEY', 'OPT_KEY', 'W_init','opt_params', 'W_est',\
                  'iters', 'objhist','xgrid', 'Sigmas_init_grid',\
                  'Sigmas_est_grid','recover_fitEllipse_scaled',\
                  'recover_fitEllipse_unscaled', 'recover_rgb_comp_scaled',\
                  'recover_rgb_contour_cov','params_ellipses','gt_sigma_scaled']
vars_dict = {}
for i in variable_names: vars_dict[i] = eval(i)

# Write the list of dictionaries to a file using pickle
with open(full_path, 'wb') as f:
    pickle.dump(vars_dict, f)

