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
import matplotlib.pyplot as plt
import os
import pickle
import sys
import numpy as np

sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids")
from core import optim, model_predictions, oddity_task
from core.wishart_process import WishartProcessModel

# Add the parent directory of the aepsych package to sys.path
path_str = '/Users/fangfang/Documents/MATLAB/toolboxes/aepsych-main'
sys.path.append(path_str)
from aepsych.plotting import plot_strat
from aepsych.server import AEPsychServer

#three variables we need to define for loading the data
plane_2D      = 'RG plane'
varying_RGBplane = [0, 1]
nSims         = 240 #number of simulations: 240 trials for each ref stimulus

#%%
# -----------------------------------------------------------
# Load data simulated using CIELab and organize data
# -----------------------------------------------------------
#file 1
#which subject's data do we want to load
strat_name = 'threshold' #MCPV, LSE
nRefs = 25
subjN = list(range(1,nRefs+1))

refStimulus = np.full((nRefs,nSims,2), np.nan)
compStimulus = np.full((nRefs,nSims,2), np.nan)
responses = np.full((nRefs,nSims), np.nan)
xref_R, xref_G = np.meshgrid([0.2,0.35, 0.5,0.65, 0.8], [0.2,0.35, 0.5,0.65, 0.8])
xref_raw = np.stack((xref_R, xref_G))
xref_raw_reshape = np.reshape(xref_raw, (2,nRefs))

#load .db file
for s in range(len(subjN)):
    file_name = 'unity_color_oddity_blobby_2d_'+strat_name+'_subj'+str(subjN[s])+'.db' #glossiness
    path_str  = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                'Meta_analysis/Simulation_DataFiles/UnityTasks/ColorDiscrimination/'
    full_path = f"{path_str}{file_name}"
    os.chdir(path_str)
    serv = AEPsychServer(database_path = file_name)
    exp_ids = [rec.experiment_id for rec in serv.db.get_master_records()]
    serv.replay(exp_ids[0], skip_computations = True)
    strat = serv._strats[0]
    refStimulus[s] = np.tile(xref_raw_reshape[:,s], (nSims,1)) * 2 - 1
    compStimulus[s]  = strat.x[:nSims] /255 * 2 - 1
    responses[s] = strat.y[:nSims]
    del serv

xref_jnp = jnp.reshape(refStimulus, (nRefs*nSims,2))
x0_jnp = jnp.reshape(refStimulus, (nRefs*nSims,2))
x1_jnp = jnp.reshape(compStimulus, (nRefs*nSims,2))
y_jnp = jnp.reshape(responses, (nRefs*nSims))

"""
Fitting would be easier if we first scale things up, and then scale the model 
predictions back down
"""
scaler_x1  = 1
data = (y_jnp, xref_jnp, x0_jnp, x1_jnp)

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
W_INIT_KEY   = jax.random.PRNGKey(223)  # Key to initialize `W_est`. 
DATA_KEY     = jax.random.PRNGKey(333)  # Key to generate datatset.
OPT_KEY      = jax.random.PRNGKey(444)  # Key passed to optimizer.

# -----------------------------
# Fit W by maximizing posterior
# -----------------------------
# Fit model, initialized at random W
W_init = 1e-1*model.sample_W_prior(W_INIT_KEY) #1e-1*

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

fig, ax = plt.subplots(1, 1)
ax.plot(iters, objhist)
fig.tight_layout()

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
target_pC = 0.75
ngrid_search            = 1000
bds_scaler_gridsearch   = [2, 40]
nTheta                  = 200
#sample total of 16 directions (0 to 360 deg) 
numDirPts     = 16
grid_theta    = np.linspace(0,2*np.pi-np.pi/8,numDirPts)
grid_theta_xy = np.stack((np.cos(grid_theta),np.sin(grid_theta)),axis = 0)

vecLen_start            = np.full((3,NUM_GRID_PTS,NUM_GRID_PTS,16), 0.01)
recover_fitEllipse_scaled, recover_fitEllipse_unscaled,\
    recover_rgb_comp_scaled, recover_rgb_contour_cov, params_ellipses =\
    model_predictions.convert_Sig_2DisothresholdContour_oddity_batch(np.transpose(xgrid,(2,0,1)),\
    varying_RGBplane, grid_theta_xy, target_pC,\
    W_est, model, vecLen_start , ngrid_bruteforce = ngrid_search,\
    scaler_bds_bruteforce = bds_scaler_gridsearch, scaler_x1 = scaler_x1,\
    nThetaEllipse = nTheta, mc_samples = MC_SAMPLES,bandwidth = BANDWIDTH,\
    opt_key = OPT_KEY)
        
#%%
# -----------------------------
# Visualize model predictions
# -----------------------------
#specify figure name and path
fig_outputDir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                        'ELPS_analysis/ModelFitting_FigFiles/Python_version/'
#fig_name_part1 = 'Fitted' + file_sim[4:-4] + '_bandwidth' + str(BANDWIDTH)

flag_saveFig = False
    
#a different way to visualize it
model_predictions.plot_2D_modelPredictions_byWishart(
    np.transpose(xref_raw*2-1,(1,2,0)), xgrid, x1_jnp, [], Sigmas_est_grid, 
    recover_fitEllipse_unscaled, 2,\
    visualize_samples= True, visualize_sigma = True,\
    visualize_groundTruth = False, visualize_modelPred = True, samples_alpha = 0.3,\
    modelpred_mc = 'k', modelpred_ls = '--', modelpred_lw = 0.75, modelpred_alpha = 1,\
    plane_2D = plane_2D, saveFig = flag_saveFig)
 
        
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

