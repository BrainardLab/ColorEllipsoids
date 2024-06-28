#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 11:52:44 2024

@author: fangfang

This fits a Wishart Process model to the pilot human data. Trial placement
was guided by AEPsych. 

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
from core import optim, model_predictions
from core.wishart_process import WishartProcessModel

# Add the parent directory of the aepsych package to sys.path
path_str = '/Users/fangfang/Documents/MATLAB/toolboxes/aepsych-main'
sys.path.append(path_str)
from aepsych.server import AEPsychServer

#data path
path_str  = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
            'Meta_analysis/Simulation_DataFiles/UnityTasks/ColorDiscrimination/'

#%%
# -----------------------------------------------------------
# Load data simulated using CIELab and organize data
# -----------------------------------------------------------
# Define the color plane for the simulations
plane_2D         = 'RG plane' #the blue component (B) is constant at 0.5.
# Indices of the RGB dimensions that are varied, here R and G.
varying_RGBplane = [0, 1] 
nSims            = 240 # Number of simulations or trials per reference stimulus.
ref_size_dim1    = 5   # Number of reference stimuli along the Red (R) dimension.
ref_size_dim2    = 5   # Number of reference stimuli along the Green (G) dimension.
# Generate linearly spaced values for the Red dimension from 0.2 to 0.8.
xref_dim1        = np.linspace(0.2, 0.8, ref_size_dim1) 
xref_dim2        = np.linspace(0.2, 0.8, ref_size_dim2)
# Calculate the total number of reference stimuli from combinations of Red and Green values.
nRefs            = xref_dim1 * xref_dim2 
# Create mesh grids for Red and Green reference values.
xref_R, xref_G   = np.meshgrid(xref_dim1, xref_dim2)
# Stack the combinations of Red and Green values.
xref_raw         = np.stack((xref_R, xref_G))
# Reshape the stacked array for further use.
xref_raw_reshape = np.reshape(xref_raw, (2,nRefs))

# Initialize arrays to store reference and comparison stimuli, and observer responses.
refStimulus      = np.full((nRefs,nSims,2), np.nan)
compStimulus     = np.full(refStimulus.shape, np.nan)
#1: the observer successfully identified the odd stimulus
#0: the observer failed to identify the odd stimulus
responses        = np.full((nRefs,nSims), np.nan) 

# Iterate over each reference stimulus.
for s in range(range(1,nRefs+1)):
    #specify the file name
    file_name = 'unity_color_oddity_blobby_2d_threshold_'+plane_2D+'_nSims'+\
        str(nSims)+'_sub'+str(s)+'.db' #glossiness
    full_path = f"{path_str}{file_name}"
    os.chdir(path_str)
    # Initialize the AEPsych server to interact with the database.
    serv             = AEPsychServer(database_path = file_name)
    exp_ids          = [rec.experiment_id for rec in serv.db.get_master_records()]
    # Replay the experiment to load the data without redoing computations.
    serv.replay(exp_ids[0], skip_computations = True)
    strat            = serv._strats[0]
    #the original range of RGB is within [0,1], but chebyshev polynomial ranges from 
    #[-1,1], so we have to rescale the RGB values to fit in the new box 
    refStimulus[s]   = np.tile(xref_raw_reshape[:,s], (nSims,1)) * 2 - 1
    #do the same for the comparison stimulus
    compStimulus[s]  = strat.x[:nSims] /255 * 2 - 1
    # Retrieve the observer responses for the stimuli.
    responses[s]     = strat.y[:nSims]
    del serv

# Reshape the arrays for fitting.
xref_jnp = jnp.reshape(refStimulus, (nRefs*nSims,2))
x0_jnp   = jnp.reshape(refStimulus, (nRefs*nSims,2))
x1_jnp   = jnp.reshape(compStimulus, (nRefs*nSims,2))
y_jnp    = jnp.reshape(responses, (nRefs*nSims))

"""
In the other script simulate_2d_CIELab.py, the isothreshold contours where simulated
comparison stimuli are drawn from are very tiny, so I had to scale up the simulated 
stimuli to make fitting easier. Here, it seems like human thresholds are not very
small, so we do noot need to scale up the data.

Scaling factor for comparison stimuli; set to 1 as no scaling is needed.
"""
scaler_x1  = 1
# Pack the processed data into a tuple for further use.
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
target_pC = 0.63
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
                        'ELPS_analysis/ModelFitting_FigFiles/Python_version/2D/'
fig_name_part1 = 'Fitted_isothreshold_'+plane_2D+'_sim'+str(nSims)+'perCond_'+\
                'AEPsychSampling_bandwidth' + str(BANDWIDTH)

flag_saveFig = True
    
#a different way to visualize it
model_predictions.plot_2D_modelPredictions_byWishart(
    np.transpose(xref_raw*2-1,(1,2,0)), xgrid, x1_jnp, [], Sigmas_est_grid, 
    recover_fitEllipse_unscaled, 2, visualize_samples= True, \
    visualize_sigma = False, visualize_groundTruth = False,\
    visualize_modelPred = True, samples_alpha = 0.2,\
    samples_label = 'Human subject data',modelpred_mc = 'k', \
    modelpred_ls = '--', modelpred_lw = 0.75, modelpred_alpha = 1,\
    plane_2D = plane_2D, saveFig = flag_saveFig, figName = fig_name_part1,\
    figDir = fig_outputDir)
 
        
#%% save data
outputDir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                        'ELPS_analysis/ModelFitting_DataFiles/'
output_file = fig_name_part1 + '.pkl'
full_path = f"{outputDir}{output_file}"

variable_names = ['plane_2D', 'nSims', 'data','model',\
                  'NUM_GRID_PTS', 'MC_SAMPLES','BANDWIDTH', 'W_INIT_KEY',\
                  'DATA_KEY', 'OPT_KEY', 'W_init','opt_params', 'W_est',\
                  'iters', 'objhist','xgrid', 'Sigmas_init_grid',\
                  'Sigmas_est_grid','recover_fitEllipse_scaled',\
                  'recover_fitEllipse_unscaled', 'recover_rgb_comp_scaled',\
                  'recover_rgb_contour_cov','params_ellipses']
vars_dict = {}
for i in variable_names: vars_dict[i] = eval(i)

# Write the list of dictionaries to a file using pickle
with open(full_path, 'wb') as f:
    pickle.dump(vars_dict, f)

