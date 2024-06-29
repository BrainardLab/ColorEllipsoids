#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 16:05:29 2024

@author: fangfang
"""

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

sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/"+\
                "Python version")
import simulations_CIELab

# Add the parent directory of the aepsych package to sys.path
path_str = '/Users/fangfang/Documents/MATLAB/toolboxes/aepsych-main'
sys.path.append(path_str)
from aepsych.server import AEPsychServer

#%%
outputDir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                        'ELPS_analysis/ModelFitting_DataFiles/'
plane_2D       = 'RG plane' #the blue component (B) is constant at 0.5.
BANDWIDTH      = 0.005
nSims          = 240 # Number of simulations or trials per reference stimulus.
fig_name_part1 = 'Fitted_isothreshold_'+plane_2D+'_sim'+str(nSims)+'perCond_'+\
                'AEPsychSampling_bandwidth' + str(BANDWIDTH)
output_file    = fig_name_part1 + '.pkl'
full_path      = f"{outputDir}{output_file}"
#Here is what we do if we want to load the data
with open(full_path, 'rb') as f:
    # Load the object from the file
    data_load = pickle.load(f)
W_est = data_load['W_est']
model = data_load['model']
OPT_KEY = data_load['OPT_KEY']
params = data_load['opt_params']
MC_SAMPLES = params['mc_samples']
DIAG_TERM = model.diag_term
BANDWIDTH = params['bandwidth']
NUM_GRID_PTS = 100

#%%
def pChoosingX1_predictedby_WPM(rgb_ref, rgb_comp, W, OPT_KEY, MC_SAMPLES, DIAG_TERM, BANDWIDTH):
    # Extract and reshape the reference RGB values of the varying plane for processing
    Uref      = model.compute_U(W, rgb_ref)
    U0        = model.compute_U(W, rgb_ref)

    # comparison stimuli
    U1  = model.compute_U(W, rgb_ref)
    # Simulate the oddity task trial and compute the signed difference
    signed_diff = oddity_task.simulate_oddity_one_trial((rgb_ref, rgb_ref,\
        rgb_comp, Uref, U0, U1), OPT_KEY, MC_SAMPLES, DIAG_TERM)

    # Approximate the cumulative distribution function for the trial
    pChoosingX1 = oddity_task.approx_cdf_one_trial(0.0, signed_diff, BANDWIDTH)
    
    return pChoosingX1

def simResp_given_pChoosingX1(pX1, nTrials):
    randNum = np.random.rand(1, nTrials)
    logical_array = randNum  < pX1
    
    return logical_array.astype(int)

def convert_eig2Q(covM):
    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(covM)
    sorted_indices = np.argsort(eigenvalues)[::-1]  # Sort eigenvalues in descending order
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    # Semi-major and semi-minor axes (for 1 std deviation)
    a = np.sqrt(eigenvalues[0])
    b = np.sqrt(eigenvalues[1])
    
    # Rotation angle in degrees
    theta = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    
    return a, b, theta
    

#%% laod AEPsych server
xref_unique = np.linspace(-0.6, 0.6, NUM_GRID_PTS)
# Specify grid over stimulus space
#xgrid = jnp.stack(jnp.meshgrid(*[xref_unique for _ in range(model.num_dims)]), axis=-1)
xrand = np.random.rand(6000,1,2)*1.2 - 0.6
#xref_rep = np.reshape(xgrid, (NUM_GRID_PTS*NUM_GRID_PTS, 2))
Sigmas_est_grid = model.compute_Sigmas(model.compute_U(W_est, xrand))

target_pC = 0.63
ngrid_search     = 1000
varying_RGBplane = [0,1]
scaler_x1        =1
bds_scaler_gridsearch   = [2, 40]
nTheta                  = 200
#sample total of 16 directions (0 to 360 deg) 
numDirPts     = 16
grid_theta    = np.linspace(0,2*np.pi-np.pi/8,numDirPts)
grid_theta_xy = np.stack((np.cos(grid_theta),np.sin(grid_theta)),axis = 0)

vecLen_start  = np.full((3,NUM_GRID_PTS,NUM_GRID_PTS,16), 0.01)
recover_fitEllipse_scaled, recover_fitEllipse_unscaled,\
    recover_rgb_comp_scaled, recover_rgb_contour_cov, params_ellipses =\
    model_predictions.convert_Sig_2DisothresholdContour_oddity_batch(np.transpose(xgrid,(2,0,1)),\
    varying_RGBplane, grid_theta_xy, target_pC,\
    W_est, model, vecLen_start , ngrid_bruteforce = ngrid_search,\
    scaler_bds_bruteforce = bds_scaler_gridsearch, scaler_x1 = scaler_x1,\
    nThetaEllipse = nTheta, mc_samples = MC_SAMPLES,bandwidth = BANDWIDTH,\
    opt_key = OPT_KEY)

num_samples = 20
# Randomly sample 20 indices from the last dimension
indices = np.random.randint(0, recover_fitEllipse_unscaled.shape[-1],\
                            size=(recover_fitEllipse_unscaled.shape[0:3] + (num_samples,)))

# Use np.take_along_axis to sample along the last dimension
sampled_comp = np.take_along_axis(recover_fitEllipse_unscaled, indices, axis=-1)










