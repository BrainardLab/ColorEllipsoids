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
import imageio.v2 as imageio

import sys
sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids')
from core import viz, utils, oddity_task, model_predictions, optim
from core.wishart_process import WishartProcessModel
sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/Python version')
from Simulate_probCorrectResp_3D import plot_3D_sampledComp

#%% ------------------------------------------
# Load data simulated using CIELab
# ------------------------------------------
import os
import pickle
import numpy as np

nSims = 560
samplingMethod = 'NearContour' #'Random'
match samplingMethod:
    case 'NearContour':
        jitter = 0.1
        file_name = 'Sims_isothreshold_ellipsoids_sim'+str(nSims)+\
                'perCond_samplingNearContour_jitter'+str(jitter)+'.pkl'
    case 'Random':
        Range = [-0.025, 0.025]
        file_name = 'Sims_isothreshold_ellipsoids_sim'+str(nSims)+\
                'perCond_samplingRandom_range'+str(Range)+'.pkl'        
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
#we take 5 samples from each color dimension
#but sometimes we don't want to sample that finely. Instead, we might just pick
#2 or 3 samples from each color dimension, and see how well the model can 
#interpolate between samples
idx_trim  = [0,2,4] #list(range(5))
#x1_raw is unscaled
data, x1_raw, xref_raw = model_predictions.organize_data(sim,\
        scaler_x1, slc_idx = idx_trim, visualize_samples = False)
# unpackage data
ref_size_dim1, ref_size_dim2, ref_size_dim3 = x1_raw.shape[0:3]
y_jnp, xref_jnp, x0_jnp, x1_jnp = data 


#%% load ground truths
fixedRGB_val_full = np.array([0.2,0.35,0.5,0.65,0.8])
fixedRGB_val = fixedRGB_val_full[idx_trim]
fixedRGB_val_scaled = [round(item*2-1,2) for item in fixedRGB_val]
path_str = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
        'ELPS_analysis/Simulation_DataFiles/'
gt_ellipses = []
for v in fixedRGB_val:
    file_name2 = 'Isothreshold_contour_CIELABderived_fixedVal'+str(v)+'.pkl'
    full_path2 = f"{path_str}{file_name2}"
    os.chdir(path_str)
    #Here is what we do if we want to load the data
    with open(full_path2, 'rb') as f:
        # Load the object from the file
        data_load = pickle.load(f)
    _, _, results, plt_specifics = data_load[0], data_load[1], data_load[2], data_load[3]
    gt_ellipses.append(results['fitEllipse_unscaled'])

#file 2
file_name3 = 'Isothreshold_ellipsoid_CIELABderived.pkl'
full_path3 = f"{path_str}{file_name3}"
os.chdir(path_str)

#Here is what we do if we want to load the data
with open(full_path3, 'rb') as f:
    # Load the object from the file
    data_load = pickle.load(f)
param3D, stim3D, results3D, plt_specifics = data_load[0], data_load[1],\
    data_load[2], data_load[3]

for fixedPlane, varyingPlanes in zip(['R','G','B'], ['GB','RB','RG']):
    for val in fixedRGB_val:
        plot_3D_sampledComp(stim3D['grid_ref'][idx_trim]*2-1, \
            results3D['fitEllipsoid_unscaled'][idx_trim,:,:][:,idx_trim,:][:,:,idx_trim]*2-1,\
            x1_raw, fixedPlane, val*2-1, plt_specifics['nPhiEllipsoid'],\
            plt_specifics['nThetaEllipsoid'],\
            slc_grid_ref_dim1 = [0,1,2], slc_grid_ref_dim2 = [0,1,2],\
            surf_alpha =  0.1,\
            samples_alpha = 0.1,scaled_neg12pos1 = True,\
            x_bds_symmetrical = 0.05,y_bds_symmetrical = 0.05,\
            z_bds_symmetrical = 0.05,title = varyingPlanes+' plane',\
            saveFig = False, figDir = path_str[0:-10] + 'FigFiles/',\
            figName = file_name + '_' + varyingPlanes + 'plane' +'_fixedVal'+str(val))

#%%
# -------------------------------
# Constants describing simulation
# -------------------------------
model = WishartProcessModel(
    5,     # Degree of the polynomial basis functions #5
    3,     # Number of stimulus dimensions
    1,     # Number of extra inner dimensions in `U`. #1
    3e-4,  # Scale parameter for prior on `W`.
    0.4,   # Geometric decay rate on `W`.  #0.4
    0,     # Diagonal term setting minimum variance for the ellipsoids.
)

NUM_GRID_PTS = 5           # Number of grid points over stimulus space.
MC_SAMPLES   = 2000        # Number of simulated trials to compute likelihood.
BANDWIDTH    = 5e-3        # Bandwidth for logistic density function. #5e-3

# Random number generator seeds
W_INIT_KEY   = jax.random.PRNGKey(222)  # Key to initialize `W_est`. 
DATA_KEY     = jax.random.PRNGKey(333)  # Key to generate datatset.
OPT_KEY      = jax.random.PRNGKey(444)  # Key passed to optimizer.

# -----------------------------
# Fit W by maximum a posteriori
# -----------------------------
# Fit model, initialized at random W
W_init = 1e-1*model.sample_W_prior(W_INIT_KEY)  

opt_params = {
    "learning_rate": 5e-2, #1e-3,
    "momentum": 0.2,
    "mc_samples": MC_SAMPLES,
    "bandwidth": BANDWIDTH,
}
W_est, iters, objhist = optim.optimize_posterior(
    W_init, data, model, OPT_KEY,
    opt_params,
    total_steps=20,
    save_every=1,
    show_progress=True
)

fig, ax = plt.subplots(1, 1)
ax.plot(iters, objhist)
fig.tight_layout()
plt.show()

# -----------------------------
# Rocover covariance matrices
# -----------------------------
# Specify grid over stimulus space
xgrid_1d = jnp.linspace(jnp.min(xref_jnp), jnp.max(xref_jnp), NUM_GRID_PTS)
xgrid = jnp.stack(jnp.meshgrid(*[xgrid_1d for _ in range(model.num_dims)]), axis=-1)

Sigmas_init_grid = model.compute_Sigmas(model.compute_U(W_init, xgrid))
Sigmas_est_grid = model.compute_Sigmas(model.compute_U(W_est, xgrid))

#%% save data
outputDir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                        'ELPS_analysis/ModelFitting_DataFiles/'
name_ext = '_withInterpolations' if np.prod(xref_raw.shape[0:3]) < np.prod(xgrid.shape[0:3]) else ''
output_file = 'Fitted'+file_name[4:-4]+'_bandwidth' + str(BANDWIDTH) + name_ext+'.pkl'
#    '_maxDeg' + str(model.degree)+'.pkl'
full_path4 = f"{outputDir}{output_file}"

variable_names = ['data', 'x1_raw', 'xref_raw', 'gt_ellipses','model',\
                  'NUM_GRID_PTS', 'MC_SAMPLES','BANDWIDTH', 'W_INIT_KEY',\
                  'DATA_KEY', 'OPT_KEY', 'W_init','opt_params', 'W_est',\
                  'iters', 'objhist','xgrid', 'Sigmas_init_grid',\
                  'Sigmas_est_grid']
vars_dict = {}
for i in variable_names: vars_dict[i] = eval(i)

# Write the list of dictionaries to a file using pickle
with open(full_path4, 'wb') as f:
    pickle.dump(vars_dict, f)

#%% Model predictions
grid_theta            = stim3D['grid_theta'] #from 0 to 2*pi
n_theta               = len(grid_theta)
n_theta_finergrid     = plt_specifics['nThetaEllipsoid']
grid_phi              = stim3D['grid_phi'] #from 0 to pi
n_phi                 = len(grid_phi)
n_phi_finergrid       = plt_specifics['nPhiEllipsoid']
nSteps_bruteforce     = 200 #number of grids
bds_scaler_gridsearch = [0.5, 3]
pC_threshold          = 0.78            

recover_fitEllipsoid_scaled, recover_fitEllipsoid_unscaled,\
    recover_rgb_comp_scaled, recover_rgb_contour_cov,\
    params_ellipsoids = model_predictions.convert_Sig_3DisothresholdContour_oddity_batch(\
        np.transpose(xgrid,(1,0,2,3)), stim3D['grid_xyz'], pC_threshold, W_est, model,\
        results3D['opt_vecLen'], scaler_x1 = scaler_x1,\
        ngrid_bruteforce=nSteps_bruteforce,\
        scaler_bds_bruteforce = bds_scaler_gridsearch,\
        bandwidth = opt_params['bandwidth'], opt_key = OPT_KEY,\
        search_method='brute force')
        
#%%derive 2D slices
# Initialize 3D covariance matrices for ground truth and predictions
gt_covMat   = np.full((NUM_GRID_PTS, NUM_GRID_PTS, NUM_GRID_PTS, 3, 3), np.nan)
pred_covMat = np.full(gt_covMat.shape, np.nan)

# Loop through each reference color in the 3D space
for g1 in range(NUM_GRID_PTS):
    for g2 in range(NUM_GRID_PTS):
        for g3 in range(NUM_GRID_PTS):
            #Convert the ellipsoid parameters to covariance matrices for the 
            #ground truth
            gt_covMat[g1,g2,g3] = model_predictions.ellParams_to_covMat(\
                            results3D['ellipsoidParams'][g1,g2,g3]['radii'],\
                            results3D['ellipsoidParams'][g1,g2,g3]['evecs'])
            ## Convert the ellipsoid parameters to covariance matrices for
            #the model predictions
            pred_covMat[g1,g2,g3] = model_predictions.ellParams_to_covMat(\
                            params_ellipsoids[g1][g2][g3]['radii'],\
                            params_ellipsoids[g1][g2][g3]['evecs'])
# Compute the 2D ellipse slices from the 3D covariance matrices for both ground 
#truth and predictions
gt_slice_2d_ellipse = model_predictions.covMat3D_to_2DsurfaceSlice(gt_covMat)
pred_slice_2d_ellipse = model_predictions.covMat3D_to_2DsurfaceSlice(pred_covMat)

#%% append data
#append data to existing file
# Load existing data from the pickle file
with open(full_path4, 'rb') as f:
    data_existing = pickle.load(f)
# Append new data
new_data = {'recover_fitEllipsoid_scaled': recover_fitEllipsoid_scaled,\
            'recover_fitEllipsoid_unscaled': recover_fitEllipsoid_unscaled,\
            'recover_rgb_comp_scaled': recover_rgb_comp_scaled,\
            'recover_rgb_contour_cov': recover_rgb_contour_cov,\
            'params_ellipsoids': params_ellipsoids,\
            'gt_covMat':gt_covMat,\
            'pred_covMat':pred_covMat,\
            'gt_slice_2d_ellipse':gt_slice_2d_ellipse,\
            'pred_slice_2d_ellipse':pred_slice_2d_ellipse}
data_existing.update(new_data)
# Save the updated dictionary back to the pickle file
with open(full_path4, 'wb') as f:
    pickle.dump(data_existing, f)
    
#%% plot figures and save them as png and gif
fig_outputDir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                        'ELPS_analysis/ModelFitting_FigFiles/Python_version/'
fig_name = 'Fitted' + file_name[4:-4] + name_ext #+'_maxDeg' + str(model.degree)
model_predictions.plot_3D_modelPredictions_byWishart(xref_raw, x1_raw,\
        xref_jnp, x1_jnp, np.transpose(xgrid,(1,0,2,3)), gt_covMat, Sigmas_est_grid,\
        recover_fitEllipsoid_scaled, gt_slice_2d_ellipse, pred_slice_2d_ellipse,\
        visualize_samples = True, saveFig = True, figDir = fig_outputDir,\
        figName = fig_name)   

# make a gif
images = [img for img in os.listdir(fig_outputDir) if img.startswith(fig_name)]
images.sort()  # Sort the images by name (optional)

# Load images using imageio.v2 explicitly to avoid deprecation warnings
image_list = [imageio.imread(f"{fig_outputDir}/{img}") for img in images]

# Create a GIF
gif_name = fig_name + '.gif'
output_path = f"{fig_outputDir}{gif_name}" 
imageio.mimsave(output_path, image_list, fps=2)  



