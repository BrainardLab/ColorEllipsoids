#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 15:54:46 2024

@author: fangfang
"""
import pickle
import os
import jax
import sys
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids")
from core import chebyshev, viz, utils, oddity_task, optim, model_predictions
from core.wishart_process import WishartProcessModel
             

#%% three variables we need to define for loading the data
plane_2D   = 'GB plane'
plane_2D_dict = {'GB plane': 0, 'RB plane': 1, 'RG plane': 2}
plane_2D_idx  = plane_2D_dict[plane_2D]
sim_jitter = '0.1'
nSims      = 240 #number of simulations: 240 trials for each ref stimulus
BANDWIDTH  = 5e-3

#load file 1
#file 1
path_str      = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                'ELPS_analysis/Simulation_DataFiles/'
os.chdir(path_str)
file_CIE      = 'Isothreshold_contour_CIELABderived.pkl'
full_path     = f"{path_str}{file_CIE}"
with open(full_path, 'rb') as f: data_load = pickle.load(f)
stim          = data_load[1]
results       = data_load[2]

#load file 2
FileDir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                        'ELPS_analysis/ModelFitting_DataFiles/'
os.chdir(FileDir)
fileName = 'Fitted_isothreshold_'+plane_2D+'_sim'+str(nSims)+'perCond_'+\
                'samplingNearContour_jitter'+sim_jitter + '_bandwidth' + str(BANDWIDTH)+ '.pkl'
full_path = f"{FileDir}{fileName}"
with open(full_path, 'rb') as f:  data_load = pickle.load(f)

#use the estimated parameters as ground truth
W_gt         = data_load['W_est']
model        = data_load['model']
Ellipses_gt  = data_load['recover_fitEllipse_unscaled']
Sigma_gt_grid = data_load['Sigmas_est_grid']
opt_params   = data_load['opt_params']
xgrid        = data_load['xgrid']
DATA_KEY     = data_load['DATA_KEY']
OPT_KEY      = data_load['OPT_KEY']
NUM_GRID_PTS = data_load['NUM_GRID_PTS']
params_ellipses = data_load['params_ellipses']

#variable_names = ['NUM_GRID_PTS', 'MC_SAMPLES','BANDWIDTH', 'W_INIT_KEY',\
#                  'DATA_KEY', 'OPT_KEY', 'W_init','opt_params', 'xgrid']
#for i in variable_names: locals()[i] = data_load[i]

#%%
# -----------------------------------------
# Simulate data from the ground truth model
# -----------------------------------------

def simulate_oddityData_given_W_gt(M, W_gt, xGrid, nS, DATA_KEY, **kwargs):
    # Default parameters for ellipsoid fitting. Can be overridden by kwargs.
    params = {
        'noise_level':0.02,
        'numSD':2.46}  # 78% correct responses
    params.update(kwargs)
    
    data_keys = jax.random.split(DATA_KEY, num=4)
    
    #compute the total number of trials
    num_grid_pts1, num_grid_pts2 = xGrid.shape[0], xGrid.shape[1]
    nSims_total = nS*num_grid_pts1*num_grid_pts2
    
    #flatten xref
    xgridflat  = xGrid.reshape(-1, M.num_dims)
    ridx       = np.repeat(np.arange(xgridflat.shape[0]), nS)
    xref       = xgridflat[ridx]
    Sigmas_ref = M.compute_Sigmas(M.compute_U(W_gt, xref))

    x0          = jnp.copy(xref)
    _z          = jax.random.normal(data_keys[1], shape=(nSims_total, M.num_dims))
    noise_x1    = params['noise_level'] * jax.random.normal(\
                        data_keys[2], shape=(nSims_total, M.num_dims))
    x1          = x0 + jnp.einsum("ijk,ik->ij",params['numSD'] * utils.sqrtm(Sigmas_ref), \
                         _z / jnp.linalg.norm(_z, axis=1, keepdims=True)) + noise_x1
    
    x1          = jnp.clip(x1, -1.0, 1.0)
    Uref        = M.compute_U(W_gt, xref)
    U0          = M.compute_U(W_gt, x0)
    U1          = M.compute_U(W_gt, x1)
    y = 1 - jnp.maximum(0, jnp.sign(oddity_task.simulate_oddity((xref, x0, x1, Uref, U0, U1),
            jax.random.split(data_keys[2], num=nSims_total), 1, M.diag_term)).ravel())
    print("Proportion of correct trials:", jnp.mean(y))
    
    # Package data
    data = (y, xref, x0, x1)
    
    return data, noise_x1, Uref, U0, U1

#%%
NOISE_LEVEL = 0.02
nRepeats    = 10
nIters      = 3
opt_steps   = 500
data_all, noise_x1_all, Uref_all, U0_all, U1_all = [],[],[],[],[]
DATA_KEY, W_INIT_KEY, OPT_KEY, W_init, W_est, iters, objhist = [],[],[],[],[],[],[]
objhist_min     = np.full((nRepeats), np.nan)
objhist_min_idx = np.full((nRepeats), np.nan)
W_est_all       = np.full((nRepeats,) + W_gt.shape, np.nan)
Sigmas_est_grid_all = np.full((nRepeats,) + Sigma_gt_grid.shape, np.nan)

for i in range(nRepeats):
    DATA_KEY_i = jax.random.PRNGKey(i+200) 
    
    data_i, noise_x1_i, Uref_i, U0_i, U1_i = simulate_oddityData_given_W_gt(\
        model, W_gt, xgrid, nSims, DATA_KEY_i, noise_level = NOISE_LEVEL)
    #save
    DATA_KEY.append(DATA_KEY_i)
    data_all.append(data_i)
    noise_x1_all.append(noise_x1_i)
    Uref_all.append(Uref_i)
    U0_all.append(U0_i)
    U1_all.append(U1_i)
    
    W_INIT_KEY_i, OPT_KEY_i, W_init_i, W_est_i, iters_i, objhist_i = [],[],[],[],[],[]
    for j in range(nIters):
        W_INIT_KEY_ij = jax.random.PRNGKey(i*200+j) 
        OPT_KEY_ij = jax.random.PRNGKey(i*300+j) 
        W_init_ij = model.sample_W_prior(W_INIT_KEY_ij)
        W_est_ij, iters_ij, objhist_ij = optim.optimize_posterior(
            W_init_ij, data_i, model, OPT_KEY_ij,
            opt_params,
            total_steps=opt_steps,
            save_every=1,
            show_progress=True
        )
        
        #save
        W_INIT_KEY_i.append(W_INIT_KEY_ij)
        OPT_KEY_i.append(OPT_KEY_ij)
        W_init_i.append(W_init_ij)
        W_est_i.append(W_est_ij)
        iters_i.append(iters_ij)
        objhist_i.append(objhist_ij)
        
    #save
    W_INIT_KEY.append(W_INIT_KEY_i)
    OPT_KEY.append(OPT_KEY_i)
    W_init.append(W_init_i)
    W_est.append(W_est_i)
    iters.append(iters_i)
    objhist.append(objhist_i)
    
    objhist_last_i = [objhist[i][j][-1] for j in range(nIters)]
    objhist_min[i] = np.min(objhist_last_i)
    objhist_min_idx[i] = int(np.argmin(objhist_last_i))
    W_est_all[i] = W_est[i][int(np.argmin(objhist_last_i))]
    Sigmas_est_grid_all[i] = model.compute_Sigmas(model.compute_U(W_est_all[i], xgrid))
    #visualize
    y_i, xref_i, x0_i, x1_i = data_all[i]
    model_predictions.plot_2D_modelPredictions_byWishart(
        xgrid, x1_i, [], Sigmas_est_grid_all[i],[], plane_2D_idx,\
        saveFig = False,visualize_samples= True, visualize_modelPred = False,\
        samples_alpha = 0.2, nSims = nSims, plane_2D = plane_2D)

#%% save the data
import datetime

# Get today's date
today = datetime.date.today()

# Convert the date to a string (format: YYYY-MM-DD)
date_string = today.strftime("%Y-%m-%d")

#output path
outputDir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                        'ELPS_analysis/ParameterRecovery_DataFiles/'
output_file = 'ParameterRecovery' + fileName[6:-4] + '_Repeat'+str(nRepeats) +\
    '_noise' +str(NOISE_LEVEL) + '_' + date_string+ '.pkl'
full_path = f"{outputDir}{output_file}"
variable_names = ['NOISE_LEVEL', 'nRepeats','nIters', 'opt_steps','data_all',\
                  'noise_x1_all', 'Uref_all','U0_all', 'U1_all',\
                  'DATA_KEY', 'W_INIT_KEY', 'OPT_KEY','W_init', 'W_est',\
                  'iters', 'objhist','objhist_min', 'objhist_min_idx',\
                  'W_est_all','Sigmas_est_grid_all']
vars_dict = {}
for i in variable_names: vars_dict[i] = eval(i)

# Write the list of dictionaries to a file using pickle
with open(full_path, 'wb') as f:
    pickle.dump(vars_dict, f)
    
#%% visualize model predictions
# -----------------------------
# Compute model predictions
# -----------------------------
ngrid_search            = 250
bds_scaler_gridsearch   = [0.5, 3]
nTheta                  = 200
outputDir_fig = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                        'ELPS_analysis/ParameterRecovery_FigFiles/'
                        
#initalize
recover_fitEllipse_scaled, recover_fitEllipse_unscaled,\
    recover_rgb_comp_scaled, recover_rgb_contour_cov, params_ellipses = [],[],[],[],[]

for i in range(1):
    recover_fitEllipse_scaled_i, recover_fitEllipse_unscaled_i,\
        recover_rgb_comp_scaled_i, recover_rgb_contour_cov_i, params_ellipses_i =\
        model_predictions.convert_Sig_2DisothresholdContour_oddity_batch(xgrid,\
        [1,2], stim['grid_theta_xy'], 0.78,\
        W_est_all[i], model, results['opt_vecLen'], ngrid_bruteforce = ngrid_search,\
        scaler_bds_bruteforce = bds_scaler_gridsearch, scaler_x1 = 5,\
        nThetaEllipse = nTheta, mc_samples = opt_params['mc_samples'],bandwidth = BANDWIDTH,\
        opt_key = OPT_KEY)
    
    recover_fitEllipse_scaled.append(recover_fitEllipse_scaled_i)
    recover_fitEllipse_unscaled.append(recover_fitEllipse_unscaled_i)
    recover_rgb_comp_scaled.append(recover_rgb_comp_scaled_i)
    recover_rgb_contour_cov.append(recover_rgb_contour_cov_i)
    params_ellipses.append(params_ellipses_i)
            
    #visualize
    y_i, xref_i, x0_i, x1_i = data_all[i]
    model_predictions.plot_2D_modelPredictions_byWishart(
        xgrid, x1_i, Ellipses_gt, Sigmas_est_grid_all[i],recover_fitEllipse_unscaled_i, plane_2D_idx,\
        saveFig = False, visualize_samples= True, visualize_sigma = False,\
        visualize_modelPred = True,\
        visualize_groundTruth = True, samples_alpha = 0.2, \
        gt_mc = 'k', gt_ls = '--', gt_lw = 1, gt_alpha = 1,\
        modelpred_mc = 'k', modelpred_ls = '-', modelpred_lw = 2, modelpred_alpha = 1,\
        nSims = nSims, plane_2D = plane_2D,\
        figDir = outputDir_fig,fig_name = output_file[0:-4] +'_'+str(i))

#%% Directory containing images
image_folder = 'path/to/your/images'

# Collect image file names
images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images.sort()  # Sort the images by name (optional)

# Load images
image_list = [imageio.imread(f"{image_folder}/{img}") for img in images]

# Create a GIF
output_path = 'output.gif'
imageio.mimsave(output_path, image_list, fps=2)        
        
        
        
        
        
        
        