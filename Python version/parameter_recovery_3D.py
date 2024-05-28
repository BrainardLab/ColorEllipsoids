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
import imageio.v2 as imageio
import datetime

sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids")
from core import chebyshev, viz, utils, oddity_task, optim, model_predictions
from core.wishart_process import WishartProcessModel
sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/Python version')
from parameter_recovery import simulate_oddityData_given_W_gt
             

#%% three variables we need to define for loading the data
sim_jitter = '0.1'
nSims      = 240 #number of simulations: 240 trials for each ref stimulus
BANDWIDTH  = 5e-3

#load file 1
#file 1
path_str      = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                'ELPS_analysis/Simulation_DataFiles/'
os.chdir(path_str)
file_CIE      = 'Isothreshold_ellipsoid_CIELABderived.pkl'
full_path     = f"{path_str}{file_CIE}"
with open(full_path, 'rb') as f: data_load = pickle.load(f)
stim          = data_load[1]
results       = data_load[2]

#load file 2
FileDir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                        'ELPS_analysis/ModelFitting_DataFiles/'
os.chdir(FileDir)
fileName = 'Fitted_isothreshold_ellipsoids_sim'+str(nSims)+'perCond_'+\
                'samplingNearContour_jitter'+sim_jitter + '_bandwidth' +\
                str(BANDWIDTH)+ '.pkl'
full_path = f"{FileDir}{fileName}"
with open(full_path, 'rb') as f:  data_load = pickle.load(f)

#use the estimated parameters as ground truth
W_gt         = data_load['W_est']
model        = data_load['model']
Ellipsoids_gt  = data_load['recover_fitEllipsoid_unscaled']
Sigma_gt_grid = data_load['Sigmas_est_grid']
opt_params   = data_load['opt_params']
xgrid        = data_load['xgrid']
DATA_KEY     = data_load['DATA_KEY']
OPT_KEY      = data_load['OPT_KEY']
NUM_GRID_PTS = data_load['NUM_GRID_PTS']
params_ellipsoids = data_load['params_ellipsoids']

#variable_names = ['NUM_GRID_PTS', 'MC_SAMPLES','BANDWIDTH', 'W_INIT_KEY',\
#                  'DATA_KEY', 'OPT_KEY', 'W_init','opt_params', 'xgrid']
#for i in variable_names: locals()[i] = data_load[i]

#%%
NOISE_LEVEL = 0.02 #0.005, 0.02
nRepeats    = 10
nIters      = 3
opt_steps   = 2000
data_all, noise_x1_all, Uref_all, U0_all, U1_all = [],[],[],[],[]
DATA_KEY, W_INIT_KEY, OPT_KEY, W_init, W_est, iters, objhist = [],[],[],[],[],[],[]
objhist_min     = np.full((nRepeats), np.nan)
objhist_min_idx = np.full((nRepeats), np.nan)
W_est_all       = np.full((nRepeats,) + W_gt.shape, np.nan)
Sigmas_est_grid_all = np.full((nRepeats,) + Sigma_gt_grid.shape, np.nan)

for i in range(1):#nRepeats):
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


#%% save the data
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
ngrid_search            = 400
bds_scaler_gridsearch   = [0.5, 10]
nTheta                  = 200
outputDir_fig = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                        'ELPS_analysis/ParameterRecovery_FigFiles/'
                        
#initalize
xgrid_3D = np.zeros((3,NUM_GRID_PTS,NUM_GRID_PTS))
xgrid_3D[0:2,:,:] = np.transpose(xgrid,(2,0,1))

for i in range(nRepeats):
    recover_fitEllipse_scaled_i, recover_fitEllipse_unscaled_i,\
        recover_rgb_comp_scaled_i, recover_rgb_contour_cov_i, params_ellipses_i =\
        model_predictions.convert_Sig_2DisothresholdContour_oddity_batch(xgrid_3D,\
        [1,2], stim['grid_theta_xy'], 0.78,\
        W_est_all[i], model, results['opt_vecLen'], ngrid_bruteforce = ngrid_search,\
        scaler_bds_bruteforce = bds_scaler_gridsearch, scaler_x1 = 5,\
        nThetaEllipse = nTheta, mc_samples = opt_params['mc_samples'],bandwidth = BANDWIDTH,\
        opt_key = OPT_KEY[i][int(objhist_min_idx[i])])
    #append data to existing file
    # Load existing data from the pickle file
    with open(full_path, 'rb') as f:
        data_existing = pickle.load(f)
    # Append new data
    new_data = {'recover_fitEllipse_scaled'+str(i): recover_fitEllipse_scaled_i,\
                'recover_fitEllipse_unscaled'+str(i): recover_fitEllipse_unscaled_i,\
                'recover_rgb_comp_scaled'+str(i): recover_rgb_comp_scaled_i,\
                'recover_rgb_contour_cov'+str(i): recover_rgb_contour_cov_i,\
                'params_ellipses' +str(i): params_ellipses_i}
    data_existing.update(new_data)
    # Save the updated dictionary back to the pickle file
    with open(full_path, 'wb') as f:
        pickle.dump(data_existing, f)
            
    #visualize
    y_i, xref_i, x0_i, x1_i = data_all[i]
    model_predictions.plot_2D_modelPredictions_byWishart(
        xgrid, x1_i, Ellipses_gt, Sigmas_est_grid_all[i],\
        recover_fitEllipse_unscaled_i, plane_2D_idx,\
        saveFig = True, visualize_samples= True, visualize_sigma = False,\
        visualize_modelPred = True, visualize_groundTruth = True, \
        samples_alpha = 0.2, samples_label = 'Simulated data based on WP model',\
        gt_mc = 'k', gt_ls = '-', gt_lw = 2, gt_alpha = 0.4, gt_label = 'Ground truths (WP model)',\
        modelpred_mc = 'k', modelpred_ls = '--', modelpred_lw = 0.5, modelpred_alpha = 1,\
        nSims = nSims, plane_2D = plane_2D,\
        figDir = outputDir_fig,fig_name = output_file[0:-4] +'_'+str(i))

#%
# output_file_temp = 'ParameterRecovery' + fileName[6:-4] + '_Repeat'+str(nRepeats) +\
#     '_noise' +str(NOISE_LEVEL) + '_2024-05-12.pkl'
# full_path_temp = f"{outputDir}{output_file_temp}"
# with open(full_path_temp, 'rb') as f:
#     data_existing = pickle.load(f)
    
# for i in range(nRepeats):
#     #visualize
#     y_i, xref_i, x0_i, x1_i = data_existing['data_all'][i]
#     recover_fitEllipse_unscaled_i = data_existing['recover_fitEllipse_unscaled'+str(i)]
#     model_predictions.plot_2D_modelPredictions_byWishart(
#         xgrid, x1_i, Ellipses_gt, Sigmas_est_grid_all[i],\
#         recover_fitEllipse_unscaled_i, plane_2D_idx,\
#         saveFig = True, visualize_samples= True, visualize_sigma = False,\
#         visualize_modelPred = True, visualize_groundTruth = True, \
#         samples_alpha = 0.2, samples_label = 'Simulated data based on WP model',\
#         gt_mc = 'k', gt_ls = '-', gt_lw = 2, gt_alpha = 0.4, gt_label = 'Ground truths (WP model)',\
#         modelpred_mc = 'k', modelpred_ls = '--', modelpred_lw = 0.5, modelpred_alpha = 1,\
#         nSims = nSims, plane_2D = plane_2D,\
#         figDir = outputDir_fig,fig_name = output_file[0:-4] +'_'+str(i))
        
#%% Directory containing images
# Collect image file names
images = [img for img in os.listdir(outputDir_fig) if img.endswith(".png")]
images.sort()  # Sort the images by name (optional)

# Load images using imageio.v2 explicitly to avoid deprecation warnings
image_list = [imageio.imread(f"{outputDir_fig}/{img}") for img in images]

# Create a GIF
gif_name = output_file[0:-4] + '.gif'
output_path = f"{outputDir_fig}{gif_name}" 
imageio.mimsave(output_path, image_list, fps=2)    
        
        
        
        
        
        
        