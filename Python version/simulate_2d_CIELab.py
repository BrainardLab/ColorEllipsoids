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

import sys
sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids")
from core import chebyshev, viz, utils, oddity_task, optim
from core.wishart_process import WishartProcessModel
sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/Python version/")
from IsothresholdContour_RGBcube import fit_2d_isothreshold_contour

# %% ------------------------------------------
# Load data simulated using CIELab
# ------------------------------------------
import os
import pickle

plane_2D_dict = {'GB plane': 0, 'RB plane': 1, 'RG plane': 2}
plane_2D      = 'RG plane'
plane_2D_idx  = plane_2D_dict[plane_2D]

file_name = 'Sims_isothreshold_'+plane_2D+'_sim240perCond_samplingNearContour_jitter0.1.pkl'
path_str  = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
            'ELPS_analysis/Simulation_DataFiles/'
full_path = f"{path_str}{file_name}"
os.chdir(path_str)

#load data    
with open(full_path, 'rb') as f:
    # Load the object from the file
    data_load = pickle.load(f)
sim              = data_load[0]
#comparisom stimulus; size: (5 x 5 x 3 x 240)
#the comparison stimulus was sampled around 5 x 5 different ref stimuli
#the original data ranges from 0 to 1; we scale it so that it fits the [-1, 1] cube
x1_temp          = sim['rgb_comp'][:,:,sim['varying_RGBplane'],:] * 2 -1
#number of reference stimuli
ref_size_dim1, ref_size_dim2 = x1_temp.shape[0:2]
#number of simulations: 240 trials for each ref stimulus
nSims            = (x1_temp.shape)[-1] 
#reshape the array so the final size is (2000, 3)
x1_temp_reshaped = np.transpose(x1_temp, (0,1,3,2)).reshape(-1, 2)
x1_jnp           = jnp.array(x1_temp_reshaped, dtype=jnp.float64)

#reference stimulus
xref_temp           = sim['ref_points'][sim['varying_RGBplane'],:,:] * 2 -1
xref_temp_expanded  = np.expand_dims(np.transpose(xref_temp, (1,2,0)), axis=-1)
xref_repeated       = np.tile(xref_temp_expanded, (1, 1, 1, nSims))
xref_temp_reshaped  = np.stack((xref_repeated[:,:,0,:].ravel(),xref_repeated[:,:,1,:].ravel()), axis=1)
xref_jnp            = jnp.array(xref_temp_reshaped, dtype = jnp.float64)

#copy the reference stimulus
x0_jnp = jnp.copy(xref_jnp)
x1_jnp = (x1_jnp - x0_jnp)*5 + x0_jnp

#binary responses 
y_temp = sim['resp_binary']
y_temp_jnp = jnp.array(sim['resp_binary'], dtype = jnp.float64)
y_jnp  = y_temp_jnp.ravel()

# Package data
data = (y_jnp, xref_jnp, x0_jnp, x1_jnp)
print("Proportion of correct trials:", jnp.mean(y_jnp))

#%%
file_name2 = 'Isothreshold_contour_CIELABderived.pkl'
path_str = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
        'ELPS_analysis/Simulation_DataFiles/'
full_path = f"{path_str}{file_name2}"
os.chdir(path_str)

#Here is what we do if we want to load the data
with open(full_path, 'rb') as f:
    # Load the object from the file
    data_load = pickle.load(f)
param, stim, results= data_load[0], data_load[1], data_load[2]
    
#import functions from the other script
func_path = '/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/Python version/'
os.chdir(func_path)
from Simulate_probCorrectResp import plot_2D_sampledComp

#%% visualize the simulated data
plot_2D_sampledComp(stim['grid_ref'], stim['grid_ref'], sim['rgb_comp'],\
                    sim['varying_RGBplane'], sim['method_sampling'], \
                    responses = sim['resp_binary'], \
                    groundTruth = results['fitEllipse_unscaled'][sim['slc_RGBplane']],\
                    x_label = plane_2D[0],\
                    y_label = plane_2D[1])  
    
#figure 1: comparison stimulus
#figure 2: reference stimulus
#figure 3: comparison - reference, which should be scattered around 0
fig, ax = plt.subplots(1,3,figsize = (12,4))
for i in range(ref_size_dim1*ref_size_dim2):
    i_lb = i*nSims
    i_ub = (i+1)*nSims
    #define the color map, which is the RGB value of the reference stimulus
    cm = (xref_jnp[i_lb] + 1)/2
    cm = np.insert(cm, plane_2D_idx, 0.5);
    ax[0].scatter(x1_jnp[i_lb:i_ub,0], x1_jnp[i_lb:i_ub,1], c = cm,s = 1, alpha = 0.5)
    ax[1].scatter(xref_jnp[i_lb:i_ub,0], xref_jnp[i_lb:i_ub,1],\
                  c = cm,s = 50,marker = '+')
    ax[2].scatter(x1_jnp[i_lb:i_ub,0] - xref_jnp[i_lb:i_ub,0],\
                x1_jnp[i_lb:i_ub,1] - xref_jnp[i_lb:i_ub,1],\
                c = cm, s = 2, alpha = 0.5)
    if i in [0,1]:
        ax[i].set_xticks(np.unique(xref_jnp)); ax[i].set_yticks(np.unique(xref_jnp))
        #ax[i].set_xlim([-0.8, 0.8]); ax[i].set_ylim([-0.8, 0.8])
        ax[i].set_xlim([-1, 1]); ax[i].set_ylim([-1, 1])
        ax[i].set_xlabel(plane_2D[0]); ax[i].set_ylabel(plane_2D[1])
ax[0].set_title('x1')
ax[1].set_title('xref = x0')
ax[2].set_title('x1 - xref')
plt.tight_layout()
plt.show()
full_path = os.path.join('/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
        'ELPS_analysis/SanityChecks_FigFiles/', 'scaledData_'+plane_2D+'.png')
#fig.savefig(full_path)

#%% -------------------------------
# Constants describing simulation
# -------------------------------
model = WishartProcessModel(
    3,     # Degree of the polynomial basis functions
    2,     # Number of stimulus dimensions
    1,     # Number of extra inner dimensions in `U`.
    4e-2,  # Scale parameter for prior on `W`.
    0.2,   # Geometric decay rate on `W`.
    0,     # Diagonal term setting minimum variance for the ellipsoids.
)

NUM_GRID_PTS = 5      # Number of grid points over stimulus space.
MC_SAMPLES   = 1000        # Number of simulated trials to compute likelihood.
BANDWIDTH    = 1e-3        # Bandwidth for logistic density function.

# Random number generator seeds
W_TRUE_KEY   = jax.random.PRNGKey(111)  # Key to generate `W`.
W_INIT_KEY   = jax.random.PRNGKey(222)  # Key to initialize `W_est`. 
DATA_KEY     = jax.random.PRNGKey(333)    # Key to generate datatset.
OPT_KEY      = jax.random.PRNGKey(444)     # Key passed to optimizer.

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

 #%% recover 
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
        print(min_idx)
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
            fit_2d_isothreshold_contour(rgb_ref, [], grid_theta_xy, \
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
recover_fitEllipse_scaled   = np.full((NUM_GRID_PTS, NUM_GRID_PTS, 2, nThetaEllipse), np.nan)
recover_fitEllipse_unscaled = np.full(recover_fitEllipse_scaled.shape, np.nan)
recover_rgb_comp_scaled     = np.full((NUM_GRID_PTS, NUM_GRID_PTS, 2, stim['grid_theta_xy'].shape[-1]), np.nan)
recover_rgb_contour_cov     = np.full((NUM_GRID_PTS, NUM_GRID_PTS, 2, 2), np.nan)
recover_rgb_comp_est        = np.full((NUM_GRID_PTS, NUM_GRID_PTS, stim['grid_theta_xy'].shape[-1], 3), np.nan)

#for each reference stimulus
for i in range(NUM_GRID_PTS):
    print(i)
    for j in range(NUM_GRID_PTS):
        #first grab the reference stimulus' RGB
        rgb_ref_scaled_ij = xref_temp[:,i,j]
        #insert the fixed R/G/B value to the corresponding plane
        rgb_ref_scaled_ij = np.insert(rgb_ref_scaled_ij, plane_2D_idx, 0)
        #from previous results, get the optimal vector length
        vecLength_ij = results['opt_vecLen'][plane_2D_idx,i,j,:]
        #use brute force to find the optimal vector length
        vecLength_test = np.linspace(min(vecLength_ij)*5, max(vecLength_ij)*15,\
            ngrid_bruteforce) #10, 3.5
        
        #fit an ellipse to the estimated comparison stimuli
        recover_fitEllipse_scaled[i,j,:,:], recover_fitEllipse_unscaled[i,j,:,:],\
            recover_rgb_comp_scaled[i,j,:,:], recover_rgb_contour_cov[i,j,:,:],\
            recover_rgb_comp_est[i,j,:,:], _ = \
                convert_Sig_2DisothresholdContour_oddity(\
                    rgb_ref_scaled_ij, sim['varying_RGBplane'], \
                    stim['grid_theta_xy'], vecLength_test,\
                    sim['pC_given_alpha_beta'],\
                    W_est)

#%%
fig, ax = plt.subplots(1, 1)
plt.rcParams['figure.dpi'] = 250 
ax.plot(iters, objhist)
fig.tight_layout()

Sigmas_init_grid = model.compute_Sigmas(
    model.compute_U(W_init, xgrid)
)
Sigmas_est_grid = model.compute_Sigmas(
    model.compute_U(W_est, xgrid)
)

#ground truth ellipses
gt_sigma = results['fitEllipse_scaled'][plane_2D_idx]
fig, ax = plt.subplots(1, 1, figsize = (5,3))
for i in range(NUM_GRID_PTS):
    for j in range(NUM_GRID_PTS):
        #define the color map, which is the RGB value of the reference stimulus
        cm = (xgrid[i, j,:] + 1)/2
        if plane_2D_idx != len(cm): cm = np.insert(cm, plane_2D_idx, 0.5);
        else: cm = np.append(cm, 0.5)   
        #ax.plot(gt_sigma[i,j,0,:] - (xref_temp[0,i,j]+1)/2 + xgrid[i, j,0],\
        #             gt_sigma[i,j,1,:] - (xref_temp[1,i,j]+1)/2 + xgrid[i, j,1], color = cm)
        i_lb = (j+i*NUM_GRID_PTS)*nSims
        i_ub = (j+i*NUM_GRID_PTS+1)*nSims
        if i == 0 and j == 0:
            ax.scatter(x1_jnp[i_lb:i_ub,0], x1_jnp[i_lb:i_ub,1], c = cm,s = 1,\
                       alpha = 0.2,label = 'Simulated data')
            viz.plot_ellipse(ax, xgrid[i, j,:], Sigmas_est_grid[i, j,:,:], \
                             color="k", alpha=.5, lw=1, label = 'Model-estimated cov matrix')
            ax.plot(recover_fitEllipse_unscaled[i,j,0,:], \
                recover_fitEllipse_unscaled[i,j,1,:], color = 'k',linewidth = 0.5,\
                linestyle = '--', label = 'Model predictions')
        else:
            ax.scatter(x1_jnp[i_lb:i_ub,0], x1_jnp[i_lb:i_ub,1], c = cm,s = 1,\
                       alpha = 0.2)
            viz.plot_ellipse(ax, xgrid[i, j,:], Sigmas_est_grid[i, j,:,:], \
                             color="k", alpha=.5, lw=1)
            ax.plot(recover_fitEllipse_unscaled[i,j,0,:], \
                recover_fitEllipse_unscaled[i,j,1,:], color = 'k',linewidth = 0.5,\
                linestyle = '--')            

tickmarks = np.unique(xgrid)
ax.set_xticks(tickmarks); ax.set_yticks(tickmarks)
#ax.set_xlim([-0.8, 0.8]); ax.set_ylim([-0.8, 0.8])
ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1])
ax.set_xlabel(plane_2D[0]); ax.set_ylabel(plane_2D[1])
ax.set_title(plane_2D)
plt.grid(alpha = 0.1)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = 8)
fig.tight_layout()
plt.show()
fig_outputDir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                        'ELPS_analysis/ModelFitting_FigFiles/'
fig_name = 'Fitted' + file_name[4:-4] + '_bandwidth' + str(opt_params['bandwidth'])
full_path = os.path.join(fig_outputDir,fig_name+'.png') #_expanded
fig.savefig(full_path)   

#%% save data
outputDir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                        'ELPS_analysis/ModelFitting_DataFiles/'
output_file = fig_name + '.pkl'
full_path = f"{outputDir}{output_file}"

variable_names = ['data','W_init','opt_params', 'W_est', 'iters', 'objhist',\
                  'Sigmas_init_grid', 'Sigmas_est_grid','recover_fitEllipse_scaled',\
                  'recover_fitEllipse_unscaled', 'recover_rgb_comp_scaled',\
                  'recover_rgb_contour_cov', 'recover_rgb_comp_est']
vars_dict = {}
for i in variable_names: vars_dict[i] = eval(i)

# Write the list of dictionaries to a file using pickle
with open(full_path, 'wb') as f:
    pickle.dump(vars_dict, f)

