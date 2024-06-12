#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 11:29:46 2024

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
sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/Python version')
fig_outputDir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                        'ELPS_analysis/ModelPerformance_FigFiles/'

#%% ------------------------------------------
# Load data simulated using CIELab
# ------------------------------------------
import os
import pickle
import numpy as np
from scipy.linalg import sqrtm

nSims     = 240
bandwidth = 0.005
jitter    = 0.1
path_str1 = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
            'ELPS_analysis/ModelFitting_DataFiles/'

file_name = 'Fitted_isothreshold_ellipsoids_sim'+str(nSims) +\
            'perCond_samplingNearContour_jitter'+str(jitter)+'_bandwidth' +\
            str(bandwidth) + '.pkl'
full_path = f"{path_str1}{file_name}"
os.chdir(path_str1)
#load data 
with open(full_path, 'rb') as f:
    # Load the object from the file
    data_load = pickle.load(f)
for key, value in data_load.items():
    locals()[key] = value
ref_size_dim1, ref_size_dim2, ref_size_dim3 = x1_raw.shape[0:3]
y_jnp, xref_jnp, x0_jnp, x1_jnp = data 

#%% load ground truths
path_str2 = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
        'ELPS_analysis/Simulation_DataFiles/'
file_name2 = 'Isothreshold_ellipsoid_CIELABderived.pkl'
full_path2 = f"{path_str2}{file_name2}"
os.chdir(path_str2)

#Here is what we do if we want to load the data
with open(full_path2, 'rb') as f:
    # Load the object from the file
    data_load2 = pickle.load(f)
_, stim3D, results3D = data_load2[0], data_load2[1], data_load2[2]

#%% 
def trim_W_est(W, basis_deg, method = 'cutoff_polyorder', cutoff_polyorder = 8,\
               cutoff_deg = 4, cutoff_extra_dim = 1):
    if method == 'cutoff_polyorder':
        W_trim = W.copy()
        W_trim = W_trim.at[basis_deg > cutoff_polyorder].set(0)
    elif method == 'cutoff_deg':
        W_trim = W.copy()
        max_deg = W.shape[0]
        W_trim = W_trim.at[list(range(cutoff_deg, max_deg)), :, :].set(0)
        W_trim = W_trim.at[:, list(range(cutoff_deg, max_deg)), :].set(0)
        W_trim = W_trim.at[:, :, list(range(cutoff_deg, max_deg))].set(0)
    elif method == 'cutoff_extra_dim':
        dim = W.shape[3]
        W_trim = W[:,:,:,:,:dim+cutoff_extra_dim]
    return W_trim

def plot_estimatedW_3D(poly_order, W, idx_slc = [], **kwargs):
    # Default parameters for ellipsoid fitting. Can be overridden by kwargs.
    pltP = {
        'marker_alpha':0.5,
        'marker_size':40,
        'marker_color': [[0.5,0.5,0.5],[0.27, 0.51, 0.70],[0,0.5,0]],
        'marker_edgecolor':[[1,1,1],[1,1,1],[1,1,1]],
        'xbds':[-0.04, 0.04],
        'saveFig':False,
        'figDir':'',
        'figName':'ModelEstimatedW'} 
    pltP.update(kwargs)
    if len(idx_slc) != 0: jitter = np.linspace(-0.2,0.2,len(idx_slc)+1); 
    else: jitter = [0]; idx_slc = [5];
    
    fig, ax = plt.subplots(1, 1, figsize=(6,4))
    ax.scatter(poly_order[:idx_slc[0],:idx_slc[0],:idx_slc[0],:,:] + jitter[0],\
                W[:idx_slc[0],:idx_slc[0],:idx_slc[0],:,:],\
                s = pltP['marker_size'], color = pltP['marker_color'][0],\
                edgecolor = pltP['marker_edgecolor'][0], \
                alpha = pltP['marker_alpha'], label = '<= '+str(idx_slc[0]-1))
    if len(idx_slc) != 0:
        for j in range(len(idx_slc)):
            idx_slc_j = idx_slc[j]
            ax.scatter(poly_order[idx_slc_j,:,:,:,:] + jitter[j+1],\
                        W[idx_slc_j,:,:,:,:], s = pltP['marker_size'],\
                        color = pltP['marker_color'][j+1],\
                        edgecolor = pltP['marker_edgecolor'][j+1],\
                        alpha = pltP['marker_alpha'],label = '='+str(idx_slc_j))
            ax.scatter(poly_order[:idx_slc_j,idx_slc_j,:,:,:] + jitter[j+1],\
                        W[:idx_slc_j,idx_slc_j,:,:,:],\
                        color = pltP['marker_color'][j+1],\
                        s = pltP['marker_size'],\
                        edgecolor = pltP['marker_edgecolor'][j+1],\
                        alpha = pltP['marker_alpha'])
            ax.scatter(poly_order[:idx_slc_j,:idx_slc_j,idx_slc_j,:,:] + jitter[j+1],\
                        W[:idx_slc_j,:idx_slc_j,idx_slc_j,:,:], \
                        color = pltP['marker_color'][j+1],\
                        s = pltP['marker_size'],\
                        edgecolor = pltP['marker_edgecolor'][j+1],\
                        alpha = pltP['marker_alpha'])
    ax.plot([0,np.max(poly_order)],[0,0],color = [0.5,0.5,0.5],\
            linestyle = '--', linewidth = 1)
    ax.set_yticks(np.linspace(pltP['xbds'][0], pltP['xbds'][-1],5))
    ax.set_ylim(pltP['xbds'])
    ax.grid(True, alpha=0.3)
    ax.legend(title = 'Polynomial degree')
    ax.set_xlabel('The order of 3D Chebyshev polynomial basis function')
    ax.set_ylabel('Model estimated weight')
    if pltP['saveFig'] and pltP['figDir'] != '':
        full_path = os.path.join(fig_outputDir, pltP['figName'])
        fig.savefig(full_path)    
        

#%%
basis_degrees = (
    jnp.arange(model.degree)[:, None, None] +
    jnp.arange(model.degree)[None, :, None] + 
    jnp.arange(model.degree)[None, None, :]
)

basis_degrees_rep = np.tile(basis_degrees,(model.num_dims,\
                                           model.num_dims+model.extra_dims,1,1,1))
basis_degrees_rep = np.transpose(basis_degrees_rep,(2,3,4,0,1))

#W_est_trim = trim_W_est(W_est, basis_degrees_rep)
W_est_trim = trim_W_est(W_est, basis_degrees_rep, method = 'cutoff_deg', cutoff_deg = 3)
#W_est_trim = trim_W_est(W_est, basis_degrees_rep, method = 'cutoff_extra_dim', \
#                        cutoff_extra_dim = 0)
Sigmas_est_grid = model.compute_Sigmas(model.compute_U(W_est_trim, xgrid))

#%%
#[]: keep all entries in W_est
#[4]: keep polynomial degree up to 4, but exclude 5
#[3,4]: keey polynomial degree up to 3/4, but exclude 4/5 respectively
idx_slc = [3,4] 
plot_estimatedW_3D(basis_degrees_rep, W_est, idx_slc , saveFig = False,\
                   figDir = fig_outputDir,\
                   figName = 'ModelEstimatedW_maxDeg'+ str(idx_slc)+\
                   '_fitted_isothreshold_ellipsoids_sim'+str(nSims) +\
                   'perCond_samplingNearContour_jitter'+str(jitter)+\
                    '_bandwidth' + str(bandwidth) +'.png')

#%% Model predictions
grid_theta            = stim3D['grid_theta'] #from 0 to 2*pi
n_theta               = len(grid_theta)
n_theta_finergrid     = 200
grid_phi              = stim3D['grid_phi'] #from 0 to pi
n_phi                 = len(grid_phi)
n_phi_finergrid       = 100
nSteps_bruteforce     = 200 #number of grids
scaler_x1             = 5
bds_scaler_gridsearch = [0.5, 3]
pC_threshold          = 0.78            

recover_fitEllipsoid_scaled, recover_fitEllipsoid_unscaled,\
    recover_rgb_comp_scaled, recover_rgb_contour_cov,\
    params_ellipsoids = model_predictions.convert_Sig_3DisothresholdContour_oddity_batch(\
        xref_raw, stim3D['grid_xyz'], pC_threshold, W_est_trim, model,\
        results3D['opt_vecLen'], scaler_x1 = scaler_x1,\
        ngrid_bruteforce=nSteps_bruteforce,\
        scaler_bds_bruteforce = bds_scaler_gridsearch,\
        bandwidth = opt_params['bandwidth'], opt_key = OPT_KEY,search_method='brute force')

#%%
gt_covMat   = np.full((ref_size_dim1, ref_size_dim2, ref_size_dim3, 3, 3), np.nan)
pred_covMat = np.full(gt_covMat.shape, np.nan)

# Loop through each reference color in the 3D space
for g1 in range(ref_size_dim1):
    for g2 in range(ref_size_dim2):
        for g3 in range(ref_size_dim3):
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

#%%
fig_outputDir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                        'ELPS_analysis/ModelFitting_FigFiles/Python_version/'+\
                        'cutoff_chebyshev_polynomial/'
#fig_name = 'Fitted' + file_name[4:-4] +'_trimmed_polyorder_cutoff' + str(poly_deg_cutoff)
fig_name = 'Fitted' + file_name[4:-4] +'_trimmed_polydeg_cutoff' + str(3)
model_predictions.plot_3D_modelPredictions_byWishart(xref_raw, x1_raw,\
        xref_jnp, x1_jnp, xgrid, gt_covMat, Sigmas_est_grid,\
        recover_fitEllipsoid_scaled, gt_slice_2d_ellipse, pred_slice_2d_ellipse,\
        saveFig = True, figDir = fig_outputDir, figName = fig_name)   

# make a gif
images = [img for img in os.listdir(fig_outputDir) if img.startswith(fig_name)]
images.sort()  # Sort the images by name (optional)

# Load images using imageio.v2 explicitly to avoid deprecation warnings
image_list = [imageio.imread(f"{fig_outputDir}/{img}") for img in images]

# Create a GIF
gif_name = fig_name + '.gif'
output_path = f"{fig_outputDir}{gif_name}" 
imageio.mimsave(output_path, image_list, fps=2)  
