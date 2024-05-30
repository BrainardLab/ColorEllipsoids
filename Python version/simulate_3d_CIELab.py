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
import warnings
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import Axes3D
import imageio.v2 as imageio

import sys
sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids')
from core import viz, utils, oddity_task, model_predictions, optim
from core.wishart_process import WishartProcessModel
sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/Python version')
from Simulate_probCorrectResp_3D import plot_3D_sampledComp
from Isothreshold_ellipsoids_CIELab import fit_3d_isothreshold_ellipsoid

#%% ------------------------------------------
# Load data simulated using CIELab
# ------------------------------------------
import os
import pickle
import numpy as np

file_name = 'Sims_isothreshold_ellipsoids_sim240perCond_samplingNearContour_jitter0.5.pkl'
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
#x1_raw is unscaled
data, x1_raw, xref_raw = model_predictions.organize_data(sim, scaler_x1,\
                                                    visualize_samples = False)
# unpackage data
ref_size_dim1, ref_size_dim2, ref_size_dim3 = x1_raw.shape[0:3]
y_jnp, xref_jnp, x0_jnp, x1_jnp = data 


#%% load ground truths
fixedRGB_val = [0.2,0.35,0.5,0.65,0.8]
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
param3D, stim3D, results3D, plt_specifics = data_load[0], data_load[1], data_load[2], data_load[3]

for fixedPlane, varyingPlanes in zip(['R','G','B'], ['GB','RB','RG']):
    for val in fixedRGB_val:
        plot_3D_sampledComp(stim3D['grid_ref']*2-1, results3D['fitEllipsoid_unscaled']*2-1,\
            x1_raw, fixedPlane, val*2-1, plt_specifics['nPhiEllipsoid'],\
            plt_specifics['nThetaEllipsoid'], slc_grid_ref_dim1 = [0,2,4],\
            slc_grid_ref_dim2 = [0,2,4], surf_alpha =  0.1,\
            samples_alpha = 0.1,scaled_neg12pos1 = True,\
            x_bds_symmetrical = 0.05,y_bds_symmetrical = 0.05,\
            z_bds_symmetrical = 0.05,title = varyingPlanes+' plane',\
            saveFig = True, figDir = path_str[0:-10] + 'FigFiles/',\
            figName = file_name + '_' + varyingPlanes + 'plane' +'_fixedVal'+str(val))

#%%
# -------------------------------
# Constants describing simulation
# -------------------------------
model = WishartProcessModel(
    5,     # Degree of the polynomial basis functions
    3,     # Number of stimulus dimensions
    1,     # Number of extra inner dimensions in `U`.
    3e-4,  # Scale parameter for prior on `W`.
    0.4,   # Geometric decay rate on `W`. 
    0,     # Diagonal term setting minimum variance for the ellipsoids.
)

NUM_GRID_PTS = 5      # Number of grid points over stimulus space.
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
    "learning_rate": 5e-2,
    "momentum": 0.2,
    "mc_samples": MC_SAMPLES,
    "bandwidth": BANDWIDTH,
}
W_est, iters, objhist = optim.optimize_posterior(
    W_init, data, model, OPT_KEY,
    opt_params,
    total_steps=2000,
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
output_file = 'Fitted'+file_name[4:-4]+'_bandwidth' + str(BANDWIDTH) + '.pkl'
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
nSteps_bruteforce     = 200 #ngrid_search 
bds_scaler_gridsearch = [0.5, 3]
pC_threshold          = 0.78            

recover_fitEllipsoid_scaled, recover_fitEllipsoid_unscaled,\
    recover_rgb_comp_scaled, recover_rgb_contour_cov,\
    params_ellipsoids = model_predictions.convert_Sig_3DisothresholdContour_oddity_batch(\
        xref_raw, stim3D['grid_xyz'], pC_threshold, W_est, model,\
        results3D['opt_vecLen'], scaler_x1 = scaler_x1,\
        ngrid_bruteforce=nSteps_bruteforce,\
        scaler_bds_bruteforce = bds_scaler_gridsearch,\
        bandwidth = opt_params['bandwidth'], opt_key = OPT_KEY)
        
#%%derive 2D slices
#initialization
gt_covMat             = np.full((ref_size_dim1, ref_size_dim2, ref_size_dim3, 3, 3), np.nan)
pred_covMat           = np.full(gt_covMat.shape, np.nan)
gt_slice_2d_ellipse   = np.full((ref_size_dim1, ref_size_dim2, ref_size_dim3, 3, 2,2), np.nan)
pred_slice_2d_ellipse = np.full(gt_slice_2d_ellipse.shape, np.nan)

for g1 in range(ref_size_dim1):
    for g2 in range(ref_size_dim2):
        for g3 in range(ref_size_dim3):
            #3D ellipsoid covariance matrix
            gt_covMat[g1,g2,g3] = model_predictions.ellParams_to_covMat(\
                            results3D['ellipsoidParams'][g1,g2,g3]['radii'],\
                            results3D['ellipsoidParams'][g1,g2,g3]['evecs'])
            #model predictions
            pred_covMat[g1,g2,g3] = model_predictions.ellParams_to_covMat(\
                            params_ellipsoids[g1][g2][g3]['radii'],\
                            params_ellipsoids[g1][g2][g3]['evecs'])
            for _idx, _idx_varying in zip([0,1,2], [[1, 2],[0, 2],[0, 1]]):
                idx = jnp.array(_idx_varying)
                #projection onto 2d plane

                # Invert the 3D covariance matrix to get the precision matrix
                gt_precision_matrix = np.linalg.inv(gt_covMat[g1,g2,g3])
                # Extract the 2x2 matrix corresponding to the x and y dimensions
                gt_precision_2d = gt_precision_matrix[idx][:,idx]
                # Invert the 2x2 precision matrix to get the covariance matrix of the 2D ellipse
                gt_slice_2d_ellipse[g1,g2,g3,_idx] = np.linalg.inv(gt_precision_2d)
                    
                #slices
                pred_precision_matrix = np.linalg.inv(pred_covMat[g1,g2,g3])
                pred_precision_2d = pred_precision_matrix[idx][:,idx]
                pred_slice_2d_ellipse[g1,g2,g3,_idx] = np.linalg.inv(pred_precision_2d)

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

#%% plotting 
for k, fixedRGB_val_scaled_k in zip(list(range(NUM_GRID_PTS)), fixedRGB_val_scaled):
    for _idx, fixedPlane, _idx_varying, varyingPlanes in zip([0,1,2],\
                                                             ['R','G','B'],\
                                                             [[1, 2],[0, 2],[0, 1]],\
                                                             ['GB','RB','RG']):
        fig, axes = plt.subplots(1, 2, figsize=(8,5))
        plt.rcParams['figure.dpi'] = 250
        idx = jnp.array(_idx_varying)
        
        #plot xref
        slc_idx_samples = np.where(np.abs(xref_jnp[:,_idx]-(fixedRGB_val_scaled_k)) < 1e-4)
        xref_jnp_slc_temp = xref_jnp[slc_idx_samples]
        xref_jnp_slc = xref_jnp_slc_temp[:,idx]
        
        x1_jnp_slc_temp = x1_jnp[slc_idx_samples]
        x1_jnp_slc = x1_jnp_slc_temp[:,idx]
        axes[0].scatter(x1_jnp_slc[:,0], x1_jnp_slc[:,1],\
                        s=1,c = (xref_jnp_slc_temp+1)/2, alpha=0.5)
                    
        for i in range(NUM_GRID_PTS):
            for j in range(NUM_GRID_PTS):
                if _idx == 0:   ii,jj,kk = k,i,j; 
                elif _idx == 1: ii,jj,kk = i,k,j; 
                elif _idx == 2: ii,jj,kk = i,j,k; 
                
                #lables
                if i == 0 and j == 0:
                    scatter_label = 'Simulated CIELab data (scaled up by ' +str(scaler_x1) +')' 
                    contour_3D_label = '3D ground truths\n(ellipsoids projecting on 2D)' 
                    contour_2D_label = '2D ground truths\nformed at the intersection\n'+\
                        'of 3D ellipsoids with '+fixedPlane +'='+str(fixedRGB_val_scaled_k) 
                    fits_label = 'model-estimated cov matrix'
                    gt_label = '3D ground truths\n(ellipsoids projecting on 2D)' 
                    pred3D_label = 'model-predicted elliposids'
                    pred2D_label = 'model-predicted ellipses'
                else:
                    scatter_label,contour_3D_label,contour_2D_label,fits_label,gt_label,\
                        pred3D_label, pred2D_label = None, None, None, None, None, None, None
                cm = (xref_raw[ii,jj,kk]+1)/2
                
                #scale up x1_raw
                x1_ij        = x1_raw[ii,jj,kk,idx]
                xref_ij      = xref_raw[ii,jj,kk,idx].reshape((2,1))
                
                #visualize                      
                viz.plot_ellipse(axes[0], xgrid[jj,ii,kk, idx],
                    (scaler_x1*2)**2*gt_covMat[ii,jj,kk][idx][:, idx],
                    color=np.array(cm), lw=2, label=contour_3D_label)
                    
                #visualize the fits
                viz.plot_ellipse(axes[0],  xgrid[jj,ii,kk, idx],
                    Sigmas_est_grid[jj,ii,kk][idx][:, idx], color="k",\
                    lw=1.5, linestyle = '-', alpha = 0.5, label = fits_label)      
                    
                #model-predicted ellipses (projections)
                recover_fitEllipse_scaled_ijk = recover_fitEllipsoid_scaled[ii,jj,kk][idx]
                polygon = Polygon(recover_fitEllipse_scaled_ijk.T, closed=True,fill=True, \
                                  color = np.array(cm), alpha = 0.3, label = pred3D_label)
                axes[0].add_patch(polygon)
                    
                #2d ground truth
                viz.plot_ellipse(axes[1], xgrid[jj,ii,kk,idx],\
                                 (scaler_x1*2)**2*gt_slice_2d_ellipse[ii,jj,kk,_idx],\
                                 color = 'r', linestyle = '--',lw=2, alpha = 0.5,\
                                     label = contour_2D_label)    
                
                viz.plot_ellipse(axes[1], xgrid[jj,ii,kk,idx],\
                                 (scaler_x1)**2*pred_slice_2d_ellipse[ii,jj,kk,_idx], \
                                lw=3, color ='g', alpha = 0.5, label = pred2D_label)
                    
        axes[0].grid(True, alpha=0.3); axes[1].grid(True, alpha=0.3)
        axes[0].set_xlim([-1,1]); axes[0].set_ylim([-1,1])
        axes[1].set_xlim([-1,1]); axes[1].set_ylim([-1,1]); 
        axes[0].set_xlabel(varyingPlanes[0]);
        axes[1].set_xlabel(varyingPlanes[0])
        axes[0].set_ylabel(varyingPlanes[1])
        axes[1].set_ylabel(varyingPlanes[1])
        ticks = np.unique(xgrid)
        axes[0].set_xticks(ticks); axes[0].set_yticks(ticks); 
        axes[1].set_xticks(ticks); axes[1].set_yticks(ticks); 
        axes[0].set_title(varyingPlanes + ' plane (Projections)')
        axes[1].set_title(varyingPlanes + ' plane (Slices)')
        axes[0].legend(loc='lower center',bbox_to_anchor=(0.5, -0.4),fontsize = 8.5)
        axes[1].legend(loc='lower center',bbox_to_anchor=(0.5, -0.4),fontsize = 8.5)
        fig.tight_layout()
        plt.show()
        fig_outputDir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                                'ELPS_analysis/ModelFitting_FigFiles/Python_version/'
        fig_name = 'Fitted' + file_name[4:-4] + '_slice_' +varyingPlanes+\
            'plane_fixedVal'+ str(fixedRGB_val_scaled_k)+'.png'
        full_path = os.path.join(fig_outputDir,fig_name)
        fig.savefig(full_path)    

#%% make a gif
outputDir_fig = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                        'ELPS_analysis/ModelFitting_FigFiles/Python_version/'
images = [img for img in os.listdir(outputDir_fig) if img.startswith(fig_name[:-30])]
images.sort()  # Sort the images by name (optional)

# Load images using imageio.v2 explicitly to avoid deprecation warnings
image_list = [imageio.imread(f"{outputDir_fig}/{img}") for img in images]

# Create a GIF
gif_name = fig_name[:-30] + '.gif'
output_path = f"{outputDir_fig}{gif_name}" 
imageio.mimsave(output_path, image_list, fps=2)  



