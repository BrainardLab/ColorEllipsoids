#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 21:01:22 2024

@author: fangfang
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
from core import model_predictions
from core.wishart_process import WishartProcessModel

#three variables we need to define for loading the data
plane_2D      = 'GB plane'
sim_jitter    = '0.1'
nSims         = 240 #number of simulations: 240 trials for each ref stimulus
BANDWIDTH     = 1e-3

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

"""
If we do not apply a scaler to x1
"""
scaler_x1  = 5
data, x1_raw, xref_raw = model_predictions.organize_data(sim, scaler_x1,\
                                                    visualize_samples = True,\
                                                    plane_2D = plane_2D)
ref_size_dim1, ref_size_dim2 = x1_raw.shape[0:2]
y_jnp, xref_jnp, x0_jnp, x1_jnp = data 

#%%
outputDir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                         'ELPS_analysis/ModelFitting_DataFiles/'  
output_file = 'Fitted' + file_sim[4:-4] + '_bandwidth' + str(BANDWIDTH) + '.pkl'
full_path = f"{outputDir}{output_file}"
with open(full_path, 'rb') as f:  data_load = pickle.load(f)
D = data_load

# variable_names = ['plane_2D', 'sim_jitter','nSims', 'data','model',\
#                   'NUM_GRID_PTS', 'MC_SAMPLES','BANDWIDTH', 'W_INIT_KEY',\
#                   'DATA_KEY', 'OPT_KEY', 'W_init','opt_params', 'W_est',\
#                   'iters', 'objhist','xgrid', 'Sigmas_init_grid',\
#                   'Sigmas_est_grid','recover_fitEllipse_scaled',\
#                   'recover_fitEllipse_unscaled', 'recover_rgb_comp_scaled',\
#                   'recover_rgb_contour_cov','params_ellipses','gt_sigma_scaled']

#%%
def simulate_zref_z0_z1(W, model, rgb_ref, rgb_comp, mc_samples, **kwargs):
    # Define default parameters with options for ellipse resolution and scaling
    params = {
        'bandwidth': 1e-3,
        'opt_key':jax.random.PRNGKey(444),
    }
    # Update default parameters with any additional keyword arguments provided
    params.update(kwargs)
    
    numDirPts = rgb_comp.shape[-1]
    Uref      = model.compute_U(W, rgb_ref)
    U0        = model.compute_U(W, rgb_ref)
    #initialize
    shape_init = (numDirPts, mc_samples, rgb_comp.shape[0])
    zref_all       = np.full(shape_init, np.nan)
    z0_all         = np.full(shape_init, np.nan)
    z1_all         = np.full(shape_init, np.nan)
    z0_to_zref_all = np.full(shape_init[0:2], np.nan)
    z1_to_zref_all = np.full(shape_init[0:2], np.nan)
    zdiff_all      = np.full(shape_init[0:2], np.nan)
    pChoosingX1    = np.full((numDirPts), np.nan)
    #for each chromatic direction
    for i in range(numDirPts):
        # Calculate RGB composition for current vector length
        rgb_comp_i = rgb_comp[:,i]
        U1 = model.compute_U(W, rgb_comp_i)
            
        # Generate random draws from isotropic, standard gaussians
        keys = jax.random.split(params['opt_key'], num=6)
        nnref = jax.random.normal(keys[0], shape=(mc_samples, U1.shape[1]))
        nn0 = jax.random.normal(keys[1], shape=(mc_samples, U1.shape[1]))
        nn1 = jax.random.normal(keys[2], shape=(mc_samples, U1.shape[1]))

        # Re-scale and translate the noisy samples to have the correct mean and
        # covariance. For example, zref ~ Normal(mref, Uref @ Uref.T).
        zref = nnref @ Uref.T + rgb_ref[None, :]
        z0 = nn0 @ U0.T + rgb_ref[None, :] 
        z1 = nn1 @ U1.T + rgb_comp_i[None, :] 
        
        zref_all[i], z0_all[i], z1_all[i] = zref, z0, z1
        
        # # Compute squared distance of each probe stimulus to reference
        z0_to_zref = jnp.sum((z0 - zref) ** 2, axis=1)
        z1_to_zref = jnp.sum((z1 - zref) ** 2, axis=1)
        zdiff = z0_to_zref - z1_to_zref
        
        z0_to_zref_all[i], z1_to_zref_all[i] = z0_to_zref, z1_to_zref
        zdiff_all[i] = zdiff
        
        # Approximate the cumulative distribution function for the trial
        pChoosingX1[i] = jnp.mean(jax.lax.logistic((0 - zdiff) / params['bandwidth']))
        
    return zref_all, z0_all, z1_all, z0_to_zref_all, z1_to_zref_all, zdiff_all, pChoosingX1
        
#%% visualize how sampled data looks like at discrete chromatic directions
slc_ref_pts1 = 0
slc_ref_pts2 = 0
rgb_ref_s  = jnp.array(xref_raw[:,slc_ref_pts1,slc_ref_pts2])
recover_rgb_comp_scaled_slc = D['recover_rgb_comp_scaled'][slc_ref_pts1, slc_ref_pts2]
recover_rgb_comp_unscaled_slc = (recover_rgb_comp_scaled_slc  -\
                               jnp.reshape(rgb_ref_s, (2, 1)))/scaler_x1 +\
                               jnp.reshape(rgb_ref_s, (2, 1))
numDirPts = stim['grid_theta_xy'].shape[1]
Zref_all, Z0_all, Z1_all, Z0_to_zref_all, Z1_to_zref_all, Zdiff_all, _ = \
    simulate_zref_z0_z1(D['W_est'], D['model'], rgb_ref_s,\
                        recover_rgb_comp_unscaled_slc, D['MC_SAMPLES'])

    
#%%                
def plot_sampled_zref_z1(Z1, Zref, rgb_ref, **kwargs):
    # Default parameters for ellipsoid fitting. Can be overridden by kwargs.
    pltP = {
        'legends':[],
        'saveFig':False,
        'figDir':'',
        'figName':'SanityCheck_sampled_zref_z1'} 
    pltP.update(kwargs)
    
    #define color map
    default_cmap = plt.get_cmap('tab20b')
    values       = np.linspace(0, 1, numDirPts)
    colors_array = default_cmap(values)
    #chromDir_deg = np.rad2deg(stim['grid_theta'])
    colors_ref   = [0.5,0.5,0.5]
    
    fig, ax1 = plt.subplots(1,1)
    plt.rcParams['figure.dpi'] = 250 
    z1_x_bds = 0
    for i in range(numDirPts):
        ax1.scatter(Z1[i,:,0],Z1[i,:,1],c = colors_array[i],s = 5,\
                    label = pltP['legends'][i])
        z1_x_bds = np.max([z1_x_bds, np.max(np.abs(Z1[i,:,0] - rgb_ref[0])),\
                           np.max(np.abs(Z1[i,:,1] - rgb_ref[1]))])
    ax1.scatter(Zref[0,:,0], Zref[0,:,1],c=colors_ref,s = 5)
    ax1.grid(True)
    ax1.set_aspect('equal')
    ax1.set_xlim(rgb_ref[0] + jnp.array([-z1_x_bds, z1_x_bds]))
    ax1.set_ylim(rgb_ref[1] + jnp.array([-z1_x_bds, z1_x_bds]))
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8,ncol = 1,\
               title='Chromatic \ndirection (deg)')
    plt.tight_layout()
    plt.show()
    if pltP['saveFig'] and pltP['figDir'] != '':
        full_path = os.path.join(pltP['figDir'],pltP['figName']+'.png')
        fig.savefig(full_path)
    return z1_x_bds

def plot_EuclideanDist_hist(Z0_to_zref, Z1_to_zref,bin_edges, **kwargs):
    # Default parameters for ellipsoid fitting. Can be overridden by kwargs.
    pltP = {
        'legends':[],
        'saveFig':False,
        'figDir':'',
        'figName':'SanityCheck_sampled_zref_z1'} 
    pltP.update(kwargs)
    
    #define color map
    default_cmap = plt.get_cmap('tab20b')
    values       = np.linspace(0, 1, numDirPts)
    colors_array = default_cmap(values)
    colors_ref   = [0.5,0.5,0.5]
    
    fig, ax = plt.subplots(4,4,figsize = (10,8))
    plt.rcParams['figure.dpi'] = 250 
    bin_ub = bin_edges[-1] #0.01
    for i in range(numDirPts):
        ax[i//4, i%4].set_xlim([0, np.around(bin_ub,2)])
        ax[i//4, i%4].hist(Z0_to_zref[i], bins = bin_edges,\
                            color=colors_ref,alpha = 0.7)
        ax[i//4, i%4].hist(Z1_to_zref[i], bins = bin_edges,\
                            color=colors_array[i],alpha = 0.8,\
                            label = 'cDir = '+pltP['legends'][i]+' deg')
        if i%4 !=0: ax[i//4, i%4].set_yticks([])
        if i//4 != 3: ax[i//4, i%4].set_xticks([]); 
        else: ax[i//4, i%4].set_xticks([0, np.around(bin_ub,2)]); 
        ax[i//4, i%4].tick_params(axis='x', labelsize=12)
        ax[i//4, i%4].tick_params(axis='y', labelsize=12)
        lgd = ax[i//4, i%4].legend()
        lgd.set_frame_on(False)
    fig.suptitle(r'$||z_0 - z_{ref}||^2 vs. ||z_1 - z_{ref}||^2$', fontsize = 14)
    plt.tight_layout()
    plt.show()
    if pltP['saveFig'] and pltP['figDir'] != '':
        full_path2 = os.path.join(pltP['figDir'],pltP['figName']+'.png')
        fig.savefig(full_path2)

def plot_EuclieanDist_diff_hist(Z_diff, bin_edges, **kwargs):
    # Default parameters for ellipsoid fitting. Can be overridden by kwargs.
    pltP = {
        'legends':[],
        'saveFig':False,
        'figDir':'',
        'figName':'SanityCheck_sampled_zref_z1'} 
    pltP.update(kwargs)
    
    #define color map
    default_cmap = plt.get_cmap('tab20b')
    values       = np.linspace(0, 1, numDirPts)
    colors_array = default_cmap(values)
    
    fig, ax = plt.subplots(4,4,figsize = (10,8))
    plt.rcParams['figure.dpi'] = 250 
    #bin_edges= np.linspace(-bin_ub,bin_ub/2,30)
    for i in range(numDirPts):
        ax[i//4, i%4].set_xlim(np.around([bin_edges[0], bin_edges[-1]],2))
        ax[i//4, i%4].set_ylim([0,Z_diff.shape[-1]/2])
        ax[i//4, i%4].plot([0,0],[0,Z_diff.shape[-1]/2],c = 'k',linestyle = '--',linewidth=0.5)
        ax[i//4, i%4].hist(Z_diff[i], bins = bin_edges,\
                            color=colors_array[i],alpha = 0.7,\
                            label = 'cDir = '+pltP['legends'][i]+' deg')
        if i%4 !=0: ax[i//4, i%4].set_yticks([])
        if i//4 != 3: ax[i//4, i%4].set_xticks([]); 
        else: ax[i//4, i%4].set_xticks(np.around([bin_edges[0], bin_edges[-1]],2)); 
        ax[i//4, i%4].tick_params(axis='x', labelsize=12)
        ax[i//4, i%4].tick_params(axis='y', labelsize=12)
        lgd = ax[i//4, i%4].legend()
        lgd.set_frame_on(False)
    fig.suptitle(r'$||z_0 - z_{ref}||^2 - ||z_1 - z_{ref}||^2 $', fontsize = 14)
    plt.tight_layout()
    plt.show()
    if pltP['saveFig'] and pltP['figDir'] != '':
        full_path3 = os.path.join(pltP['figDir'],pltP['figName']+'.png')
        fig.savefig(full_path3)



#%%
fig_outputDir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                'ELPS_analysis/SanityChecks_FigFiles'
x_str, y_str = str(np.round(rgb_ref_s[0], 2)), str(np.round(rgb_ref_s[1], 2))                
figName1 = 'sampled_zref_z1_'+plane_2D+'_x'+x_str[0:4] +\
                '_y' + y_str[0:4]
plt_bds = plot_sampled_zref_z1(Z1_all, Zref_all, rgb_ref_s, saveFig = True,\
                     legends = [str(items) for items in np.rad2deg(stim['grid_theta'])], \
                     figDir = fig_outputDir, figName = figName1)

hist_bin_edges = np.linspace(0,plt_bds/8, 30)
plot_EuclideanDist_hist(Z0_to_zref_all, Z1_to_zref_all,hist_bin_edges,saveFig = True,\
                        legends = [str(items) for items in np.rad2deg(stim['grid_theta'])],\
                        figDir = fig_outputDir, figName = figName1 + '_hist')

hist_diff_bin_edges  = np.linspace(-plt_bds/8,plt_bds/16, 30)
plot_EuclieanDist_diff_hist(Zdiff_all, hist_diff_bin_edges, saveFig = True,\
                            legends = [str(items) for items in np.rad2deg(stim['grid_theta'])],\
                            figDir = fig_outputDir, figName = figName1 + '_diff_hist')

#%% create a grid of x1
scaler_x1_vec = np.arange(0.1,2,0.1)
rgb_comp_varying = np.full((len(scaler_x1_vec), recover_rgb_comp_scaled_slc.shape[0],\
                            recover_rgb_comp_scaled_slc.shape[1]),np.nan)
for s in range(len(scaler_x1_vec)):
    rgb_comp_varying[s] = (recover_rgb_comp_unscaled_slc - jnp.reshape(rgb_ref_s, (2, 1)))*\
                    scaler_x1_vec[s] + jnp.reshape(rgb_ref_s, (2, 1))
rgb_comp_varying_all = np.stack((rgb_comp_varying[:,0,:].ravel(),\
                                rgb_comp_varying[:,1,:].ravel()), axis=0)
_, _, _, _, _, _, pChoosingX1_all = \
    simulate_zref_z0_z1(D['W_est'], D['model'], rgb_ref_s,\
                        rgb_comp_varying_all, D['MC_SAMPLES'])
    
fig, ax = plt.subplots(1,1)


