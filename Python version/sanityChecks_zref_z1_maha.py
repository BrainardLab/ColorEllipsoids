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

#%%
def simulate_zref_z0_z1(W, model, rgb_ref, rgb_comp, mc_samples, **kwargs):
    """
    Simulate stimuli in perceptual space for a reference and comparison stimuli,
    and calculate the probability of choosing one comparison over another based on
    their Euclidean distances from the reference stimulus.
    
    Note: this function is very similar to a function in Alex's code (oddity_task),
    but that function doesn't return sampled zref, z0 and z1. This script relies 
    on z's for computation, so that's why I made this function.
    
    Parameters:
        W (numpy.ndarray): Weight matrix for the model.
        model (object): Model object with methods to compute transformations.
        rgb_ref (numpy.ndarray; size: 2, ): 
            RGB values for the reference stimulus.
        rgb_comp (numpy.ndarray; size: 2, N): 
            RGB values for the comparison stimuli. N: #chromatic directions
        mc_samples (int): Number of Monte Carlo samples to draw.
        **kwargs: Optional keyword arguments to override default model parameters.
    
    Returns:
        tuple: Contains arrays of simulated reference, comparison stimuli (z0, z1),
               distances to reference, difference in distances, and 
               probability of choosing x1.
"""

    # Define default parameters
    params = {
        'bandwidth': 1e-3,
        'opt_key':jax.random.PRNGKey(444),
    }
    # Update default parameters with any additional keyword arguments provided
    params.update(kwargs)
    
    # Determine the number of direction points based on the last dimension of rgb_comp
    numDirPts = rgb_comp.shape[-1]
    
    # Compute U (weighted sum of basis functions) for the reference stimulus
    Uref      = model.compute_U(W, rgb_ref)
    U0        = model.compute_U(W, rgb_ref) # Since reference and comparison z0 are the same
    
    #initialize
    shape_init = (numDirPts, mc_samples, rgb_comp.shape[0])
    zref_all       = np.full(shape_init, np.nan)
    z0_all         = np.full(shape_init, np.nan)
    z1_all         = np.full(shape_init, np.nan)
    z0_to_zref_all = np.full(shape_init[0:2], np.nan)
    z1_to_zref_all = np.full(shape_init[0:2], np.nan)
    zdiff_all      = np.full(shape_init[0:2], np.nan)
    pChoosingX1    = np.full((numDirPts), np.nan)
    
    # Iterate over each chromatic direction
    for i in range(numDirPts):
        # Current RGB composition for the comparison
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
        #save
        zref_all[i], z0_all[i], z1_all[i] = zref, z0, z1
        
        # Compute squared distance of each probe stimulus to reference
        # z0_to_zref = jnp.sum((z0 - zref) ** 2, axis=1)
        # z1_to_zref = jnp.sum((z1 - zref) ** 2, axis=1)
        # Reshape U, flattening all but the last two dimensions.
        Uref_z = model.compute_U(W, zref)
        U0_z = model.compute_U(W, z0)
        U1_z = model.compute_U(W, z1)

        # Compute covariance matrix
        Sig0 = model.compute_Sigmas(U0_z)
        Sig1 = model.compute_Sigmas(U1_z)
        Sigref = model.compute_Sigmas(Uref_z)

        # Calculate the inverse of the covariance matrices
        inv_U0_Uref = jnp.linalg.inv(Sigref + Sig0)
        inv_U1_Uref = jnp.linalg.inv(Sigref + Sig1)

        # Compute squared Mahalanobis distance of each probe stimulus to reference
        z0_to_zref_temp = z0 - zref
        z1_to_zref_temp = z1 - zref
            
        einsum_prod_z0 = jnp.einsum('ij,ijk->ik', z0_to_zref_temp, inv_U0_Uref)
        einsum_prod_z1 = jnp.einsum('ij,ijk->ik', z1_to_zref_temp, inv_U1_Uref)
        
        z0_to_zref = jnp.sqrt(jnp.einsum('ik,ki->i', einsum_prod_z0, z0_to_zref_temp.T))
        z1_to_zref = jnp.sqrt(jnp.einsum('ik,ki->i', einsum_prod_z1, z1_to_zref_temp.T))
        
        zdiff = z0_to_zref - z1_to_zref
        #save
        z0_to_zref_all[i], z1_to_zref_all[i] = z0_to_zref, z1_to_zref
        zdiff_all[i] = zdiff
        
        # Approximate the cumulative distribution function for the trial
        pChoosingX1[i] = jnp.mean(jax.lax.logistic((0 - zdiff) / params['bandwidth']))
        
    return zref_all, z0_all, z1_all, z0_to_zref_all, z1_to_zref_all, zdiff_all, pChoosingX1

#%%
def plot_sampled_zref_z1(Z1, Zref, rgb_ref, **kwargs):
    """
    Plots sampled Z1 data for various chromatic directions and one set of 
    sampled Zref. Highlights the dispersion of Z1 around the reference stimulus 
    Zref.
    
    Parameters:
        Z1 (numpy.ndarray; size: N x M x 2): 
            The sampled comparison stimuli across different chromatic directions.
            N: # chromatic directions
            M: #MC samples
            2 dimensions
        Zref (numpy.ndarray):
            The sampled reference stimulus (with the same size as Z1).
        rgb_ref (numpy.ndarray; 2,): RGB values for the reference stimulus.
        **kwargs: Additional parameters for plot customization including saving options.
    
    Returns:
        float: The maximum bound used to set the x and y limits on the plot, centered on rgb_ref.
    """

    # Default plot parameters; can be updated with kwargs to customize plot behavior
    pltP = {
        'legends':[],   # List of legends for each chromatic direction
        'saveFig':False,# Flag to save the figure
        'figDir':'',    # Directory to save the figure
        'figName':'SanityCheck_sampled_zref_z1'}  # Default figure name
    pltP.update(kwargs)  # Update plot parameters with any additional keyword arguments
    
    # Define color map and color assignments for plots
    default_cmap = plt.get_cmap('tab20b') 
    numDirPts    = Z1.shape[0]
    values       = np.linspace(0, 1, numDirPts)
    colors_array = default_cmap(values)
    colors_ref   = [0.5,0.5,0.5]
    
    # Initialize plot
    fig, ax1 = plt.subplots(1,1)
    plt.rcParams['figure.dpi'] = 250 
    z1_x_bds = 0 # Initialize the boundary for plot limits
    # Plot each set of Z1 samples
    for i in range(numDirPts):
        ax1.scatter(Z1[i,:,0],Z1[i,:,1],c = colors_array[i],s = 5,\
                    label = pltP['legends'][i])
        # Update the maximum boundary for plot limits based on Z1 data
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
    """
    Plots histograms of the squared Euclidean distances from Z0 and Z1 to Zref
    for various chromatic directions in a multi-panel figure.

    Parameters:
        Z0_to_zref (numpy.ndarray; size: N x M): 
            Array of squared distances from Z0 to Zref for each direction.
            N: #chromatic directions
            M: #MC trials
        Z1_to_zref (numpy.ndarray size: N x M): 
            Array of squared distances from Z1 to Zref for each direction.
            (same size as Z0_to_zref)
        bin_edges (numpy.ndarray): 
            Array of bin edges for the histograms.
        **kwargs: Optional keyword arguments for plot customization such as 
            legends and saving.

    Returns:
        None: The function plots the histograms and may save the figure based 
        on input parameters.
    """
    
    # Set default plot parameters, update with any provided keyword arguments
    pltP = {
        'legends':[],
        'saveFig':False,
        'figDir':'',
        'figName':'SanityCheck_sampled_zref_z1'} 
    pltP.update(kwargs)
    
    #define color map
    default_cmap = plt.get_cmap('tab20b')
    numDirPts    = Z0_to_zref.shape[0]
    values       = np.linspace(0, 1, numDirPts)
    colors_array = default_cmap(values)
    colors_ref   = [0.5,0.5,0.5]
    
    # Setup figure with multiple subplots
    fig, ax = plt.subplots(4,4,figsize = (10,8))
    plt.rcParams['figure.dpi'] = 250 
    # Upper bound for histogram x-axis
    bin_ub = bin_edges[-1] #0.01
    
    # Plot histograms for each direction
    for i in range(numDirPts):
        # Configure axis for the current subplot
        ax[i//4, i%4].set_xlim([0, np.around(bin_ub,2)])
        # Plot histogram for Z0 to Zref distances
        ax[i//4, i%4].hist(Z0_to_zref[i], bins = bin_edges,\
                            color=colors_ref,alpha = 0.7)
        # Plot histogram for Z1 to Zref distances
        ax[i//4, i%4].hist(Z1_to_zref[i], bins = bin_edges,\
                            color=colors_array[i],alpha = 0.8,\
                            label = 'cDir = '+pltP['legends'][i]+' deg')
         # Adjust ticks and labels for clarity and aesthetics
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
    """
    Plots histograms of the difference between squared Euclidean distances from 
    Z0 and Z1 to Zref for various chromatic directions in a multi-panel figure.

    Parameters:
        Z_diff (numpy.ndarray; size: N x M): 
            Array of squared distances from Z0 to Zref for each direction.
            N: #chromatic directions
            M: #MC trials
        bin_edges (numpy.ndarray): 
            Array of bin edges for the histograms.
        **kwargs: Optional keyword arguments for plot customization such as 
            legends and saving.

    Returns:
        None: The function plots the histograms and may save the figure based 
        on input parameters.
    """
    pltP = {
        'legends':[],
        'saveFig':False,
        'figDir':'',
        'figName':'SanityCheck_sampled_zref_z1'} 
    pltP.update(kwargs)
    
    #define color map
    default_cmap = plt.get_cmap('tab20b')
    numDirPts    = Z_diff.shape[0]
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

#%% three variables we need to define for loading the data
def visualize_samplesZ_givenW(plane_2D, sim_jitter, nSims, BANDWIDTH,
                              idx1, idx2, **kwargs):
    params = {
        'visualize_samples_allPlanes': False,
        'scaler_x1':5,
        'saveFig':False,
        'figDir':'',
        } 
    params.update(kwargs)
    
    #%% -----------------------------------------------------------
    # Load data simulated using CIELab and organize data
    # -----------------------------------------------------------
    #file 1
    path_str      = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                    'ELPS_analysis/Simulation_DataFiles/'
    os.chdir(path_str)
    file_CIE      = 'Isothreshold_contour_CIELABderived.pkl'
    full_path1    = f"{path_str}{file_CIE}"
    with open(full_path1, 'rb') as f: data_load1 = pickle.load(f)
    stim          = data_load1[1]
    results       = data_load1[2]
    gt_rgb_comp_scaled = results['rgb_comp_contour_scaled'][0] * 2 - 1
    
    #file 2
    file_sim      = 'Sims_isothreshold_'+plane_2D+'_sim'+str(nSims)+'perCond_'+\
                    'samplingNearContour_jitter'+sim_jitter+'.pkl'
    full_path2     = f"{path_str}{file_sim}"   
    with open(full_path2, 'rb') as f:  data_load2 = pickle.load(f)
    sim           = data_load2[0]
    scaler_x1     = params['scaler_x1']
    _, x1_raw, xref_raw = model_predictions.organize_data(sim, scaler_x1,\
        visualize_samples = params['visualize_samples_allPlanes'],\
        plane_2D = plane_2D)
    #get the number of ref points for the two dimensions
    ref_size_dim1, ref_size_dim2 = x1_raw.shape[0:2]
    
    #file 3: model fits
    path_str2 = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                             'ELPS_analysis/ModelFitting_DataFiles/'  
    file_fits = 'Fitted' + file_sim[4:-4] + '_bandwidth' + str(BANDWIDTH) + '_maha.pkl' 
    full_path3 = f"{path_str2}{file_fits}"
    with open(full_path3, 'rb') as f:  data_load3 = pickle.load(f)
    D = data_load3
            
    #%% 
    rgb_ref_s  = jnp.array(xref_raw[:,idx1,idx2])
    #recover_rgb_comp_scaled_slc = D['recover_rgb_comp_scaled'][idx1, idx2]
    recover_rgb_comp_scaled_slc = gt_rgb_comp_scaled[idx1, idx2]
    # Retrieve and adjust comparison RGB values based on model recovery and scaling.
    recover_rgb_comp_unscaled_slc = (recover_rgb_comp_scaled_slc  -\
                                   jnp.reshape(rgb_ref_s, (2, 1)))/scaler_x1 +\
                                   jnp.reshape(rgb_ref_s, (2, 1))
    # Simulate Zref, Z0, Z1, and distances using the estimated parameters from the model.
    Zref_all, Z0_all, Z1_all, Z0_to_zref_all, Z1_to_zref_all, Zdiff_all, _ = \
        simulate_zref_z0_z1(D['W_est'], D['model'], rgb_ref_s,\
                            recover_rgb_comp_unscaled_slc, D['MC_SAMPLES'])
    
    #%% visualization
    # Construct strings for RGB reference values for use in filenames.
    x_str, y_str = str(np.round(rgb_ref_s[0], 2)), str(np.round(rgb_ref_s[1], 2))    
    # Define a filename for saving plot output.            
    figName1 = 'sampled_zref_z1_'+plane_2D+'_x'+x_str[0:4] +'_y' + y_str[0:4]
    
    # Plot sampled data using custom plotting function.
    plt_bds = plot_sampled_zref_z1(Z1_all, Zref_all, rgb_ref_s, \
                                   saveFig = params['saveFig'],\
                legends = [str(items) for items in np.rad2deg(stim['grid_theta'])], \
                figDir = params['figDir'], figName = figName1)
    
    # Define histogram bin edges
    hist_bin_edges = np.linspace(0,15, 30)#np.linspace(0,plt_bds/8, 30)
    plot_EuclideanDist_hist(Z0_to_zref_all, Z1_to_zref_all,hist_bin_edges,\
                            saveFig = params['saveFig'],\
                legends = [str(items) for items in np.rad2deg(stim['grid_theta'])],\
                figDir = params['figDir'], figName = figName1 + '_hist')
    
    hist_diff_bin_edges  = np.linspace(-15,10, 30)#np.linspace(-plt_bds/8,plt_bds/16, 30)
    plot_EuclieanDist_diff_hist(Zdiff_all, hist_diff_bin_edges, \
                                saveFig = params['saveFig'],\
                legends = [str(items) for items in np.rad2deg(stim['grid_theta'])],\
                figDir = params['figDir'], figName = figName1 + '_diff_hist')
    
#%%
def main():
    plane_2D     = 'GB plane'
    sim_jitter   = '0.1'
    nSims        = 240 #number of simulations: 240 trials for each ref stimulus
    BANDWIDTH    = 0.1
    slc_ref_pts1 = [0,4]
    slc_ref_pts2 = [4,0]
    
    fig_outputDir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                    'ELPS_analysis/SanityChecks_FigFiles'
                        
    for i in range(len(slc_ref_pts1)):
        visualize_samplesZ_givenW(plane_2D, sim_jitter, nSims, BANDWIDTH,\
                                  slc_ref_pts1[i], slc_ref_pts2[i],\
                                  saveFig = False, figDir = fig_outputDir)
        #'visualize_samples_allPlanes': make it true if we want to first visualize
        #which plane we are smapling
        
if __name__ == "__main__":
    main()    
    

