#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 20:43:18 2024

@author: fangfang
"""

import jax
jax.config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
import os
import pickle
import sys
import numpy as np
from scipy.io import loadmat

sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/Python version")
import simulations_CIELab 
from sanityChecks_zref_z1 import simulate_zref_z0_z1

#%% plotting function
def plot_contour_pChoosingX1(ref_grids, ref_grids_X, 
                             ref_grids_Y, P_X1, **kwargs):
    # Default parameters for ellipsoid fitting. Can be overridden by kwargs.
    pltP = {
        'cmap_filled':'twilight',
        'cmap_contour':'white',
        'cmap_nLevels': 15,
        'title':"",
        'saveFig':False,
        'figDir':'',
        'figName':'contour_pChoosingX1'} 
    pltP.update(kwargs)
    
    num_ref_pts1, num_ref_pts2 = ref_grids.shape[1], ref_grids.shape[2]
    fig, ax = plt.subplots(num_ref_pts1, num_ref_pts2, figsize = (12,12))
    contour_levels = np.linspace(0.45,1, pltP['cmap_nLevels'])
    
    for j in range(num_ref_pts2):
        jj = num_ref_pts2 -1 - j 
        for i in range(num_ref_pts1):
            contour_filled = ax[j,i].contourf(ref_grids_X[i,jj],ref_grids_Y[i,jj],\
                                          P_X1[i,jj], levels = contour_levels,\
                                          cmap = pltP['cmap_filled']) 
            contour_line = ax[j,i].contour(ref_grids_X[i,jj],ref_grids_Y[i,jj],\
                                       P_X1[i,jj], levels=[0.78], \
                                       colors=[pltP['cmap_contour']],\
                                       linewidths=2)
            ax[j,i].clabel(contour_line, fmt={0.78: '78%'})
            ax[j,i].set_aspect('equal')  
            if i != 0: ax[j,i].set_yticks([])
            else:
                # Setting tick font size for x and y axes
                ax[j,i].set_yticks(ref_grids[1][jj,i]*2-1+ np.array([-0.2, 0, 0.2])) 
                ax[j,i].tick_params(axis='y', labelsize=16)  # Set fontsize for x-axis ticks
            if j != num_ref_pts2-1: ax[j,i].set_xticks([])
            else:
                ax[j,i].set_xticks(ref_grids[0][jj,i]*2-1+ np.array([-0.2, 0, 0.2])) 
                ax[j,i].tick_params(axis='x', labelsize=16)  # Set fontsize for y-axis ticks
    fig.suptitle(pltP['title'], fontsize = 20)
    fig.subplots_adjust(right=1.1) 
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])  # x, y, width, height
    cbar = fig.colorbar(contour_filled, cax=cbar_ax)
    tick_values = np.linspace(0.5, 1, num=5)
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels([f"{v:.2f}" for v in tick_values])
    fig.supxlabel(plane_2D[0], fontsize = 20)
    fig.supylabel(plane_2D[1], fontsize = 20)
    plt.tight_layout(rect=[0, 0, 0.95, 0.95]) 
    plt.show()
    if pltP['saveFig'] and pltP['figDir'] != '':
        full_path2 = os.path.join(pltP['figDir'],pltP['figName']+'.png')
        fig.savefig(full_path2)

#%%
def compute_probC1_contour(plane_2D, sim_jitter, nSims, bandwidth, scaler_x1,
                             nGrid_x, nGrid_y, width):    
    # -----------------------------------------------------------
    # Load data simulated using CIELab and organize data
    # -----------------------------------------------------------
    #file 1
    path_str      = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                    'ELPS_analysis/Simulation_DataFiles/'
    os.chdir(path_str)
    plane_2D_dict = {'GB plane': 0, 'RB plane': 1, 'RG plane': 2}
    plane_2D_idx  = plane_2D_dict[plane_2D]
    file_sim      = 'Sims_isothreshold_'+plane_2D+'_sim'+str(nSims)+'perCond_'+\
                    'samplingNearContour_jitter'+sim_jitter+'.pkl'
    full_path     = f"{path_str}{file_sim}"   
    with open(full_path, 'rb') as f:  data_load = pickle.load(f)
    sim = data_load[0]
    
    #file 2: model fits
    path_str2     = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                             'ELPS_analysis/ModelFitting_DataFiles/'  
    file_fits     = 'Fitted' + file_sim[4:-4] + '_bandwidth' + str(bandwidth) + '.pkl'
    full_path2    = f"{path_str2}{file_fits}"
    with open(full_path2, 'rb') as f:  data_load2 = pickle.load(f)
    D = data_load2
    
    #load monitor info
    B_monitor_mat  = loadmat("/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/"+\
                    "FilesFromPsychtoolbox/B_monitor.mat")
    B_monitor      = B_monitor_mat['B_monitor'] #size: (61, 3)
    
    # -------------------------------------------------------------
    # probability contour map predicted by the Wishart process model
    # ---------------------------------------------------------------
    # create a grid of x1 
    nRef_x = sim['ref_points'].shape[1] #the number of ref stimuli along the 1st dim
    nRef_y = sim['ref_points'].shape[2] #the number of ref stimuli along the 2nd dim
    
    #initialize
    pChoosingX1_grid_allRef = np.full((nRef_y,nRef_x,nGrid_y, nGrid_x), np.nan)
    rgb_comp_dim1_grid      = np.full(pChoosingX1_grid_allRef.shape,np.nan)
    rgb_comp_dim2_grid      = np.full(pChoosingX1_grid_allRef.shape,np.nan)
    rgb_comp_varying_grid   = np.full((nRef_y,nRef_x,2,nGrid_x*nGrid_y),np.nan)
    
    #for each ref stimulus
    for j in range(nRef_y):
        for i in range(nRef_x):
            #scale the reference stimulus to fit in the [-1, 1] cube
            rgb_ref_ij = (sim['ref_points'][:,i,j]*2)-1
            #select the two dimensions that we are varying
            rgb_ref_ij_2D = rgb_ref_ij[sim['varying_RGBplane']]
            #create a fine 2D grid within [rgb_ref_ij_2D[0]-width, rgb_ref_ij_2D[0]+width]
            rgb_comp_dim1_vec = np.linspace(rgb_ref_ij_2D[0]-width, \
                                            rgb_ref_ij_2D[0]+width, nGrid_x)
            rgb_comp_dim2_vec = np.linspace(rgb_ref_ij_2D[1]-width, \
                                            rgb_ref_ij_2D[1]+width, nGrid_y)
            rgb_comp_dim1_grid[j,i], rgb_comp_dim2_grid[j,i] = \
                np.meshgrid(rgb_comp_dim1_vec, rgb_comp_dim2_vec) 
                
            #re-org the data so that rgb_comp_varying_grid has a shape of 
            #(2, N) where N = nGrid_x x nGrid_y
            rgb_comp_varying_grid[j,i] = np.stack((rgb_comp_dim1_grid[j,i].ravel(), \
                                              rgb_comp_dim2_grid[j,i].ravel()), \
                                              axis = 0)
            #compute the probability of choosing X1 as the odd stimulus
            #given a selected ref stimulus and many different comparison stimuli
            _, _, _, _, _, _, pChoosingX1_grid = simulate_zref_z0_z1(\
                                    D['W_est'], D['model'], rgb_ref_ij_2D,\
                                    rgb_comp_varying_grid[j,i], D['MC_SAMPLES'])
            #reshape pChoosingX1_grid to have a size of (nGrid_y, nGrid_x) and save
            pChoosingX1_grid_allRef[j,i] = np.reshape(pChoosingX1_grid,(nGrid_y,nGrid_x))
    
    # -------------------------------------------------------------
    # probability contour map predicted by the Wishart process model
    # ---------------------------------------------------------------
    #background RGB
    background_RGB = [0.5,0.5,0.5]
    #Weibull psychometric function computed with pre-specified alpha and beta
    WeibullFunc = lambda x: (1 - 0.5*np.exp(- (x/sim['alpha'])** sim['beta']))
    
    #initialize
    pC_weibull_allRef = np.full((nRef_y,nRef_x,nGrid_y, nGrid_x), np.nan)
    deltaE_allRef = np.full((nRef_y,nRef_x,nGrid_y, nGrid_x), np.nan)
    
    #for each ref stimulus
    for i in range(nRef_y):
        for j in range(nRef_x):
            #ref stimulus in RGB unit [0, 1]
            rgb_ref_ij_unscaled = sim['ref_points'][:,j,i]
            #convert it to lab
            lab_ref_ij,_,_ = simulations_CIELab.convert_rgb_lab(B_monitor,\
                background_RGB, rgb_ref_ij_unscaled)
    
            #initialize the comparison stimulus
            rgb_comp_ij_unscaled = np.full((3,nGrid_x*nGrid_y),np.nan)
            #unscale the comparison stimulus so that it fits within [0, 1] 
            #instead of [-1. 1]
            rgb_comp_ij_unscaled[sim['varying_RGBplane']] = \
                (rgb_comp_varying_grid[i,j] +1)/2
            #there is one plane that has fixed RGB value; we need to put it in 
            #for converting to LAB values
            rgb_comp_ij_unscaled[plane_2D_idx] = 0.5
                
            #initialize
            deltaE = np.full((nGrid_x*nGrid_y), np.nan)
            pC_weibull = np.full((nGrid_x*nGrid_y), np.nan)
            for k in range(nGrid_x*nGrid_y):
                #unscale again so that the comparison stimuli are around the 
                #threshold contour instead of scaler_x1 times larger for the 
                #purpose of model fitting
                rgb_comp_ij_unscaled_k = (rgb_comp_ij_unscaled[:,k] -\
                                          rgb_ref_ij_unscaled)/scaler_x1 +\
                                          rgb_ref_ij_unscaled
                
                # Convert the computed RGB values of the comparison stimulus 
                # into Lab values 
                comp_Lab_i,_,_ = simulations_CIELab.convert_rgb_lab(\
                    B_monitor, background_RGB, rgb_comp_ij_unscaled_k)
                # Calculate the perceptual difference (deltaE) between the 
                # reference and comparison stimuli as the Euclidean distance 
                # between their Lab values.
                deltaE[k] = np.linalg.norm(comp_Lab_i - lab_ref_ij)
                pC_weibull[k] = WeibullFunc(deltaE[k])
            #store them
            deltaE_allRef[i,j] = np.reshape(deltaE, (nGrid_y,nGrid_x))
            pC_weibull_allRef[i,j] = np.reshape(pC_weibull,(nGrid_y,nGrid_x))
        
    return sim, rgb_comp_dim1_grid, rgb_comp_dim2_grid, pChoosingX1_grid_allRef,\
        pC_weibull_allRef, deltaE_allRef, rgb_comp_varying_grid

#%% three variables we need to define for loading the data
plane_2D   = 'RG plane'
sim_jitter = '0.1'
nSims      = 240 #number of simulations: 240 trials for each ref stimulus
BANDWIDTH  = 5e-3
scaler_x1  = 5
nGrid_x    = 40
nGrid_y    = 50
width      = 0.25

fig_outputDir2 = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                 'ELPS_analysis/ModelFitting_FigFiles/'
sim, rgb_comp_dim1_grid, rgb_comp_dim2_grid, pChoosingX1_grid_allRef,\
    pC_weibull_allRef, deltaE_allRef, rgb_comp_varying_grid = compute_probC1_contour(\
    plane_2D, sim_jitter, nSims, BANDWIDTH, scaler_x1, nGrid_x, nGrid_y, width)

#%%
# Default parameters for ellipsoid fitting. Can be overridden by kwargs.
pltP = {
    'cmap_filled':'twilight',
    'cmap_nLevels': 20,
    'cmap_contour':'white',
    'title':"",
    'saveFig':True,
    'figDir': fig_outputDir2} 

#visualize 
plot_contour_pChoosingX1(sim['ref_points'][sim['varying_RGBplane']], rgb_comp_dim1_grid,\
    rgb_comp_dim2_grid, pChoosingX1_grid_allRef,cmap_nLevels = pltP['cmap_nLevels'],\
    title   = "Probability of choosing X1 predicted by the Wishart model",\
    saveFig = pltP['saveFig'], figDir  = pltP['figDir'],\
    figName = 'Contour_pChoosingX1_'+plane_2D +'_sim'+str(nSims)+'perCond_'+\
                    'samplingNearContour_jitter'+sim_jitter+'_WishartFits')

plot_contour_pChoosingX1(sim['ref_points'][sim['varying_RGBplane']], rgb_comp_dim1_grid,\
    rgb_comp_dim2_grid, pC_weibull_allRef, cmap_nLevels = pltP['cmap_nLevels'],\
    title = "Probability of choosing X1 predicted by Weibull"+\
            " Psychometric function",\
    saveFig = pltP['saveFig'], figDir  = pltP['figDir'],\
    figName = 'Contour_pChoosingX1_'+plane_2D+'_WeibullPMF')
    
    