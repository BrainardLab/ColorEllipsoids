#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 23:37:26 2024

@author: fangfang
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
import sys
import numpy as np
import os
import imageio.v2 as imageio
import configparser
import io
import ast

#load functions from other modules
sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids")
from core import optim, model_predictions, oddity_task
from core.wishart_process import WishartProcessModel
sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/"+\
                "Python version")
from simulations_CIELab import plot_2D_randRef_nearContourComp

#%%set path and load another file
# Define a dictionary to map plane names to indices
plane_2D_dict    = {'GB plane': 0, 'RB plane': 1, 'RG plane': 2}
# Select the current plane of interest
plane_2D         = 'RG plane' 
# Get the index of the current plane from the dictionary
plane_2D_idx     = plane_2D_dict[plane_2D]
# Create a list of indices representing RGB planes and remove the index of the current plane
varying_RGBplane = list(range(3)); varying_RGBplane.remove(plane_2D_idx)

# Set the path to the directory containing model fitting data files
path_str_WP      = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                   'ELPS_analysis/ModelFitting_DataFiles/2D_oddity_task/'  
file_fits        = f'Fitted_isothreshold_{plane_2D}_sim240perCond_'+\
                   'samplingNearContour_jitter0.1_bandwidth0.005.pkl'
full_path_fits   = f"{path_str_WP}{file_fits}"
# Open and load the model fitting data from the pickle file
with open(full_path_fits, 'rb') as f: data_load = pickle.load(f)
W_est            = data_load['W_est']      #estimated weight matrix
model            = data_load['model']      #model used for fitting
opt_params       = data_load['opt_params'] #parameters used for optimization
modelPred_ell    = data_load['recover_fitEllipse_unscaled']

#%%
degree =2 
NUM_GRID_PTS = 100
xgrid = jnp.stack(jnp.meshgrid(*[jnp.linspace(-1,1, NUM_GRID_PTS) \
                                 for _ in range(model.num_dims)]), axis=-1)
Sigmas_est_grid = model.compute_Sigmas(model.compute_U(W_est, xgrid))

#%%
val_slc = np.linspace(20,80,5)
for p in range(5):
    for q in range(5):
        ticks = np.linspace(0.2, 0.8,3)
        bds = np.max(np.abs(Sigmas_est_grid))
        plt.rcParams['figure.dpi'] = 250 
        fig, axes = plt.subplots(degree, degree+2, figsize=(8,4), sharex=True, sharey=True)
        # Define the grid layout
        gs = gridspec.GridSpec(2, 4)
        cmap = plt.get_cmap('PRGn')
        
        for i in range(degree):
            for j in range(degree):      
                # Show the 2D polynomial data
                axes[i, j].imshow(Sigmas_est_grid[:,:,i,j], cmap = cmap, vmin = -bds, vmax = bds)
                axes[i, j].plot([0, NUM_GRID_PTS], [val_slc[q], val_slc[q]], c = 'k')
                axes[i, j].plot([val_slc[p], val_slc[p]], [0, NUM_GRID_PTS], c = 'k')
                axes[i, j].set_xticks(ticks * NUM_GRID_PTS)
                axes[i, j].set_xticklabels(ticks)
                axes[i, j].set_yticks(ticks * NUM_GRID_PTS)
                axes[i, j].set_yticklabels(ticks)
                axes[i, j].set_xlim([0,NUM_GRID_PTS])
                axes[i, j].set_ylim([0,NUM_GRID_PTS])
                axes[i, j].set_title(r'$\sigma_{{({}, {})}}$'.format(i+1, j+1))
        
        # Remove the axes that will be merged into the big plot
        plt.delaxes(axes[0, 2])
        plt.delaxes(axes[0, 3])
        plt.delaxes(axes[1, 2])
        plt.delaxes(axes[1, 3])
        
        # Add a new large subplot that spans the last two columns of both rows
        ax_ell = fig.add_subplot(1, 2, 2)  # This creates a subplot that spans the right half of the figure
        for pp in range(p+1):
            for qq in range(q+1):
                ax_ell.plot(modelPred_ell[pp,qq,0], modelPred_ell[pp,qq,1],c=[0,0,0])
        ax_ell.set_xlim([-1,1])
        ax_ell.set_ylim([-1,1])
        ax_ell.set_xticks(np.linspace(-0.6,0.6,5))
        ax_ell.set_xticklabels((np.linspace(-0.6,0.6,5)+1)/2)
        ax_ell.set_yticks(np.linspace(-0.6,0.6,5))
        ax_ell.set_yticklabels((np.linspace(-0.6,0.6,5)+1)/2)
        ax_ell.grid(True)
        # Show the plot
        plt.tight_layout()
        plt.show()
        
    
    
