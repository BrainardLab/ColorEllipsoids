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
import numpy as np
import os
import imageio.v2 as imageio
from matplotlib.colors import LinearSegmentedColormap

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

#%%  Grid settings
# 2D plane
degree            = 2
# Number of coarse grid points used for plotting
NUM_GRID_PTS      = 5  
# Number of fine grid points used for detailed plotting
NUM_GRID_PTS_FINE = 100  
# Lower bound of RGB space in Wishart unit
lb_W_unit         = -1  
# Upper bound of RGB space in Wishart unit
ub_W_unit         = 1  
# Coarse grid for plotting, in the range -0.6 to 0.6
xgrid_dim1        = jnp.linspace(-0.6, 0.6, NUM_GRID_PTS)
# Normalize the grid to range from 0 to 1  
xgrid_dim1_N_unit = (xgrid_dim1 + 1) / 2  
# Fine grid for detailed plotting
xgrid_dim1_fine   = jnp.linspace(lb_W_unit, ub_W_unit, NUM_GRID_PTS_FINE)  
# Create a meshgrid for the fine grid
xgrid_fine        = jnp.stack(jnp.meshgrid(*[xgrid_dim1_fine \
                                             for _ in range(model.num_dims)]), axis=-1) 
# Compute the estimated covariance matrices for the fine grid
Sigmas_est_grid   = model.compute_Sigmas(model.compute_U(W_est, xgrid_fine))  

#%% plotting
# Define the file name and output directory for model fitting data files
fig_outputDir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                        'ELPS_analysis/WishartPractice_FigFiles/CovarianceMatrix_2d/'
# Define titles for subplots using the first two letters of the plane
ttl_list = [[r'$\sigma^2_{' + plane_2D[0] + '}$', r'$\sigma_{' + plane_2D[0:2] + '}$'],\
            [r'$\sigma_{' + plane_2D[0:2] + '}$', '$\sigma^2_{' + plane_2D[1] + '}$']]
    
#color map
# Get a colormap
cmap_continuous = plt.get_cmap('PRGn')

# Create a discrete colormap
colors = cmap_continuous(np.linspace(0, 1, 100))
cmap_discrete = LinearSegmentedColormap.from_list("Discrete PRGn", colors, N=100)
# Calculate the maximum absolute value from the covariance matrices for color scaling
bds = np.max(np.abs(Sigmas_est_grid))

for p in range(NUM_GRID_PTS):
    for q in range(NUM_GRID_PTS):
        plt.rcParams['figure.dpi'] = 250 
        fig, axes = plt.subplots(degree, degree+2, figsize=(8,4), sharex=True, sharey=True)
        
        for i in range(degree):
            for j in range(degree):      
                # Plot the covariance matrix using a heatmap
                im = axes[i, j].imshow(Sigmas_est_grid[:,:,i,j], cmap = cmap_discrete,\
                                  vmin = -bds*1.2, vmax = bds)
                # Calculate the scaled position for the horizontal line
                xgrid_scaled_p = xgrid_dim1_N_unit[4-p]*NUM_GRID_PTS_FINE
                # Draw horizontal line
                axes[i, j].plot([0, NUM_GRID_PTS_FINE], [xgrid_scaled_p,xgrid_scaled_p],\
                                c = 'grey',lw = 0.5)
                # Calculate the scaled position for the vertical line
                xgrid_scaled_q = xgrid_dim1_N_unit[q]*NUM_GRID_PTS_FINE
                # Draw vertical line
                axes[i, j].plot([xgrid_scaled_q, xgrid_scaled_q],\
                                [0, NUM_GRID_PTS_FINE],\
                                c = 'grey',lw = 0.5)
                # Mark the intersection point
                axes[i, j].scatter(xgrid_scaled_q, xgrid_scaled_p, c = 'k', s = 10)
                # ticks and title
                axes[i, j].set_xticks(xgrid_dim1_N_unit[::2] * NUM_GRID_PTS_FINE)
                axes[i, j].set_xticklabels(xgrid_dim1_N_unit[::2])
                axes[i, j].set_yticks(xgrid_dim1_N_unit[::2] * NUM_GRID_PTS_FINE)
                axes[i, j].set_yticklabels(xgrid_dim1_N_unit[::2])
                axes[i, j].set_xlim([0,NUM_GRID_PTS_FINE-1])
                axes[i, j].set_ylim([0,NUM_GRID_PTS_FINE-1])
                axes[i, j].set_title(ttl_list[i][j])
        
        # Remove the axes that will be merged into the big plot
        plt.delaxes(axes[0, 2])
        plt.delaxes(axes[0, 3])
        plt.delaxes(axes[1, 2])
        plt.delaxes(axes[1, 3])
        
        cbar_ax = fig.add_axes([0.065, 0.1, 0.4, 0.02])  # [left, bottom, width, height]
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
        
        # Add a new large subplot that spans the last two columns of both rows
        # This creates a subplot that spans the right half of the figure
        ax_ell = fig.add_subplot(1, 2, 2)  
        for pp in range(NUM_GRID_PTS):
            for qq in range(NUM_GRID_PTS):
                # Plot ellipses based on the model predictions
                if pp < p or (pp == p and qq <= q):
                    ax_ell.plot(modelPred_ell[NUM_GRID_PTS-1-pp,qq,0],\
                                modelPred_ell[NUM_GRID_PTS-1-pp,qq,1],c='k')
        # ticks and title
        ax_ell.set_xlim([lb_W_unit,ub_W_unit])
        ax_ell.set_ylim([lb_W_unit,ub_W_unit])
        ax_ell.set_xticks(xgrid_dim1)
        ax_ell.set_xticklabels('{:.2f}'.format(x) for x in xgrid_dim1_N_unit)
        ax_ell.set_yticks(xgrid_dim1)
        ax_ell.set_yticklabels('{:.2f}'.format(x) for x in xgrid_dim1_N_unit)
        ax_ell.grid(True, alpha=0.5)
        ax_ell.set_xlabel(plane_2D[0], fontsize = 12)
        ax_ell.set_ylabel(plane_2D[1], fontsize = 12)
        ax_ell.set_title(plane_2D, fontsize = 12)
        ax_ell.set_aspect('equal')
        # Show the plot
        #plt.tight_layout(pad=-0.25)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.925, bottom=0.2)
        plt.show()
        fig_name = f'CovarianceMatrix_{plane_2D}_{(p*NUM_GRID_PTS+q):02d}.png' 
        full_path = f"{fig_outputDir}{fig_name}"
        fig.savefig(full_path) 

#%% make a gif
images = [img for img in os.listdir(fig_outputDir) \
          if img.startswith(f'CovarianceMatrix_{plane_2D}') and img.endswith('.png')]
images.sort()  # Sort the images by name (optional)
image_list = [imageio.imread(f"{fig_outputDir}/{img}") for img in images]
# Create a GIF
gif_name = f'CovarianceMatrix_{plane_2D}.gif'
output_path = f"{fig_outputDir}{gif_name}" 
imageio.mimsave(output_path, image_list, fps=2)  
    
