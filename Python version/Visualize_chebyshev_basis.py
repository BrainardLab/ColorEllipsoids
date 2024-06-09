#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 16:13:50 2024

@author: fangfang
"""

from jax import config
config.update("jax_enable_x64", True)
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.polynomial.chebyshev import chebval, chebval2d
import sys
sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids')
from core import chebyshev
import jax.numpy as jnp
import os
import imageio.v2 as imageio

# Set the output directory for figures
fig_outputDir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                        'ELPS_analysis/WishartPractice_FigFiles/'
                        
#%% 1D Chebyshev Visualization
# Degree of the Chebyshev polynomial
degree = 5 
# Define the bounds and number of bins for the grid
lb, ub, nbins = -1, 1, 30
# Generate a linear grid
grid = np.linspace(lb, ub, nbins)
# Get basis coefficients for Chebyshev polynomials
basis_coeffs = chebyshev.chebyshev_basis(degree)
 # Evaluate Chebyshev polynomials at the grid points
lg = chebyshev.evaluate(basis_coeffs, grid)

# Create a subplot for each degree of polynomial and plot them
fig, axes = plt.subplots(degree,1, figsize=(2,8), sharex=True, sharey=True)
for i in range(degree):
    axes[i].plot(grid,lg[:,i],color = 'k', linewidth = 2)
    axes[i].set_aspect('equal')
plt.show()
full_path = os.path.join(fig_outputDir,'Chebyshev_basis_functions_1D.png') 
fig.savefig(full_path)   

#%% 2D Chebyshev Visualization
# Create a 2D grid for x and y
xg, yg = np.meshgrid(grid,grid)
# Initialize the coefficient grid for 2D
cg = np.zeros((degree, degree))
fig, axes = plt.subplots(degree, degree, figsize=(8,8), sharex=True, sharey=True)
cmap = plt.get_cmap('PRGn')
for i in range(degree):
    for j in range(degree):
        # Set current coefficient to 1 to visualize its effect
        cg[i, j] = 1.0
        # Evaluate the 2D polynomial at the grid
        zg_2d = chebval2d(xg, yg, cg)
        
        # Show the 2D polynomial data
        axes[i, j].imshow(zg_2d, cmap = cmap, vmin = lb, vmax = ub)
        # Reset the coefficient
        cg[i, j] = 0.0

        axes[i, j].set_xticks([])
        axes[i, j].set_xticklabels([])
        axes[i, j].set_yticks([])
        axes[i, j].set_yticklabels([])
        axes[i, j].set_title(f"({i}, {j})")
        axes[i, j].set_xlim([0,nbins-1])
        axes[i, j].set_ylim([0,nbins-1])

fig.tight_layout()
full_path = os.path.join(fig_outputDir,'Chebyshev_basis_functions_2D.png') 
fig.savefig(full_path) 

#%% 3D Chebyshev Visualization
# Set a fixed layer for visualization
# This value is between 0 and 4
fixed_l = 0 
# Color map
for l in range(nbins):
    # Create a 3D plot
    #since we can only visualize 2D basis function, the 3rd dimension is 
    #illustrated as time dimension
    fig, ax = plt.subplots(degree,degree, figsize=(8,9),subplot_kw={'projection': '3d'})
    for i in range(cg.shape[0]):
        for j in range(cg.shape[1]):        
            cg[i, j] = 1.0
            # Evaluate the 2D Chebyshev polynomial
            zg = chebval2d(xg, yg, cg)
            # MULTIPLY BY lg!! 
            zg = lg[l,fixed_l]*zg
            # Plot each as a surface in the 3D space
            ax[i, j].plot_surface(xg, grid[l]*np.ones(xg.shape), yg,\
                facecolors=cmap(((zg +1)/ (2+1e-10))),
                rstride=1, cstride=1)
            # Reset the coefficient
            cg[i, j] = 0.0
            
            ax[i, j].set_xticks([])
            ax[i, j].set_xticklabels([])
            #ax[i, j].set_yticks([])
            ax[i, j].set_yticklabels([])
            ax[i, j].set_zticks([])
            ax[i, j].set_zticklabels([])
            ax[i, j].set_xlim([-1.05,1.05])
            ax[i, j].set_ylim([-1.05,1.05])
            ax[i, j].set_zlim([-1.05,1.05])
            ax[i, j].view_init(20,-75)
            ax[i, j].set_aspect('equal')
            ax[i, j].set_autoscale_on(False)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,\
                        wspace=-0.1, hspace=-0.1)
    plt.show()
    fig_name = f'Chebyshev_basis_function_z{fixed_l}_slice{l:02}'
    full_path = os.path.join(fig_outputDir,fig_name+'.png') 
    fig.savefig(full_path)   

#%%
# make a gif
images = [img for img in os.listdir(fig_outputDir) if img.startswith(fig_name[:-2])]
images.sort()  # Sort the images by name (optional)

# Load images using imageio.v2 explicitly to avoid deprecation warnings
image_list = [imageio.imread(f"{fig_outputDir}/{img}") for img in images]

# Create a GIF
gif_name = fig_name[:-8] + '.gif'
output_path = f"{fig_outputDir}{gif_name}" 
imageio.mimsave(output_path, image_list, fps=2)  








