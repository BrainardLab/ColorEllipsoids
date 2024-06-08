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
fig_outputDir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                        'ELPS_analysis/WishartPractice_FigFiles/'
                        
#%% 1D
degree = 5 
lb, ub, nbins = -1, 1, 30
grid = np.linspace(lb, ub, nbins)
basis_coeffs = chebyshev.chebyshev_basis(degree)
lg = chebyshev.evaluate(basis_coeffs, grid)


#%% 2D
xg, yg = np.meshgrid(grid,grid)

cg = np.zeros((degree, degree))
fig, axes = plt.subplots(degree, degree, figsize=(8,8), sharex=True, sharey=True)
cmap = plt.get_cmap('PRGn')
for i in range(degree):
    for j in range(degree):
        cg[i, j] = 1.0
        zg_2d = chebval2d(xg, yg, cg)
        axes[i, j].imshow(zg_2d, cmap = cmap, vmin = lb, vmax = ub)
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

#%% 3D
fixed_l = 0
# Color map
for l in range(nbins):
    # Create a 3D plot
    fig, ax = plt.subplots(degree,degree, figsize=(8,9),subplot_kw={'projection': '3d'})
    for i in range(cg.shape[0]):
        for j in range(cg.shape[1]):        
            cg[i, j] = 1.0
            # Evaluate the 2D Chebyshev polynomial
            zg = chebval2d(xg, yg, cg)
            zg = lg[l,fixed_l]*zg
            # Plot each as a surface in the 3D space, with an offset in z to separate them visually
            ax[i, j].plot_surface(xg, grid[l]*np.ones(xg.shape), yg,\
                facecolors=cmap(((zg +1)/ (2+1e-10))),#cmap((zg - zg.min()) / (zg.max() - zg.min() + 1e-10)),\
                rstride=1, cstride=1)
            
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








