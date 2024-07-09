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
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from numpy.polynomial.chebyshev import chebval, chebval2d
import sys
sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids')
from core import chebyshev
import jax.numpy as jnp
import os
import imageio.v2 as imageio
import pickle

# Set the output directory for figures
fig_outputDir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                        'ELPS_analysis/WishartPractice_FigFiles/'
                        
#%% 1D Chebyshev Visualization
# Degree of the Chebyshev polynomial
degree = 5 
# Define the bounds and number of bins for the grid
lb, ub, nbins = -1, 1, 41
# Generate a linear grid
grid = np.linspace(lb, ub, nbins)
# Get basis coefficients for Chebyshev polynomials
basis_coeffs = chebyshev.chebyshev_basis(degree)
 # Evaluate Chebyshev polynomials at the grid points
lg = chebyshev.evaluate(basis_coeffs, grid)

# Create a subplot for each degree of polynomial and plot them
plt.rcParams['figure.dpi'] = 250 
fig, axes = plt.subplots(degree,1, figsize=(2,8), sharex=True, sharey=True)
for i in range(degree):
    axes[i].plot(grid,lg[:,i],color = 'k', linewidth = 2)
    axes[i].set_aspect('equal')
plt.show()
full_path = os.path.join(fig_outputDir,'Chebyshev_basis_functions_1D.png') 
#fig.savefig(full_path)   

#%% 2D Chebyshev Visualization
# Create a 2D grid for x and y
xg, yg = np.meshgrid(grid,grid)
# Initialize the coefficient grid for 2D
cg = np.zeros((degree, degree))
plt.rcParams['figure.dpi'] = 250 
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
#fig.savefig(full_path) 

#%% 3D Chebyshev Visualization
def plot_basis_functions_3D(XG, YG, ZG, M, **kwargs):
    pltP = {
        'figSize':(8,9),
        'xlim':[-1.05, 1.05],
        'ylim':[-1.05, 1.05],
        'zlim':[-1.05, 1.05],
        'saveFig':False,
        'saveGif':False,
        'figDir':'',
        'figName':'Chebyshev_basis_function',
        } 
    pltP.update(kwargs)
    fig_outputDir = pltP['figDir']
    
    nbins = M.shape[0]
    num_dim1, num_dim2 = M.shape[3:5]
    # Color map
    for l in range(nbins):
        plt.rcParams['figure.dpi'] = 250 
        # Create a 3D plot
        #since we can only visualize 2D basis function, the 3rd dimension is 
        #illustrated as time dimension
        fig, ax = plt.subplots(num_dim1, num_dim2, figsize= pltP['figSize'],\
                               subplot_kw={'projection': '3d'})
        for i in range(num_dim1):
            for j in range(num_dim2): 
                max_val = np.max([-np.min(M), np.max(M)])
                ax[i, j].plot_surface(XG[:,:,l], ZG[:,:,l], YG[:,:,l],\
                    facecolors=cmap(((M[:,:,l,i,j] + max_val)/ (2*max_val+1e-10))),
                    rstride=1, cstride=1)
                
                ax[i, j].set_xticks([])
                ax[i, j].set_xticklabels([])
                ax[i, j].set_yticklabels([])
                ax[i, j].set_zticks([])
                ax[i, j].set_zticklabels([])
                ax[i, j].set_xlim(pltP['xlim'])
                ax[i, j].set_ylim(pltP['ylim'])
                ax[i, j].set_zlim(pltP['zlim'])
                ax[i, j].view_init(20,-75)
                ax[i, j].set_aspect('equal')
                ax[i, j].set_autoscale_on(False)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None,\
                            wspace=-0.1, hspace=-0.1)
        plt.show()
        if pltP['saveFig'] and pltP['figDir'] != '':
            fig_name = pltP['figName'] + f'_slice{l:02}'
            full_path = os.path.join(pltP['figDir'],fig_name+'.png') 
            fig.savefig(full_path)   
    if pltP['saveFig'] and pltP['figDir'] != '' and pltP['saveGif']:
        # make a gif
        images = [img for img in os.listdir(pltP['figDir']) if img.startswith(fig_name[:-2])]
        images.sort()  # Sort the images by name (optional)
        image_list = [imageio.imread(f"{fig_outputDir}/{img}") for img in images]
        # Create a GIF
        gif_name = fig_name[:-8] + '.gif'
        output_path = f"{fig_outputDir}{gif_name}" 
        imageio.mimsave(output_path, image_list, fps=2)  
        
#%% Visualize 3D basis functions
# Create 3D meshgrids from the provided `grid` array for x, y, and z coordinates respectively.
X_mesh, Y_mesh, Z_mesh = np.meshgrid(grid, grid, grid)
# Stack the flattened x, y, and z arrays into a single matrix and then transpose it,
# resulting in a matrix where each row represents a point (x, y, z) in the 3D space.
XYZ_mesh = np.transpose(np.stack((X_mesh.flatten(), Y_mesh.flatten(), Z_mesh.flatten())),(1,0))

# Evaluate Chebyshev polynomials for each coordinate (x, y, z) and compute their outer product.
# This results in a high-dimensional array where each element represents the product of
# Chebyshev polynomial values at each point in the 3D space.
phi = (chebyshev.evaluate(basis_coeffs, XYZ_mesh[:,0])[:,:,None,None] *\
       chebyshev.evaluate(basis_coeffs, XYZ_mesh[:,1])[:,None,:,None] *\
       chebyshev.evaluate(basis_coeffs, XYZ_mesh[:,2])[:,None,None,:])

# Reshape the resulting tensor `phi` to have dimensions corresponding to bins and degrees
# for x, y, z coordinates, and polynomial degrees, making it easier to handle and interpret.
phi_org = np.reshape(phi, (nbins, nbins, nbins, degree, degree, degree))

# Loop through each degree of the Chebyshev polynomials.
for i in range(degree):
    # Plot 3D basis functions for each degree using the calculated values in `phi_org`,
    # and save the plots and GIFs to a specified directory.
    plot_basis_functions_3D(X_mesh, Y_mesh, Z_mesh, phi_org[:,:,:,:,:,i],\
            saveFig = False, saveGif = False, figDir = fig_outputDir + \
                'Chebyshev_basis_functions/', \
            figName = 'Chebyshev_3D_basis_function_degree'+str(i+1))

#%% load W from a file that contains estimated W matrix
# Define the file name and output directory for model fitting data files
file_name = 'Sims_isothreshold_ellipsoids_sim240perCond_samplingNearContour_jitter0.1.pkl'
outputDir_file = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                        'ELPS_analysis/ModelFitting_DataFiles/3d_oddity_task/'
output_file = 'Fitted'+file_name[4:-4]+'_bandwidth0.005.pkl'
full_path4 = f"{outputDir_file}{output_file}"

# Open the specified pickle file and load data structures contained within it
with open(full_path4, 'rb') as f:
    # Load the object from the file
    data_load = pickle.load(f)
    W_est = data_load['W_est']
    model = data_load['model']
    Sigmas_est_grid = data_load['Sigmas_est_grid']
    fitEllipsoid = data_load['recover_fitEllipsoid_scaled']

# Compute basis orders for polynomial expansion in 3D
basis_orders = (
    jnp.arange(degree)[:, None, None] +
    jnp.arange(degree)[None, :, None] + 
    jnp.arange(degree)[None, None, :]
)

#%%  Visualize W
slc_slice = [0,0] #selected indices for the 4th and 5th dimension
cmap_b = plt.get_cmap('RdBu')
cmap_bds = np.max([-np.min(W_est), np.max(W_est)])
for i in range(degree):
    fig, ax = plt.subplots(1, 1, figsize=(5,5), sharex=True, sharey=True)
    ax.imshow(W_est[:,:,i,*(slc_slice)], cmap = cmap_b, vmin = -cmap_bds,\
              vmax = cmap_bds)
    # Loop through each grid point to display the corresponding basis order as text
    for j in range(degree):
        for k in range(degree):
            # Display text over the image
            ax.text(j, k, str(basis_orders[j,k,i]), color='black',\
                    fontsize=20, ha='center', va='center')
    ax.set_xticks(list(range(degree)))
    ax.set_xticklabels([])
    ax.set_yticks(list(range(degree)))
    ax.set_yticklabels([])
    ax.set_xticks(np.arange(-0.5, W_est.shape[0], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, W_est.shape[1], 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
    fig_name = 'EstimatedWeightMatrix_degree'+str(i)+'_'+str(slc_slice)
    full_path = os.path.join(fig_outputDir + 'Estimated_W_matrix/',fig_name+'.png') 
    #fig.savefig(full_path)   


#%% Visualize weighted basis functions, a.k.a., U
U = np.einsum('ijkxyz, xyzpq -> ijkpq', phi_org, W_est)
plot_basis_functions_3D(X_mesh, Y_mesh, Z_mesh, U, figSize = (8,7),\
                        saveFig = False, saveGif = False, figDir = fig_outputDir + \
                            'U_given_estimatedWeightMatrix/', \
                        figName = 'U_given_estimatedWeightMatrix')

#%% Visualize positive semi-definite covariance matrices computed using U
Sigmas = np.einsum('ijkxm, ijkym -> ijkxy', U, U)
slc_idx = [8,14,20,26,32] #array([-0.6, -0.3,  0. ,  0.3,  0.6])
Sigmas_subset = Sigmas[slc_idx][:,slc_idx][:,:,slc_idx]

#quick sanity check
if np.all(np.abs(Sigmas_est_grid - Sigmas_subset) < 1e-10): print('sanity check passed!')
#plot
plot_basis_functions_3D(X_mesh, Y_mesh, Z_mesh, Sigmas, figSize = (6,6.5),\
                        saveFig = False, saveGif = False, figDir = fig_outputDir + \
                            'CovarianceMatrix/', \
                        figName = 'CovarianceMatrix')

#%% Visualize ellipsoids defined by 3 x 3 positive semi-definite cov matrices
from matplotlib.colors import LightSource, LinearSegmentedColormap
idx_slc = [0, 4]
fitEllipsoid_slc = fitEllipsoid[idx_slc][:,idx_slc][:,:,idx_slc]
fitEllipsoid_reshape = np.reshape(fitEllipsoid_slc, (np.prod(fitEllipsoid_slc.shape[0:3]),)+\
                                  fitEllipsoid_slc.shape[3:])

fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection='3d')
ticks = np.array([-0.6, 0, 0.6])

numE = fitEllipsoid_reshape.shape[0]
for i in range(numE):
    ell_i = fitEllipsoid_reshape[i]
    ell_i_x = np.reshape(ell_i[0],(100,200))
    ell_i_y = np.reshape(ell_i[1],(100,200))
    ell_i_z = np.reshape(ell_i[2],(100,200))
    ellipsoid_reshape = np.reshape(ell_i, (3, 100, 200))
    cmap_i = (np.mean(ell_i, axis = 1) + 1)/2
    ax.plot_surface(ell_i_x, ell_i_y,\
                    ell_i_z, color = cmap_i, edgecolor =cmap_i,alpha = 0.5, lw = 0.5)
        
        # Create light source object.
#    ls = LightSource(azdeg=0, altdeg=65)
#    # Shade data, creating an rgb array.
#    rgb_temp = (np.mean(ell_i, axis = 1) + 1)/2
#    # Create a custom colormap with a single color
#    ls = LightSource(azdeg=0, altdeg=65)
#    cmap = LinearSegmentedColormap.from_list("single_color", [rgb_temp,rgb_temp], N=256)
#    rgb = ls.shade(ellipsoid_reshape[2], cmap=cmap)
#    surf = ax.plot_surface(ellipsoid_reshape[0], ellipsoid_reshape[1],\
#                ellipsoid_reshape[2], rstride=1, cstride=1, linewidth=0,\
#                       antialiased=False, facecolors=rgb, alpha = 0.5)
ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_zlim([-1,1])
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_zticks(ticks)
ax.set_xlabel('R')
ax.set_ylabel('G')
ax.set_zlabel('B')
ax.grid(True)
ax.set_aspect('equal')
    



