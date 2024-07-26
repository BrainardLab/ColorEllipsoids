#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 16:13:50 2024

@author: fangfang
"""

from jax import config
config.update("jax_enable_x64", True)
import numpy as np
import sys
sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids')
from core import chebyshev
import jax.numpy as jnp
import pickle
sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/"+\
                "Python version/wishart_visualization")
from wishart_plotting import wishart_model_basics_visualization

# Set the output directory for figures
fig_outputDir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                'ELPS_analysis/WishartPractice_FigFiles/'
                        
#%% Chebyshev Visualization
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

# 1D plot
visualize_basis_1D_2D = wishart_model_basics_visualization(\
        fig_dir=fig_outputDir +'Chebyshev_basis_functions/',  save_fig=False)
visualize_basis_1D_2D.plot_basis_function_1d(degree, grid, lg)

# 2D Chebyshev Visualization
visualize_basis_1D_2D.plot_basis_function_2D(degree, grid)

#%% Visualize 3D basis functions
# Create 3D meshgrids from the provided `grid` array for x, y, and z coordinates respectively.
X_mesh, Y_mesh, Z_mesh = np.meshgrid(grid, grid, grid)
# Stack the flattened x, y, and z arrays into a single matrix and then transpose it,
# resulting in a matrix where each row represents a point (x, y, z) in the 3D space.
XYZ_mesh = np.transpose(np.stack((X_mesh.flatten(), Y_mesh.flatten(), \
                                  Z_mesh.flatten())),(1,0))

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
visualize_basis3D = wishart_model_basics_visualization(\
        fig_dir=fig_outputDir +'Chebyshev_basis_functions/',\
        save_fig=False, save_gif=False)
for i in range(degree):        
    # Plot 3D basis functions for each degree using the calculated values in `phi_org`,
    # and save the plots and GIFs to a specified directory.
    visualize_basis3D.plot_basis_functions_3D(degree, X_mesh, Y_mesh, Z_mesh,\
            phi_org[:,:,:,:,:,i], fig_name = f'Chebyshev_3D_basis_function_degree{i+1}')

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
# weighted basis functions, a.k.a., U
U = np.einsum('ijkxyz, xyzpq -> ijkpq', phi_org, W_est)

# Weight matrix
visualize_basis3D.fig_dir = fig_outputDir + 'Estimated_W_matrix/'
visualize_basis3D.plot_W_selected_slice(degree, W_est, basis_orders,\
                                        slc_slice = [0,0])
# Visualize U
visualize_U = wishart_model_basics_visualization(fig_dir=fig_outputDir +\
        'U_given_estimatedWeightMatrix/', save_fig=True, save_gif=True)
visualize_U.plot_basis_functions_3D(X_mesh, Y_mesh, Z_mesh,\
        U, fig_size = (8,7), fig_name = 'U_given_estimatedWeightMatrix')

#%% Visualize positive semi-definite covariance matrices computed using U
Sigmas = np.einsum('ijkxm, ijkym -> ijkxy', U, U)
slc_idx = [8,14,20,26,32] #array([-0.6, -0.3,  0. ,  0.3,  0.6])
Sigmas_subset = Sigmas[slc_idx][:,slc_idx][:,:,slc_idx]

#quick sanity check
if np.all(np.abs(Sigmas_est_grid - Sigmas_subset) < 1e-10): print('sanity check passed!')
#plot
visualize_U.fig_dir = fig_outputDir + 'CovarianceMatrix/'
visualize_U.plot_basis_functions_3D(X_mesh, Y_mesh, Z_mesh,\
        Sigmas, fig_size = (6,6.5), fig_name ='CovarianceMatrix')


    


