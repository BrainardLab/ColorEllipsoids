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
import pickle
import numpy as np
import sys

#load functions from other modules
sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids")
from core import model_predictions
from core.wishart_process import WishartProcessModel

#load functions from other modules
sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/"+\
                "Python version")
from simulations_CIELab import PointsOnEllipseQ
sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/"+\
                "Python version/wishart_visualization")
from wishart_plotting import wishart_model_basics_visualization

# Define the file name and output directory for model fitting data files
fig_outputDir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                        'ELPS_analysis/WishartPractice_FigFiles/CovarianceMatrix_2d/'

#%%set path and load another file
# Select the current plane of interest
plane_2D         = 'GB plane' 
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
modelPred_ell    = data_load['recover_fitEllipse_unscaled']

#%%  Grid settings
# Number of coarse grid points used for plotting
NUM_GRID_PTS      = 5  
# Number of fine grid points used for detailed plotting
NUM_GRID_PTS_FINE = 100  
# Coarse grid for plotting, in the range -0.6 to 0.6
xgrid_dim1        = jnp.linspace(-0.6, 0.6, NUM_GRID_PTS)
# Normalize the grid to range from 0 to 1  
xgrid_dim1_N_unit = (xgrid_dim1 + 1) / 2  
# Fine grid for detailed plotting
xgrid_dim1_fine   = jnp.linspace(-1, 1, NUM_GRID_PTS_FINE)  
# Create a meshgrid for the fine grid
xgrid_fine        = jnp.stack(jnp.meshgrid(*[xgrid_dim1_fine \
                                             for _ in range(model.num_dims)]), axis=-1) 
# Compute the estimated covariance matrices for the fine grid
Sigmas_est_grid   = model.compute_Sigmas(model.compute_U(W_est, xgrid_fine))  

#%%
visualize_sigma2D = wishart_model_basics_visualization(fig_dir=fig_outputDir,\
                                             save_fig=False, save_gif=False)
    
# Define titles for subplots using the first two letters of the plane
ttl_list = [[r'$\sigma^2_{' + plane_2D[0] + '}$', r'$\sigma_{' + plane_2D[0:2] + '}$'],\
            [r'$\sigma_{' + plane_2D[0:2] + '}$', '$\sigma^2_{' + plane_2D[1] + '}$']]
    
for p in range(NUM_GRID_PTS):
    for q in range(NUM_GRID_PTS):
        visualize_sigma2D.plot_2D_covMat(Sigmas_est_grid, modelPred_ell,\
                    xgrid_dim1_N_unit, slc_idx_dim1 = p, slc_idx_dim2 = q, \
                    title_list = ttl_list, plane_2D = plane_2D)
        
#save as a gif
if visualize_sigma2D.save_gif: 
    visualize_sigma2D._save_gif(visualize_sigma2D.pltP['fig_name'][:-7], \
                                   visualize_sigma2D.pltP['fig_name'][:-7])
    
#%% Randomly draw from prior 
# Random number generator seeds
W_INIT_KEY   = jax.random.PRNGKey(222)
model_test = WishartProcessModel(
    5,     # Degree of the polynomial basis functions
    2,     # Number of stimulus dimensions
    1,     # Number of extra inner dimensions in `U`.
    3e-4,  # Scale parameter for prior on `W`.
    0.1,   # Geometric decay rate on `W`.  #0.8 or 0.1
    0,     # Diagonal term setting minimum variance for the ellipsoids.
)

xgrid = jnp.stack(jnp.meshgrid(*[xgrid_dim1 for _ in range(model_test.num_dims)]), axis=-1)
W_test = model_test.sample_W_prior(W_INIT_KEY)
Sigmas_test_grid_fine = model_test.compute_Sigmas(model_test.compute_U(W_test, xgrid_fine))
Sigmas_test_grid = model_test.compute_Sigmas(model_test.compute_U(W_test, xgrid))

ellipses_rand = np.full((NUM_GRID_PTS, NUM_GRID_PTS, 2, 200), np.nan)
for p in range(NUM_GRID_PTS):
    for q in range(NUM_GRID_PTS):
        _,_, axes_lengths, theta = model_predictions.covMat_to_ellParamsQ(Sigmas_test_grid[p,q])
        ellipses_rand[p,q] = PointsOnEllipseQ(*axes_lengths*2, theta, *xgrid[p,q], nTheta = 200)

#%%
for p in range(NUM_GRID_PTS):
    for q in range(NUM_GRID_PTS):
        visualize_sigma2D.plot_2D_covMat(Sigmas_test_grid_fine, ellipses_rand,\
                       xgrid_dim1_N_unit, slc_idx_dim1 = p, slc_idx_dim2 = q,\
                       figName_ext = f'decayRate{model_test.decay_rate}_seed222')

#save as a gif
if visualize_sigma2D.save_gif: 
    visualize_sigma2D._save_gif(visualize_sigma2D.pltP['fig_name'][:-7], \
                                   visualize_sigma2D.pltP['fig_name'][:-7])

