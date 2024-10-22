#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 12:06:56 2024

@author: fangfang
"""

import jax
jax.config.update("jax_enable_x64", True)  # Enable 64-bit precision in JAX
import matplotlib.pyplot as plt
import sys
import numpy as np
import jax.numpy as jnp
sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids')

# Import functions and classes from your project
from core.probability_surface import IndividualProbSurfaceModel, unrotate_unstretch_one_ref_stim_3D,\
    unrotate_unstretch_3D, pC_Weibull_one_ref_stim, pC_Weibull, estimate_nloglikelihood_3D,\
        optimize_nloglikelihood
from analysis.ellipsoids_tools import UnitCircleGenerate_3D, PointsOnEllipsoid, rotation_angles_to_eigenvectors

#%%
# Initialize an instance of IndividualProbSurfaceModel
model_indv = IndividualProbSurfaceModel(2, 
                                        [1e-1,3],  # Bounds for radii
                                        [0, 2*jnp.pi],  # Bounds for angle in radians
                                        [0.5, 5],  # Bounds for Weibull parameter 'a'
                                        [0.1, 5],
                                        ndims = 3)  # Bounds for Weibull parameter 'b'
                                        
KEY = jax.random.PRNGKey(225)  # Initialize random key for reproducibility
init_params = model_indv.sample_params_prior(KEY)  # Sample initial parameters for the model

# Define reference points (xy_ref) where the ellipses are located
xyz_ref = jnp.array(np.meshgrid([0, 1], [0, 1], [0, 1])).T.reshape(-1, 3)
nRefs = xyz_ref.shape[0]  # Number of reference points (4 in this case)

# Define ellipse parameters: large radius (r1), small radius (r2), angle (in radians)
par_all = model_indv.sample_params_prior(jax.random.PRNGKey(3))
par_all = par_all[:,0:6]

nTrials = 3600  # Number of trials (or points) to generate for each ellipse
jitter = 0.3  # Amount of random jitter to add to the sampled points
unitEll = UnitCircleGenerate_3D(200, int(nTrials/200))
xyz_comp = np.full((nRefs, nTrials, 3), np.nan)  # Initialize array for comparison points (elliptical points)
unitEll_finer = UnitCircleGenerate_3D(200, 100)
xyz_comp_finer = np.full((nRefs, 20000, 3), np.nan)

#%% Loop over reference points and generate points on ellipses using the provided parameters
for i in range(nRefs):
    eigvec_i = rotation_angles_to_eigenvectors(*par_all[i,-3:])
    xyz_comp_i = PointsOnEllipsoid(par_all[i,0:3],
                             np.reshape(xyz_ref[i],[3,1]),
                             eigvec_i, unitEll)         # Center of ellipse, number of points (nTrials)
    xyz_comp_finer_i = PointsOnEllipsoid(par_all[i,0:3],
                             np.reshape(xyz_ref[i],[3,1]),
                             eigvec_i, unitEll_finer)         # Center of ellipse, number of points (nTrials)
    xyz_comp[i] = np.transpose(xyz_comp_i, [1,0])
    xyz_comp_finer[i] = np.transpose(xyz_comp_finer_i, [1,0])

# Transform the points using unrotate_unstretch for one reference ellipse
xyz_transformed_one_ref = unrotate_unstretch_one_ref_stim_3D(par_all[0], xyz_ref[0], xyz_comp[0])

# Add random jitter to the comparison points (xy_comp) and create xy_jitter
xyz_jitter = xyz_comp #+ jax.random.normal(KEY, shape=xyz_comp.shape) * jitter

# Transform the jittered points for all ellipses
xyz_transformed = unrotate_unstretch_3D(par_all, 
                                        xyz_ref, 
                                        xyz_jitter)

# %%Create 3D subplots
fig2 = plt.figure(figsize=(12, 6))  # Adjust size for 4 subplots

#Plot the transformed points in 3D
for i in range(nRefs):
    ax = fig2.add_subplot(2, 4, i+1, projection='3d')  # Create a 3D subplot for each ellipse
    ax.scatter(xyz_transformed[i, :, 0], xyz_transformed[i, :, 1],
               xyz_transformed[i, :, 2], s=2, c='gray',alpha = 0.1)  # Plot transformed points
    ax.scatter(0,0,0, c='r', marker='+')  # Plot the center (reference point)
    ax.set_aspect('equal')  # 3D aspect ratio can't be set to 'equal' like 2D

    # Optionally set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

plt.tight_layout()  # Adjust spacing between subplots
plt.show()

#%%
# Set Weibull parameters a and b (these control threshold and slope)
a = 1.17
b = 2.33
weibull_params = jnp.array([a, b])  # Store Weibull parameters in an array

# Compute the probability of correct response (pC) for one reference ellipse
pC_one_ref = pC_Weibull_one_ref_stim(weibull_params, xyz_transformed_one_ref)

# Compute pC for all ellipses using the Weibull psychometric function
pC = pC_Weibull(weibull_params, xyz_transformed)

# Generate random binary outcomes (y) for comparison based on the probabilities (pC)
y = (jax.random.uniform(KEY, shape=pC.shape) < pC).astype(int)

# Package data for likelihood estimation (observed responses, reference points, and jittered comparison points)
data = (y, xyz_ref, xyz_jitter)

# Estimate the negative log likelihood of the model with initial parameters
meanLL = estimate_nloglikelihood_3D(init_params, data)
print(meanLL)  # Print the initial mean log likelihood

# Stack ground truth parameters (ellipse + Weibull) for comparison
gt_params = np.hstack((par_all, np.tile(weibull_params[np.newaxis], (nRefs, 1))))

# Estimate negative log likelihood for the ground truth parameters
meanLL_gt = estimate_nloglikelihood_3D(gt_params, data)
print(meanLL_gt)  # Print the mean log likelihood for ground truth parameters

#%
# Run optimization to recover the best-fit parameters
params_recover, iters, objhist = optimize_nloglikelihood(
    init_params, data, 
    ndims = 3,
    total_steps=15000,           # Number of total optimization steps
    save_every=10,               # Save the objective value every 10 steps
    learning_rate = 5e-1,
    fixed_weibull_params=weibull_params,  # Fix the Weibull parameters during optimization
    bds_radii = jnp.array([1e-1,3]),
    bds_angle = jnp.array([0, 2*jnp.pi]),
    show_progress=True           # Show progress using tqdm
)

# Plot the optimization history (objective value vs iterations)
fig3, ax3 = plt.subplots(1, 1)
ax3.plot(iters, objhist)  # Plot iterations vs objective history
fig3.tight_layout()

#%%
for v in np.linspace(-100, 100, 10):
    # Recover ellipses from the optimized parameters
    xyz_recover = np.full((nRefs, 20000,3), np.nan)  # Initialize array to store recovered ellipses
    for i in range(nRefs):
        eigvec_i = rotation_angles_to_eigenvectors(*params_recover[i,3:6])
        # Reconstruct the recovered ellipses using the optimized parameters
        
        xyz_recover_i = PointsOnEllipsoid(params_recover[i,0:3],  # Semi-major and semi-minor axes
                                          np.reshape(xyz_ref[i],[3,1]),
                                          eigvec_i, unitEll_finer)       # Center of ellipse, number of points
        
        xyz_recover[i] = np.transpose(xyz_recover_i,[1,0])
        
    # Final visualization: compare recovered ellipses with ground truth ellipses
    fig = plt.figure(figsize=(12, 6), dpi = 256)  # Adjust size for 4 subplots
    for i in range(nRefs):
        ax = fig.add_subplot(2, 4, i+1, projection='3d')  # Create a 3D subplot for each ellipse
        # Plot original (ground truth) ellipses in red
        ax.plot(xyz_comp_finer[i,:,0], xyz_comp_finer[i,:,1], 
                xyz_comp_finer[i,:,2], c='r', lw=3, alpha = 0.8, label = 'ground truth')
        
        # Scatter plot of jittered comparison points
        ax.scatter(xyz_jitter[i,:,0], xyz_jitter[i,:,1], 
                      xyz_jitter[i,:,2], s=2, c='gray', alpha=0.1, label = 'sampled data')
        
        # Plot recovered ellipses in green on top of the original plot
        ax.plot(xyz_recover[i,:,0], xyz_recover[i,:,1],xyz_recover[i,:,2], c='g', 
                alpha = 0.6, label = 'model fits')
        
        # Plot the reference point (center of the ellipse)
        ax.scatter(*xyz_ref[i], c='r', marker='+', label ='ref stimulus')
        ax.set_aspect('equal')  # Ensure square aspect ratio for each plot    
        ax.view_init(30, v)
    ax.legend()
    