#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 17:26:19 2024

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
from core.probability_surface import IndividualProbSurfaceModel, unrotate_unstretch_one_ref_stim,\
    unrotate_unstretch, pC_Weibull_one_ref_stim, pC_Weibull, estimate_nloglikelihood,\
        optimize_nloglikelihood
from analysis.ellipses_tools import PointsOnEllipseQ

#%%

# Initialize an instance of IndividualProbSurfaceModel
model_indv = IndividualProbSurfaceModel(2, 
                                        [1e-6,3],  # Bounds for radii
                                        [0, 2*jnp.pi],  # Bounds for angle in radians
                                        [0.5, 5],  # Bounds for Weibull parameter 'a'
                                        [0.1, 5])  # Bounds for Weibull parameter 'b'
                                        
KEY = jax.random.PRNGKey(222)  # Initialize random key for reproducibility
init_params = model_indv.sample_params_prior(KEY)  # Sample initial parameters for the model

# Define reference points (xy_ref) where the ellipses are located
xy_ref = jnp.array([[0,0],
                    [1,1],
                    [0,1],
                    [1,0]])
nRefs = xy_ref.shape[0]  # Number of reference points (4 in this case)

# Define ellipse parameters: large radius (r1), small radius (r2), angle (in radians)
par_all = model_indv.sample_params_prior(jax.random.PRNGKey(0))
par_all = par_all[:,0:3]

nTrials = 1000  # Number of trials (or points) to generate for each ellipse
jitter = 0.3  # Amount of random jitter to add to the sampled points
xy_comp = np.full((nRefs, nTrials, 2), np.nan)  # Initialize array for comparison points (elliptical points)

# Loop over reference points and generate points on ellipses using the provided parameters
for i in range(nRefs):
    ell_x, ell_y = PointsOnEllipseQ(par_all[i,0], par_all[i,1],  # Large and small radii
                             jnp.rad2deg(par_all[i,2]),          # Convert angle from radians to degrees
                             *xy_ref[i], nTheta=nTrials)         # Center of ellipse, number of points (nTrials)

    xy_comp[i] = jnp.vstack((ell_x, ell_y)).T  # Store the points for each ellipse

# Transform the points using unrotate_unstretch for one reference ellipse
xy_transformed_one_ref = unrotate_unstretch_one_ref_stim(par_all[0], xy_ref[0], xy_comp[0])

# Add random jitter to the comparison points (xy_comp) and create xy_jitter
xy_jitter = xy_comp + jax.random.normal(KEY, shape=xy_comp.shape) * jitter

# Transform the jittered points for all ellipses
xy_transformed = unrotate_unstretch(par_all, 
                                    xy_ref, 
                                    xy_jitter)

# Plot the transformed (unrotated and unstretched) points
fig2, ax2 = plt.subplots(1, 4, figsize=(8, 2))  # 4 subplots, one for each ellipse
for i in range(4):
    ax2[i].scatter(xy_transformed[i,:,0], xy_transformed[i,:,1], s=2, c='gray', alpha = 0.2)  # Plot transformed points
    ax2[i].scatter(0, 0, c='r', marker='+')  # Plot the center (reference point)
    ax2[i].set_aspect('equal', 'box')  # Ensure square aspect ratio for each plot

#%%
# Set Weibull parameters a and b (these control threshold and slope)
a = 1.17
b = 2.33
weibull_params = jnp.array([a, b])  # Store Weibull parameters in an array

# Compute the probability of correct response (pC) for one reference ellipse
pC_one_ref = pC_Weibull_one_ref_stim(weibull_params, xy_transformed_one_ref)

# Compute pC for all ellipses using the Weibull psychometric function
pC = pC_Weibull(weibull_params, xy_transformed)

# Generate random binary outcomes (y) for comparison based on the probabilities (pC)
y = (jax.random.uniform(KEY, shape=pC.shape) < pC).astype(int)

# Package data for likelihood estimation (observed responses, reference points, and jittered comparison points)
data = (y, xy_ref, xy_jitter)

# Estimate the negative log likelihood of the model with initial parameters
meanLL = estimate_nloglikelihood(init_params, data)
print(meanLL)  # Print the initial mean log likelihood

# Stack ground truth parameters (ellipse + Weibull) for comparison
gt_params = np.hstack((par_all, np.tile(weibull_params[np.newaxis], (4, 1))))

# Estimate negative log likelihood for the ground truth parameters
meanLL_gt = estimate_nloglikelihood(gt_params, data)
print(meanLL_gt)  # Print the mean log likelihood for ground truth parameters

#%%
# Run optimization to recover the best-fit parameters
params_recover, iters, objhist = optimize_nloglikelihood(
    init_params, data, 
    total_steps=10000,           # Number of total optimization steps
    save_every=10,               # Save the objective value every 10 steps
    learning_rate = 1e-1,
    fixed_weibull_params=weibull_params,  # Fix the Weibull parameters during optimization
    show_progress=True           # Show progress using tqdm
)

# Plot the optimization history (objective value vs iterations)
fig3, ax3 = plt.subplots(1, 1)
ax3.plot(iters, objhist)  # Plot iterations vs objective history
fig3.tight_layout()

#%%
# Final visualization: compare recovered ellipses with ground truth ellipses
fig, ax = plt.subplots(1, 4, figsize=(8, 2))  # 4 subplots, one for each ellipse
for i in range(4):
    # Plot original (ground truth) ellipses in red
    ax[i].plot(xy_comp[i,:,0], xy_comp[i,:,1], c='r', lw=3, label = 'ground truth')
    
    # Scatter plot of jittered comparison points
    ax[i].scatter(xy_jitter[i,:,0], xy_jitter[i,:,1], s=2, c='gray', alpha=0.1, label = 'sim data')
    
    # Plot the reference point (center of the ellipse)
    ax[i].scatter(xy_ref[i][0], xy_ref[i][1], c='r', marker='+', label = 'ref stimulus')
    ax[i].set_aspect('equal', 'box')  # Ensure square aspect ratio for each plot

# Recover ellipses from the optimized parameters
xy_recover = np.full(xy_comp.shape, np.nan)  # Initialize array to store recovered ellipses
for i in range(nRefs):
    # Reconstruct the recovered ellipses using the optimized parameters
    x_recover, y_recover = PointsOnEllipseQ(*params_recover[i,0:2],  # Semi-major and semi-minor axes
                                   jnp.rad2deg(params_recover[i,2]),  # Convert angle back to degrees
                                   *xy_ref[i], nTheta=nTrials)       # Center of ellipse, number of points
    
    xy_recover[i] = jnp.vstack((x_recover, y_recover)).T  # Store recovered ellipse points

    # Plot recovered ellipses in green on top of the original plot
    ax[i].plot(x_recover, y_recover, c='g', label = 'model fits')
    
ax[-1].legend(loc='upper right', fontsize = 8, bbox_to_anchor=(2, 1))

