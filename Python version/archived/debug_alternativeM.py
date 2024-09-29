#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 17:26:19 2024

@author: fangfang
"""

import jax
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
import sys
import numpy as np
import jax.numpy as jnp
sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids')
from core.probability_surface import IndividualProbSurfaceModel, unrotate_unstretch_one_ref_stim,\
    unrotate_unstretch, pC_Weibull_one_ref_stim, pC_Weibull,estimate_nloglikelihood,\
        optimize_nloglikelihood
from analysis.ellipses_tools import PointsOnEllipseQ

#%%
model_indv = IndividualProbSurfaceModel(2, 
                                        [1e-6,0.03], 
                                        [0, 360], 
                                        [0.5, 5], 
                                        [0.1, 5])
KEY   = jax.random.PRNGKey(227)
init_params = model_indv.sample_params_prior(KEY)

xy0_par1 = jnp.array([0, 0])
par1 = np.array([0.5, 1.5, 45])
xy0_par2 = jnp.array([1,1])
par2 = np.array([1, 1, 10])
xy0_par3 = jnp.array([0,1])
par3 = np.array([1.5, 1.6, 30])
xy0_par4 = jnp.array([1,0])
par4 = np.array([2, 2, -70])

x1,y1 = PointsOnEllipseQ(*par1, *xy0_par1)
x2,y2 = PointsOnEllipseQ(*par2, *xy0_par2)
x3,y3 = PointsOnEllipseQ(*par3, *xy0_par3)
x4,y4 = PointsOnEllipseQ(*par4, *xy0_par4)

xy1 = jnp.vstack((x1,y1)).T
xy2 = jnp.vstack((x2,y2)).T
xy3 = jnp.vstack((x3,y3)).T
xy4 = jnp.vstack((x4,y4)).T

xy_transformed_one_ref = unrotate_unstretch_one_ref_stim(par1, xy0_par1, xy1)

ell_params_batch = np.stack((par1, par2, par3, par4))

# Stack the xy_coords for both reference locations
xy_coords_batch = np.stack((xy1, xy2, xy3, xy4))
xy_ref = np.stack((xy0_par1, xy0_par2, xy0_par3, xy0_par4))

# Now call unrotate_unstretch with the properly formatted inputs
xy_transformed = unrotate_unstretch(ell_params_batch, 
                                    xy_ref, 
                                    xy_coords_batch)
plt.plot(xy_transformed[3,:,0], xy_transformed[3,:,1])
#plt.plot(xy_transformed[1,:,0], xy_transformed[1,:,1])

#%%
a = 1.17
b = 2.33
weibull_params = np.array([a,b])
pC_one_ref = pC_Weibull_one_ref_stim(weibull_params, xy_transformed_one_ref)
pC = pC_Weibull(weibull_params, 
                xy_transformed + jax.random.normal(KEY, shape=xy_transformed.shape)*0.3)

# Generate random values using JAX and compare with pC
y = (jax.random.uniform(KEY, shape=pC.shape) < pC).astype(int)
data = (y, xy_ref, xy_coords_batch)
meanLL =  estimate_nloglikelihood(init_params, data)

#%%
opt_params = {
    "learning_rate": 1e-3,#1e-2
    "momentum": 0.2,
}
    
params_recover, iters, objhist = optimize_nloglikelihood(
    init_params, data, 
    opt_params,
    total_steps=100,
    save_every=1,
    show_progress=True
)

fig, ax = plt.subplots(1, 1)
ax.plot(iters, objhist)
fig.tight_layout()














