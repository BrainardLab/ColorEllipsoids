#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 21:15:16 2024

@author: fangfang
"""
import jax
import sys
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids")
from core import chebyshev, viz, utils, oddity_task, optim, model_predictions
from core.wishart_process import WishartProcessModel

#%%
def simulate_oddityData_given_W_gt(M, W_gt, xGrid, nS, DATA_KEY, **kwargs):
    # Default parameters for ellipsoid fitting. Can be overridden by kwargs.
    params = {
        'noise_level':0.02,
        'numSD':2.46}  # 78% correct responses
    params.update(kwargs)
    
    data_keys = jax.random.split(DATA_KEY, num=4)
    
    #compute the total number of trials
    if M.num_dims == 2:
        num_grid_pts1, num_grid_pts2 = xGrid.shape[0], xGrid.shape[1]
        nSims_total = nS*num_grid_pts1*num_grid_pts2
    elif M.num_dims == 3:
        num_grid_pts1, num_grid_pts2, num_grid_pts3 = xGrid.shape[0], xGrid.shape[1], xGrid.shape[2]
        nSims_total = nS*num_grid_pts1*num_grid_pts2*num_grid_pts3
    
    #flatten xref
    xgridflat  = xGrid.reshape(-1, M.num_dims)
    ridx       = np.repeat(np.arange(xgridflat.shape[0]), nS)
    xref       = xgridflat[ridx]
    Sigmas_ref = M.compute_Sigmas(M.compute_U(W_gt, xref))

    x0          = jnp.copy(xref)
    _z          = jax.random.normal(data_keys[1], shape=(nSims_total, M.num_dims))
    noise_x1    = params['noise_level'] * jax.random.normal(\
                        data_keys[2], shape=(nSims_total, M.num_dims))
    x1          = x0 + jnp.einsum("ijk,ik->ij",params['numSD'] * utils.sqrtm(Sigmas_ref), \
                         _z / jnp.linalg.norm(_z, axis=1, keepdims=True)) + noise_x1
    
    x1          = jnp.clip(x1, -1.0, 1.0)
    Uref        = M.compute_U(W_gt, xref)
    U0          = M.compute_U(W_gt, x0)
    U1          = M.compute_U(W_gt, x1)
    y = 1 - jnp.maximum(0, jnp.sign(oddity_task.simulate_oddity((xref, x0, x1, Uref, U0, U1),
            jax.random.split(data_keys[2], num=nSims_total), 1, M.diag_term)).ravel())
    print("Proportion of correct trials:", jnp.mean(y))
    
    # Package data
    data = (y, xref, x0, x1)
    
    return data, noise_x1, Uref, U0, U1