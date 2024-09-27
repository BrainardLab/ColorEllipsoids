#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 10:37:24 2024

@author: fangfang
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import diffrax
from diffrax import ODETerm, Dopri5, SaveAt
import dill as pickled
import sys

sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids')
from core import viz

#%% load file
base_dir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'
fileDir_fits = base_dir +f'META_analysis/ModelFitting_DataFiles/2dTask/pilot/'
figDir_fits = base_dir +f'META_analysis/ModelFitting_FigFiles/2dTask/pilot/'

subj = 1
plane_2D = 'GB plane'
file_name = f'Fitted_isothreshold_{plane_2D}_sim360perRef_25refs_AEPsychSampling_'+\
            f'bandwidth0.005_pilot_sub{subj}.pkl'
full_path = f"{fileDir_fits}{file_name}"
with open(full_path, 'rb') as f: vars_dict = pickled.load(f)
#for var_name, var_value in vars_dict.items():
#    locals()[var_name] = var_value

model_pred_Wishart = vars_dict['model_pred_Wishart']
model = model_pred_Wishart.model
num_grid_pts = 10#model_pred_Wishart.num_grid_pts
W_est = model_pred_Wishart.W_est

#%% -------------------------------
# Functions to Compute Geodesics
# -------------------------------

def geodesic_vector_field(P):
    """
    Given a function `P(x)` that returns the inverse
    covariance (a.k.a. precision matrix) at location
    `x` in stimulus space, define a function that
    computes the vector field of geodesic flows:
    `dxdt` and `dvdt` are respectively the velocity
    and acceleration vectors along the geodesic path.
    """
    jacP = jax.jacobian(P)
    def vector_field(t, state, args):
        x, v = state
        Pdx = jacP(x)
        q1 = 0.5 * jnp.einsum("jki,j,k->i",Pdx, v, v)
        q2 = jnp.einsum("ilp,l,p->i", Pdx, v, v)
        dxdt = v
        dvdt = jnp.linalg.solve(P(x), q1 - q2)
        return (dxdt, dvdt)
    return vector_field

def exponential_map(x0, v0, odeterm, odesolver, dt0=0.001):
    """
    Compute the geodesic starting at position `x0` and with
    initial velocity `v0`. Uses package diffrax to numerically
    solve the ODE.
    """
    return diffrax.diffeqsolve(
        odeterm, odesolver, t0=0, t1=1, dt0=dt0, y0=(x0, v0),
        saveat=SaveAt(t0=True, t1=True, steps=True)
    ).ys[0]

#%% -------------------------------
# Constants describing simulation
# -------------------------------
odeterm = ODETerm(geodesic_vector_field(
    lambda x: jnp.linalg.inv(model.compute_Sigmas(model.compute_U(W_est, x)))
))
odesolver = Dopri5()

# -------------------------
# Sample ground truth model
# -------------------------
grid = jnp.stack(
    jnp.meshgrid(
        *[jnp.linspace(-1, 1, num_grid_pts + 2)[1:-1] for _ in range(model.num_dims)]
    ), axis=-1
)
U_grid = model.compute_U(W_est, grid)
Sigmas_grid = model.compute_Sigmas(U_grid)

# ---------------------------
# Run ODE Solver for Geodesic
# ---------------------------

# Define initial position and velocity
x0 = jnp.array([-0.75, -0.75])
v0 = jnp.array([1, 0])#jnp.array([2.4, 1.44])

# Compute geodesic and determine final location
geo_path = exponential_map(x0, v0, odeterm, odesolver)
x1 = geo_path[-1]

# Euclidean shortest path for comparison
euc_path = jnp.column_stack(
    [jnp.linspace(x0[0], x1[0]), jnp.linspace(x0[1], x1[1])]
)

#%% ---------------
# Plot the Result
# ---------------
fig, ax = plt.subplots(1, 1)
for i in range(num_grid_pts):
    for j in range(num_grid_pts):
        viz.plot_ellipse(
            ax, grid[i, j], 2.56 * Sigmas_grid[i, j], color="k", alpha=.5, lw=2)
ax.plot(euc_path[:, 0], euc_path[:, 1], lw=2, color="b")
ax.plot(geo_path[:, 0], geo_path[:, 1], lw=1, color="r")
ax.set_aspect('equal', adjustable='box')
ax.set_xticks(np.linspace(-0.75, 0.75,5))
ax.set_yticks(np.linspace(-0.75, 0.75,5))
fig.tight_layout()
plt.show()