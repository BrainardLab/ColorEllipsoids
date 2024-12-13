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
from analysis.color_thres import color_thresholds
from plotting.wishart_predictions_plotting import WishartPredictionsVisualization
from analysis.ellipses_tools import UnitCircleGenerate

#%% load file
base_dir = '/Volumes/T9/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'
plane_2D = 'RB plane'
flag_load_pilot = False
if flag_load_pilot:
    fileDir_fits = base_dir +f'META_analysis/ModelFitting_DataFiles/2dTask/pilot/'
    figDir_fits = base_dir +f'META_analysis/ModelFitting_FigFiles/2dTask/pilot/'
    subj = 1
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
else:
    figDir_fits = base_dir + 'ELPS_analysis/ModelFitting_FigFiles/Python_version/2D_oddity_task/'

    #file 1
    path_str   = base_dir + 'ELPS_analysis/Simulation_DataFiles/'
    # Create an instance of the class
    color_thres_data = color_thresholds(2, 
                                        base_dir + 'ELPS_analysis/',
                                        plane_2D = plane_2D)
    # Retrieve specific data from Wishart_data
    color_thres_data.load_model_fits()
    model_pred_Wishart  = color_thres_data.get_data('model_pred_Wishart', dataset = 'Wishart_data')
    model = model_pred_Wishart.model
    W_est = model_pred_Wishart.W_est
    grid_trans = color_thres_data.get_data('grid_trans', dataset = 'Wishart_data')
    grid = color_thres_data.get_data('grid', dataset = 'Wishart_data')
    sim_trial_by_CIE = color_thres_data.get_data('sim_trial_by_CIE', dataset = 'Wishart_data')
    num_grid_pts = 5#model_pred_Wishart.num_grid_pts
    color_thres_data.load_CIE_data()
    results = color_thres_data.get_data('results2D', dataset ='CIE_data')

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
U_grid = model.compute_U(W_est, grid)
Sigmas_grid = model.compute_Sigmas(U_grid)

# ---------------------------
# Run ODE Solver for Geodesic
# ---------------------------

# Define initial position and velocity
x0 = jnp.array([0, 0])
nStarts = 40
v0 = 3*jnp.array(UnitCircleGenerate(nStarts))

#initialize
geo_path = []
# Compute geodesic and determine final location
for i in range(nStarts):
    geo_path_i = exponential_map(x0, v0[:,i], odeterm, odesolver)
    geo_path.append(geo_path_i)
    #x1 = geo_path[-1]
    
    # # Euclidean shortest path for comparison
    # euc_path = jnp.column_stack(
    #     [jnp.linspace(x0[0], x1[0]), jnp.linspace(x0[1], x1[1])]
    # )

#%%
wishart_pred_vis = WishartPredictionsVisualization(sim_trial_by_CIE,
                                                   model, 
                                                   model_pred_Wishart, 
                                                   color_thres_data,
                                                   fig_dir = figDir_fits, 
                                                   save_fig = True)

fig, ax = plt.subplots(1, 1, figsize = (3.1,3.6), dpi= 256)
geo_bds =[-0.9, 0.9]
for i in range(nStarts):
    #ax.plot(euc_path[:, 0], euc_path[:, 1], lw=1, color="k")
    geo_path_i = geo_path[i]
    rows_slc = (geo_path_i[:, 0] > geo_bds[0]) & (geo_path_i[:, 0] < geo_bds[1]) & \
        (geo_path_i[:, 1] > geo_bds[0]) & (geo_path_i[:, 1] < geo_bds[1])
    geo_path_i_trim = geo_path_i[rows_slc]
    ax.plot(geo_path_i_trim[:, 0], geo_path_i_trim[:, 1], lw=1, color="gray", alpha = 0.75)

#ground truth ellipses
gt_covMat_CIE = color_thres_data.N_unit_to_W_unit(results['fitEllipse_scaled'][color_thres_data.fixed_color_dim])
wishart_pred_vis.plot_2D(
    grid, 
    grid,
    gt_covMat_CIE, 
    ax = ax,
    visualize_samples= False,
    visualize_gt = False,
    visualize_model_estimatedCov = False,
    samples_alpha = 1,
    samples_s = 1,
    plane_2D = plane_2D,
    modelpred_ls = '-',
    modelpred_lw = 2,
    modelpred_lc = None,
    modelpred_alpha = 0.8,
    gt_lw= 0.5,
    gt_lc =[0.1,0.1,0.1],
    fontsize = 8.5,
    fig_name = f"Geodesics_{plane_2D}.png") 
        

#%% ---------------
# Plot the Result
# ---------------
fig, ax = plt.subplots(1, 1)
for i in range(num_grid_pts):
    for j in range(num_grid_pts):
        viz.plot_ellipse(
            ax, grid[i, j], 2.56 * Sigmas_grid[i, j], color="k", alpha=.5, lw=2)
#ax.plot(euc_path[:, 0], euc_path[:, 1], lw=2, color="b")
ax.plot(geo_path[:, 0], geo_path[:, 1], lw=1, color="r")
ax.set_aspect('equal', adjustable='box')
ax.set_xticks(np.linspace(-0.75, 0.75,5))
ax.set_yticks(np.linspace(-0.75, 0.75,5))
fig.tight_layout()
plt.show()