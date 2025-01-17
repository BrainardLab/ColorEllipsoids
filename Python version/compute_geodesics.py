#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 10:37:24 2024

@author: fangfang
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
from diffrax import ODETerm, Dopri5 
import dill as pickled
import sys
import numpy as np

sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids')
from core import geodesics
from core.model_predictions import wishart_model_pred
from analysis.color_thres import color_thresholds
from plotting.wishart_predictions_plotting import WishartPredictionsVisualization
from analysis.ellipses_tools import UnitCircleGenerate

class expt_data:
    """A class to encapsulate experimental trial data."""
    def __init__(self, xref_all, x1_all, y_all, pseudo_order=None):
        self.xref_all = xref_all  # Reference stimuli for all trials
        self.x1_all = x1_all  # Comparison stimuli for all trials
        self.y_all = y_all  # Responses (e.g., correct/incorrect)
        self.pseudo_order = pseudo_order  # Optional: Pseudo-random trial order

#%% Load configuration and data
# Base directory for the analysis files
base_dir = '/Volumes/T9/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'

# Dimensionality of the color space (e.g., 2D or 3D)
color_dim = 2

# Flag to determine whether to load pilot data or simulated data
flag_load_pilot = True

# Specify the plane for color thresholds (e.g., "RB plane")
plane_2D = 'isoluminant plane'#'RG plane'

# Create an instance of the `color_thresholds` class
# This initializes the analysis environment with the specified color dimension and plane
color_thres_data = color_thresholds(
    color_dim,
    base_dir + 'ELPS_analysis/',
    plane_2D=plane_2D
)

if flag_load_pilot:
    # Load pilot data
    # Use a popup dialog to select the file containing pilot subject data
    file_name = color_thres_data._file_selection_popup()
    
    # Find the full file path of the selected file
    full_path = color_thres_data._find_exact_path(file_name)
    
    # Load the pilot data variables from the selected file
    with open(full_path, 'rb') as f:
        vars_dict = pickled.load(f)
    
    # Extract relevant data from the loaded file
    model_pred_Wishart = vars_dict['model_pred_Wishart'] # Model predictions
    model = model_pred_Wishart.model                     # Wishart model object
    grid = vars_dict['grid']                             # Grid points used in the analysis
    W_est = model_pred_Wishart.W_est                     # Best-fit weight matrix
    
    # Unpack experimental trial data (responses and stimuli)
    y_jnp, xref_jnp, x1_jnp = vars_dict['data']
    
    # Reformat the unpacked data into a structured object
    expt_trial = expt_data(xref_jnp, x1_jnp, y_jnp, None)
    
    # Compute the performance field (threshold contours at 66.7% correct response)
    performance_field = color_thres_data.N_unit_to_W_unit(
        model_pred_Wishart.fitEll_scaled
    )
    #color_thres_data.fixed_color_dim = 0  # Fix the color dimension for further analysis
    #color_thres_data.varying_RGBplane = [1,2]
    color_thres_data = vars_dict['color_thres_data']

    # Define the output figure directory based on the loaded file path
    figDir_fits_temp = full_path[:full_path.rfind('/') + 1]
    figDir_fits = figDir_fits_temp.replace('Experiment_DataFiles', 'Experiment_FigFiles')

else:    
    # Load model fits from the Wishart dataset
    color_thres_data.load_model_fits()
    
    # Retrieve specific data from the Wishart dataset
    model_pred_Wishart = color_thres_data.get_data(
        'model_pred_Wishart', dataset='Wishart_data'
    )
    grid = color_thres_data.get_data('grid', dataset='Wishart_data')
    expt_trial = color_thres_data.get_data(
        'sim_trial_by_CIE', dataset='Wishart_data'
    )
    
    # Extract model and best-fit weight matrix
    model = model_pred_Wishart.model
    W_est = model_pred_Wishart.W_est

    # Load CIE data for the simulation
    color_thres_data.load_CIE_data()
    results = color_thres_data.get_data(
        'results2D', dataset='CIE_data'
    )
    
    # Compute the performance field using the fitted ellipse scaled for 2D data
    performance_field = color_thres_data.N_unit_to_W_unit(
        results['fitEllipse_scaled'][color_thres_data.fixed_color_dim]
    )

    # Define the output figure directory for simulated data
    figDir_fits = base_dir + 'ELPS_analysis/ModelFitting_FigFiles/Python_version/2D_oddity_task/'

#%% Visualize
wishart_pred_vis = WishartPredictionsVisualization(expt_trial,
                                                   model, 
                                                   model_pred_Wishart, 
                                                   color_thres_data,
                                                   fig_dir = figDir_fits, 
                                                   save_fig = False)

#%% loop through different starting points
# Define initial position and velocity
x0_temp = grid[::2,:,:][:,::2,:] #jnp.array([0.6, 0.6])
x0_all = x0_temp.reshape(9, 2)
nStarts = 40
v0 = 6*jnp.array(UnitCircleGenerate(nStarts))
geo_path_all_x0 = []
geo_path_trim_all_x0 = []
geo_bds =[-0.9, 0.9]

for s in range(x0_all.shape[0]): #x0_all.shape[0]
    # ---------------------------
    # Run ODE Solver for Geodesic
    # ---------------------------
    x0 = x0_all[s]
    #initialize
    geo_path = []
    
    # Constants describing simulation
    odeterm = ODETerm(geodesics.geodesic_vector_field(
        lambda x: jnp.linalg.inv(model.compute_Sigmas(model.compute_U(W_est, x)))
    ))
    
    odesolver = Dopri5()
    # Compute geodesic and determine final location
    for i in range(nStarts):
        geo_path_i = geodesics.exponential_map(x0, v0[:,i], odeterm, odesolver, dt0 = 0.001)
        geo_path.append(geo_path_i)
        #x1 = geo_path[-1]
        
        # # Euclidean shortest path for comparison
        # euc_path = jnp.column_stack(
        #     [jnp.linspace(x0[0], x1[0]), jnp.linspace(x0[1], x1[1])]
        # )
        
    geo_path_all_x0.append(geo_path)
    
    #plot the geodesics along with model predictions
    fig, ax = plt.subplots(1, 1, figsize = (3.1,3.6), dpi= 1024)
    geo_path_trim = []
    for i in range(nStarts):
        #ax.plot(euc_path[:, 0], euc_path[:, 1], lw=1, color="k")
        geo_path_i = geo_path[i]
        rows_slc = (geo_path_i[:, 0] > geo_bds[0]) & (geo_path_i[:, 0] < geo_bds[1]) & \
            (geo_path_i[:, 1] > geo_bds[0]) & (geo_path_i[:, 1] < geo_bds[1])
        geo_path_i_trim = geo_path_i[rows_slc]
        geo_path_trim.append(geo_path_i_trim)
        if i == 0: label = 'Geodesics'; 
        else: label = None
        ax.plot(geo_path_i_trim[:, 0], geo_path_i_trim[:, 1], lw=1, color="gray", 
                alpha = 0.75, label = label)
    geo_path_trim_all_x0.append(geo_path_trim)    
    
    #ground truth ellipses
    wishart_pred_vis.plot_2D(
        grid, 
        grid,
        performance_field, 
        ax = ax,
        visualize_samples= False,
        visualize_gt = False,
        visualize_model_estimatedCov = True,
        flag_rescale_axes_label = False,
        samples_alpha = 1,
        samples_s = 1,
        plane_2D = plane_2D,
        modelpred_ls = '-',
        modelpred_lw = 1.5,
        modelpred_lc = None,
        modelpred_alpha = 0.8,
        fontsize = 8.5) 
    if plane_2D == 'isoluminant plane':
        ax.set_title(plane_2D, fontsize = 10)    
        ax.set_xlabel('Wishart dimension 1')
        ax.set_ylabel('Wishart dimension 2')   
    wishart_pred_vis._save_figure(fig,f"Geodesics_{plane_2D}_startingPoint[{x0[0]:.1f}, {x0[1]:.1f}].pdf") 
    
#%%
# x0_test = jnp.array([-0.6,0.6])
# x1_test = jnp.array([0.6, 0.6])

# def P(x):
#     return jnp.linalg.inv(
#         model.compute_Sigmas(model.compute_U(W_est, x))
#     )

# best_v0, simulator = geodesics.estimate_geodesic(
#     P, x0_test, x1_test, jax.random.PRNGKey(0), dt0=0.1, num_restarts=10, tol=1e-1
# )

#%% select two geodesic paths
"""
There are many different initial locations and velocities.
We will pick 1 example location (idx_x0), and three motion directions
"""
idx_x0 = 6
idx_path = [30, 35, 39]
skipping_step = 38
geo_path_trim_slc = geo_path_trim_all_x0[idx_x0]
model_pred_Wishart_geopath_all = []
ref_geopath_all = []

#plot the geodesics along with model predictions
for i in range(len(idx_path)):
    # -----------------------------
    # Rocover covariance matrices
    # -----------------------------
    # Specify grid over stimulus space
    ref_geopath = np.transpose(geo_path_trim_slc[idx_path[i]][::skipping_step,:][jnp.newaxis,:,:], (1,0,2))
    ref_geopath_all.append(ref_geopath)
    Sigmas_ref_geopath = model.compute_Sigmas(model.compute_U(W_est, ref_geopath_all[i]))
    
    # -----------------------------
    # Compute model predictions
    # -----------------------------
    model_pred_Wishart_geopath = wishart_model_pred(model, model_pred_Wishart.opt_params, 
                                            model_pred_Wishart.w_init_key,
                                            model_pred_Wishart.opt_key, 
                                            model_pred_Wishart.W_init, 
                                            W_est, Sigmas_ref_geopath, 
                                            color_thres_data,
                                            target_pC= 0.667,
                                            ngrid_bruteforce = 500,
                                            bds_bruteforce = [0.01, 0.25])
    model_pred_Wishart_geopath_all.append(model_pred_Wishart_geopath)
    
    #ref_geopath_trans = np.transpose(ref_geopath_all[i],(2,0,1))
    model_pred_Wishart_geopath_all[i].convert_Sig_Threshold_oddity_batch(np.transpose(ref_geopath_all[i],(2,0,1)))

#%%
fig2, ax2 = plt.subplots(1, 1, figsize = (3.1,3.6), dpi= 1024)
for i in idx_path:
    if i == 0: label = 'Geodesics'; 
    else: label = None
    ax2.plot(geo_path_trim_slc[i][:, 0], geo_path_trim_slc[i][:, 1], lw=1, color="gray", 
            alpha = 0.75, label = label)
    
for i in range(len(idx_path)):
    wishart_pred_geopath_vis = WishartPredictionsVisualization(expt_trial,
                                                       model, 
                                                       model_pred_Wishart_geopath_all[i], 
                                                       color_thres_data,
                                                       fig_dir = figDir_fits, 
                                                       save_fig = False)
        
    #ground truth ellipses
    wishart_pred_geopath_vis.plot_2D(
        ref_geopath_all[i], 
        ref_geopath_all[i],
        model_pred_Wishart_geopath_all[i].fitEll_unscaled, 
        ax = ax2,
        visualize_samples= False,
        visualize_gt = False,
        visualize_model_estimatedCov = True,
        flag_rescale_axes_label = False,
        samples_alpha = 1,
        samples_s = 1,
        plane_2D = plane_2D,
        modelpred_ls = '-',
        modelpred_lw = 1.5,
        modelpred_lc = None,
        modelpred_alpha = 0.8,
        fontsize = 8.5) 
if plane_2D == 'isoluminant plane':
    ax2.set_title(plane_2D, fontsize = 10)    
    ax2.set_xlabel('Wishart dimension 1')
    ax2.set_ylabel('Wishart dimension 2')  
tks = np.around(np.linspace(-0.6, 0.6, 3),1)
ax2.set_xticks(tks) 
ax2.set_yticks(tks) 
ax2.set_xticklabels(tks) 
ax2.set_yticklabels(tks) 
ax2.get_legend().remove()
wishart_pred_geopath_vis._save_figure(fig2,f"Geodesics_{plane_2D}_startingPoint[{x0[0]:.1f}, {x0[1]:.1f}]_selectedPaths.pdf") 
    
#%% visualize 3D projection
grid_fine = jnp.stack(jnp.meshgrid(*[jnp.linspace(-0.9,0.9,100) for _ in range(model.num_dims)]), axis=-1)
Sigmas_ref_finegrid = model.compute_Sigmas(model.compute_U(W_est, grid_fine))

# Compute determinant for each 2x2 matrix
determinants = np.log(np.linalg.det(Sigmas_ref_finegrid))
# Compute trace for each 2x2 matrix
traces = np.trace(Sigmas_ref_finegrid, axis1=-2, axis2=-1)

# Extract the grid x and y coordinates
x_coords = grid_fine[..., 0]
y_coords = grid_fine[..., 1]

fig = plt.figure(figsize=(11, 18))
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D surface
surface = ax.plot_surface(x_coords, y_coords, determinants, 
                          cmap='viridis', linewidth=0.3,edgecolors=(0, 0, 0, 0.5), alpha=0.8)

# Add 2D projection on a fixed Z plane
fixed_z = np.min(determinants) - 3  # Fixed z-level for the projection
contour = ax.contourf(x_coords, y_coords, determinants, 
                      zdir='z', offset=fixed_z, cmap='viridis',
                      levels=50, alpha=0.01)
# Add ellipses on the projection
for i in range(model_pred_Wishart_geopath_all[1].fitEll_unscaled.shape[0]):
    cm = color_thres_data.M_2DWToRGB @ np.insert(ref_geopath_all[1][i], 2, 1)
    # Extract x and y coordinates for the ellipse
    ellipse_x = model_pred_Wishart_geopath_all[1].fitEll_unscaled[i, 0, 0]
    ellipse_y = model_pred_Wishart_geopath_all[1].fitEll_unscaled[i, 0, 1]
    
    # Plot the ellipse in 3D, keeping z fixed at the projection level
    ax.plot(ellipse_x, ellipse_y, zs=fixed_z, zdir='z', color=cm, alpha=0.8)

# Customize axis and labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Log Determinant')
ax.set_zlim(fixed_z, np.max(determinants))  # Ensure the Z-axis includes the projection

# Add color bar
fig.colorbar(surface, ax=ax, shrink=0.2, aspect=10, label='Log Determinant Value')
plt.tight_layout()
plt.show()













