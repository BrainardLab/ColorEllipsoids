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
    color_thres_data.fixed_color_dim = 0  # Fix the color dimension for further analysis
    color_thres_data.varying_RGBplane = [1,2]

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

for s in range(x0_all.shape[0]):
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
        geo_path_i = geodesics.exponential_map(x0, v0[:,i], odeterm, odesolver)
        geo_path.append(geo_path_i)
        #x1 = geo_path[-1]
        
        # # Euclidean shortest path for comparison
        # euc_path = jnp.column_stack(
        #     [jnp.linspace(x0[0], x1[0]), jnp.linspace(x0[1], x1[1])]
        # )
        
    geo_path_all_x0.append(geo_path)
    
    #plot the geodesics along with model predictions
    fig, ax = plt.subplots(1, 1, figsize = (3.1,3.6), dpi= 1024)
    geo_bds =[-0.9, 0.9]
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
        visualize_model_estimatedCov = False,
        flag_rescale_axes_label = True,
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
        fig_name = f"Geodesics_{plane_2D}_startingPoint[{x0[0]:.1f}, {x0[1]:.1f}].pdf") 
    #ax.set_title(plane_2D, fontsize = 10)    
    #ax.set_xlabel('Wishart dimension 1')
    #ax.set_ylabel('Wishart dimension 2')   
    
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
idx_x0 = 6
idx_path = [32, 36]
geo_path_trim_slc = geo_path_trim_all_x0[idx_x0]

#plot the geodesics along with model predictions
fig2, ax2 = plt.subplots(1, 1, figsize = (3.1,3.6), dpi= 1024)
geo_bds =[-0.9, 0.9]
for i in idx_path:
    if i == 0: label = 'Geodesics'; 
    else: label = None
    ax2.plot(geo_path_trim_slc[i][:, 0], geo_path_trim_slc[i][:, 1], lw=1, color="gray", 
            alpha = 0.75, label = label)
    
    # -----------------------------
    # Rocover covariance matrices
    # -----------------------------
    # Specify grid over stimulus space
    ref_geopath = np.transpose(geo_path_trim_slc[i][::100,:][jnp.newaxis,:,:], (1,0,2))
    Sigmas_ref_geopath = model.compute_Sigmas(model.compute_U(W_est, ref_geopath))
    
    # -----------------------------
    # Compute model predictions
    # -----------------------------
    model_pred_Wishart_geopath = wishart_model_pred(model, model_pred_Wishart.opt_params, 
                                            6, 2000, 
                                            5e-3,
                                            model_pred_Wishart.w_init_key,
                                            model_pred_Wishart.opt_key, 
                                            model_pred_Wishart.W_init, 
                                            W_est, Sigmas_ref_geopath, 
                                            color_thres_data,
                                            target_pC= 0.667,
                                            ngrid_bruteforce = 500,
                                            bds_bruteforce = [0.01, 0.25])
    
    ref_geopath_trans = np.transpose(ref_geopath,(2,0,1))
    model_pred_Wishart_geopath.convert_Sig_Threshold_oddity_batch(ref_geopath_trans)
    performance_field_geopath = color_thres_data.N_unit_to_W_unit(
        model_pred_Wishart_geopath.fitEll_scaled
    )

#ground truth ellipses
wishart_pred_vis.plot_2D(
    ref_geopath, 
    ref_geopath,
    performance_field_geopath, 
    ax = ax2,
    visualize_samples= False,
    visualize_gt = False,
    visualize_model_estimatedCov = False,
    flag_rescale_axes_label = True,
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
    fig_name = f"Geodesics_{plane_2D}_startingPoint[{x0[0]:.1f}, {x0[1]:.1f}].pdf") 
#ax.set_title(plane_2D, fontsize = 10)    





