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

sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids')
from core import geodesics
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

#%% load file
base_dir = '/Volumes/T9/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'
plane_2D = 'isoluminant_plane'
color_dim = 2
flag_load_pilot = True
# Create an instance of the class
color_thres_data = color_thresholds(color_dim, 
                                    base_dir + 'ELPS_analysis/',
                                    plane_2D = plane_2D)
if flag_load_pilot:
    #select the data of a pilot subject
    file_name = color_thres_data._file_selection_popup()
    #find the full path
    full_path = color_thres_data._find_exact_path(file_name) 
    with open(full_path, 'rb') as f: vars_dict = pickled.load(f)
    #for var_name, var_value in vars_dict.items():
    #    locals()[var_name] = var_value
    
    #model predictions
    model_pred_Wishart = vars_dict['model_pred_Wishart']
    #wishart model
    model = model_pred_Wishart.model
    #grid points
    grid = vars_dict['grid']
    #best-fit weight matrix
    W_est = model_pred_Wishart.W_est
    #unpack data
    y_jnp, xref_jnp, x1_jnp = vars_dict['data']
    #reformat them to an object
    expt_trial = expt_data(xref_jnp, x1_jnp, y_jnp, None)
    #performance field (threshold contours that correspond to 66.7% correct response)
    performance_field = color_thres_data.N_unit_to_W_unit(model_pred_Wishart.fitEll_scaled)
    color_thres_data.fixed_color_dim =0
    #output figure directory
    figDir_fits = full_path[:full_path.rfind('/') + 1]
    figDir_fits = figDir_fits.replace('Experiment_DataFiles', 'Experiment_FigFiles')

else:
    figDir_fits = base_dir + 'ELPS_analysis/ModelFitting_FigFiles/Python_version/2D_oddity_task/'

    #file 1
    path_str   = base_dir + 'ELPS_analysis/Simulation_DataFiles/'
    # Retrieve specific data from Wishart_data
    color_thres_data.load_model_fits()
    model_pred_Wishart  = color_thres_data.get_data('model_pred_Wishart', dataset = 'Wishart_data')
    model = model_pred_Wishart.model
    W_est = model_pred_Wishart.W_est
    grid_trans = color_thres_data.get_data('grid_trans', dataset = 'Wishart_data')
    grid = color_thres_data.get_data('grid', dataset = 'Wishart_data')
    expt_trial = color_thres_data.get_data('sim_trial_by_CIE', dataset = 'Wishart_data')
    color_thres_data.load_CIE_data()
    results = color_thres_data.get_data('results2D', dataset ='CIE_data')
    performance_field = color_thres_data.N_unit_to_W_unit(results['fitEllipse_scaled'][color_thres_data.fixed_color_dim])

#%% -------------------------------
# Visualize
#-------------------------------
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
    
    
    fig, ax = plt.subplots(1, 1, figsize = (3.1,3.6), dpi= 1024)
    geo_bds =[-0.9, 0.9]
    for i in range(nStarts):
        #ax.plot(euc_path[:, 0], euc_path[:, 1], lw=1, color="k")
        geo_path_i = geo_path[i]
        rows_slc = (geo_path_i[:, 0] > geo_bds[0]) & (geo_path_i[:, 0] < geo_bds[1]) & \
            (geo_path_i[:, 1] > geo_bds[0]) & (geo_path_i[:, 1] < geo_bds[1])
        geo_path_i_trim = geo_path_i[rows_slc]
        if i == 0: label = 'Geodesics'; 
        else: label = None
        ax.plot(geo_path_i_trim[:, 0], geo_path_i_trim[:, 1], lw=1, color="gray", 
                alpha = 0.75, label = label)
    #ground truth ellipses
    wishart_pred_vis.plot_2D(
        grid, 
        grid,
        performance_field, 
        ax = ax,
        visualize_samples= False,
        visualize_gt = False,
        visualize_model_estimatedCov = False,
        flag_rescale_axes_label = False,
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
    ax.set_title('Isoluminant plane', fontsize = 10)    
    ax.set_xlabel('Wishart dimension 1')
    ax.set_ylabel('Wishart dimension 2')   

