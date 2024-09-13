#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 22:04:36 2024

@author: fangfang
"""

import jax
jax.config.update("jax_enable_x64", True)
import dill as pickled
import pickle as pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.ops import unary_union

sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids')
from analysis.color_thres import color_thresholds
from plotting.wishart_predictions_plotting import WishartPredictionsVisualization
from analysis.ellipses_tools import UnitCircleGenerate, find_inner_outer_contours

#three variables we need to define for loading the data
nSims     = 240
jitter    = 0.1
ndims     = 3

base_dir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'
fileDir_fits = base_dir +f'ELPS_analysis/ModelFitting_DataFiles/{ndims}D_oddity_task/'
figDir_fits = base_dir +f'ELPS_analysis/ModelFitting_FigFiles/Python_version/{ndims}D_oddity_task/'

#%% ------------------------------------------
# Load data simulated using CIELab
# ------------------------------------------
if ndims == 3:
    #file 1
    path_str = base_dir + 'ELPS_analysis/Simulation_DataFiles/'
    # Create an instance of the class
    color_thres_data = color_thresholds(3, base_dir + 'ELPS_analysis/')
    # Load Wishart model fits
    color_thres_data.load_CIE_data()  
    
    
    file_name = f'Fitted_isothreshold_ellipsoids_sim{nSims}perCond_samplingNearContour'+\
        f'_jitter{jitter}_seed0_bandwidth0.005_oddity.pkl'
    full_path = f"{fileDir_fits}{file_name}"
    with open(full_path, 'rb') as f:  vars_dict = pickled.load(f)
    
    for var_name, var_value in vars_dict.items():
        locals()[var_name] = var_value
        
    #%%
    fig_name = 'Fitted' + file_name[4:-4]
    
    wishart_pred_vis = WishartPredictionsVisualization(sim_trial_by_CIE,
                                                         model, 
                                                         model_pred_Wishart, 
                                                         color_thres_data,
                                                         fig_dir = figDir_fits + '3D_oddity_task/', 
                                                         save_fig = False,
                                                         save_gif = False)
            
    wishart_pred_vis.plot_3D(
        grid_trans, 
        grid_trans,
        gt_covMat_CIE, 
        gt_slice_2d_ellipse_CIE,
        fontsize = 12,
        gt_ls = '--',
        gt_lw = 1,
        gt_alpha = 0.85,
        modelpred_alpha = 0.55,
        fig_name = fig_name) 

else:
    #%%
    #three variables we need to define for loading the data
    plane_2D      = 'RG plane'
    plane_2D_dict = {'GB plane': 0, 'RB plane': 1, 'RG plane': 2}
    plane_2D_idx  = plane_2D_dict[plane_2D]
    rnd_seed_list = list(range(10))
    # Create an instance of the class
    color_thres_data_2D = color_thresholds(2, base_dir + 'ELPS_analysis/', plane_2D = plane_2D)
    # Load Wishart model fits
    color_thres_data_2D.load_CIE_data()  
    
    file_name_2D = f'Fitted_isothreshold_{plane_2D}_sim{nSims}perCond_samplingNearContour_'+\
        f'jitter{jitter}'
        
    nTheta= 1000
    num_grid_pts = 5
    params_all = np.full((num_grid_pts, num_grid_pts,len(rnd_seed_list), 5), np.nan)
    ell_all = np.full((num_grid_pts, num_grid_pts,2, nTheta, len(rnd_seed_list)), np.nan)
    unitC = UnitCircleGenerate(nTheta)
    for r in rnd_seed_list:
        file_name_r = file_name_2D + '_seed'+str(r) +'_bandwidth0.005_oddity.pkl'
        full_path = f"{fileDir_fits}{file_name_r}"
        with open(full_path, 'rb') as f:  vars_dict_2D = pickled.load(f)
        for var_name, var_value in vars_dict_2D.items():
            locals()[var_name] = var_value
        param_ell_r = model_pred_Wishart.params_ell
        for i in range(num_grid_pts):
            for j in range(num_grid_pts):
                ref_ij = grid[i,j]
                params_all[i,j,r]= param_ell_r[i][j]
    
    fitEll_min = np.full((num_grid_pts, num_grid_pts, ndims,nTheta*2), np.nan)
    fitEll_max = np.full((num_grid_pts, num_grid_pts, ndims,nTheta*2), np.nan)
    for i in range(num_grid_pts):
        for j in range(num_grid_pts):
            params_ij = params_all[i,j]
            xu_ij, yu_ij, xi_ij, yi_ij = find_inner_outer_contours(params_ij)
            idx_u = xu_ij.shape[0]
            idx_i = xi_ij.shape[0]
            fitEll_max[i,j,0,:idx_u] = xu_ij
            fitEll_max[i,j,1,:idx_u] = yu_ij
            fitEll_min[i,j,0,:idx_i] = xi_ij
            fitEll_min[i,j,1,:idx_i] = yi_ij
            
    #visualize
    wishart_pred_vis_wCI = WishartPredictionsVisualization(sim_trial_by_CIE,
                                                          model, 
                                                          model_pred_Wishart, 
                                                          color_thres_data_2D,
                                                          fig_dir = figDir_fits, 
                                                          save_fig = True)
    fig, ax = plt.subplots(1, 1, figsize = (3.1,3.6), dpi= 256)
    for i in range(num_grid_pts):
        for j in range(num_grid_pts):
            #define the color map, which is the RGB value of the reference stimulus
            cm = color_thres_data_2D.W_unit_to_N_unit(grid[i, j])
            # Adjust the color map based on the fixed color dimension.
            cm = np.insert(cm, color_thres_data_2D.fixed_color_dim,
                           color_thres_data_2D.fixed_value)
            idx_max_nonan = ~np.isnan(fitEll_max[i, j, 0])
            ax.fill(fitEll_max[i,j,0,idx_max_nonan], fitEll_max[i,j,1,idx_max_nonan], 
                    color= cm)
            idx_min_nonan = ~np.isnan(fitEll_min[i, j, 0])
            ax.fill(fitEll_min[i,j,0,idx_min_nonan], fitEll_min[i,j,1,idx_min_nonan], 
                    color='white')
    wishart_pred_vis_wCI.plot_2D(
        grid, 
        grid,
        gt_covMat_CIE, 
        ax = ax,
        visualize_samples= True,
        visualize_gt = True,
        visualize_model_estimatedCov = False,
        visualize_model_pred = False,
        samples_alpha = 0.5,
        samples_s = 1,
        plane_2D = plane_2D,
        gt_lw= 1,
        gt_lc =[0.3,0.3,0.3],
        fig_name = file_name_r[:-31] + file_name_r[-25:-4] +'_withSamples_withCI.pdf') 
            



