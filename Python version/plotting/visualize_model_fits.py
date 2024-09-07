#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 22:04:36 2024

@author: fangfang
"""

import jax
jax.config.update("jax_enable_x64", True)
import dill as pickled
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids')
from analysis.color_thres import color_thresholds
from plotting.wishart_predictions_plotting import WishartPredictionsVisualization
from analysis.ellipses_tools import ellParamsQ_to_covMat, UnitCircleGenerate, PointsOnEllipseQ

#three variables we need to define for loading the data
nSims     = 240
jitter    = 0.3

base_dir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'
fileDir_fits = base_dir +'ELPS_analysis/ModelFitting_DataFiles/'
figDir_fits = base_dir +'ELPS_analysis/ModelFitting_FigFiles/Python_version/'

ndims = 2

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
    
    
    file_name = 'Fitted_isothreshold_ellipsoids_sim240perCond_samplingNearContour_jitter0.1_bandwidth0.005_oddity.pkl'
    full_path = f"{fileDir_fits}3D_oddity_task/{file_name}"
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
                                                         save_fig = True,
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
    #%% 2D
    #three variables we need to define for loading the data
    plane_2D      = 'RB plane'
    plane_2D_dict = {'GB plane': 0, 'RB plane': 1, 'RG plane': 2}
    plane_2D_idx  = plane_2D_dict[plane_2D]
    rnd_seed_list = list(range(9))
    # Create an instance of the class
    color_thres_data_2D = color_thresholds(2, base_dir + 'ELPS_analysis/', plane_2D = plane_2D)
    # Load Wishart model fits
    color_thres_data_2D.load_CIE_data()  
    
    file_name_2D = f'Fitted_isothreshold_{plane_2D}_sim{nSims}perCond_samplingNearContour_'+\
        f'jitter{jitter}'
        
    nTheta= 1000
    ell_all = np.full((5,5,2, nTheta, len(rnd_seed_list)), np.nan)
    unitC = UnitCircleGenerate(nTheta)
    for r in rnd_seed_list:
        file_name_r = file_name_2D + '_seed'+str(r) +'_bandwidth0.005_oddity.pkl'
        full_path = f"{fileDir_fits}2D_oddity_task/{file_name_r}"
        with open(full_path, 'rb') as f:  vars_dict_2D = pickled.load(f)
        for var_name, var_value in vars_dict_2D.items():
            locals()[var_name] = var_value
        param_ell_r = model_pred_Wishart.params_ell
        for i in range(5):
            for j in range(5):
                ref_ij = grid[i,j]
                params_ell_ij = param_ell_r[i][j]
                covMat_ij = ellParamsQ_to_covMat(params_ell_ij[2], params_ell_ij[3],params_ell_ij[4])
                L = np.linalg.cholesky(covMat_ij)
                ell_all[i,j,:,:,r] = L @ unitC + ref_ij[:,None]
                plt.plot(ell_all[i,j,0,:,r], ell_all[i,j,1,:,r])
                plt.scatter(ell_all[i,j,0,0,r], ell_all[i,j,1,0,r],c='r')
    
    fitEll_min = np.full((5,5,2,nTheta), np.nan)
    fitEll_max = np.full((5,5,2,nTheta), np.nan)
    for i in range(5):
        for j in range(5):
            ref_ij = grid[i,j]
            ell_all_ij = ell_all[i,j]
            vecLength_ijr = np.sqrt((ell_all_ij[0]-ref_ij[0])**2 + (ell_all_ij[1]-ref_ij[1])**2)
            sort_idx = np.argsort(vecLength_ijr, axis = -1)
            min_idx_ij = sort_idx[:,0]
            max_idx_ij = sort_idx[:,-1]
            fitEll_min[i,j] = ell_all_ij[np.arange(2)[:, None], np.arange(nTheta),min_idx_ij]
            fitEll_max[i,j]= ell_all_ij[np.arange(2)[:, None], np.arange(nTheta),max_idx_ij]



