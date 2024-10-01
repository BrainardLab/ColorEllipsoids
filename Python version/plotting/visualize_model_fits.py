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
from analysis.ellipses_tools import UnitCircleGenerate, find_inner_outer_contours, ellParams_to_covMat, covMat_to_ellParamsQ

##this new part below is needed since I updated jax
class CustomUnpickler(pickled.Unpickler):
    def find_class(self, module, name):
        # Skip shard_arg by returning a dummy function
        if name == 'shard_arg':
            return lambda *args, **kwargs: None  # Return a dummy function or None
        return super().find_class(module, name)

#three variables we need to define for loading the data
nSims     = 240
jitter    = 0.3
ndims     = 3
nTheta    = 1000
num_grid_pts = 5
rnd_seed_list = list(range(10))
fitting_method = ''#'indvEll'
if fitting_method == 'indvEll':
    str_ext = '_indvEll'
else:
    str_ext = ''
base_dir = '/Volumes/T9/Aguirre-Brainard Lab Dropbox/Fangfang Hong/ELPS_analysis/'
fileDir_fits = base_dir +f'ModelFitting_DataFiles/{ndims}D_oddity_task{str_ext}/'
figDir_fits = base_dir +f'ModelFitting_FigFiles/Python_version/{ndims}D_oddity_task{str_ext}/'

#%% ------------------------------------------
# Load data simulated using CIELab
# ------------------------------------------
if ndims == 3:
    #file 1
    path_str = base_dir + 'Simulation_DataFiles/'
    # Create an instance of the class
    color_thres_data = color_thresholds(3, base_dir)
    # Load Wishart model fits
    color_thres_data.load_CIE_data()
    
    #initialize
    covMat_all = np.full((len(rnd_seed_list), num_grid_pts, num_grid_pts,
                          num_grid_pts, ndims, ndims), np.nan)
    covMat_proj_all = np.full((len(rnd_seed_list), num_grid_pts, num_grid_pts,
                               num_grid_pts, ndims, ndims-1, ndims-1), np.nan)
    params_proj_all = np.full((num_grid_pts, num_grid_pts, num_grid_pts, ndims,
                               len(rnd_seed_list), 5), np.nan)
    covMat_slice_all = np.full(covMat_proj_all.shape, np.nan)
    params_slice_all = np.full(params_proj_all.shape, np.nan)
    for r in rnd_seed_list:
        file_name = f'Fitted_isothreshold_ellipsoids_sim{nSims}perCond_samplingNearContour'+\
            f'_jitter{jitter}_seed{r}_bandwidth0.005_oddity.pkl'
        full_path = f"{fileDir_fits}{file_name}"
        with open(full_path, 'rb') as f: 
            vars_dict = CustomUnpickler(f).load()
            #vars_dict = pickled.load(f)
        for var_name, var_value in vars_dict.items():
            locals()[var_name] = var_value
        params_ell_r = model_pred_Wishart.params_ell
        for i in range(num_grid_pts):
            for j in range(num_grid_pts):
                for k in range(num_grid_pts):
                    grid_ijk = np.array(grid[i,j,k])
                    params_ell_r_ijk = params_ell_r[j][i][k]
                    #check for matching
                    assert(np.sum(np.abs(params_ell_r_ijk['center'].T - grid_ijk)) < 0.1)
                    radii_ijk = params_ell_r_ijk['radii']
                    evecs_ijk = params_ell_r_ijk['evecs']
                    #convert ell parameters to covariance matrix
                    covMat_ijk = ellParams_to_covMat(radii_ijk, evecs_ijk)
                    covMat_all[r,i,j,k] = covMat_ijk
                    #retrieve 2D slices
                    covMat_slice_all[r,i,j,k] = model_pred_Wishart.pred_slice_2d_ellipse[j,i,k]
                    
                    #convert to 2D projections
                    for l in range(ndims):
                        slc_dims = list(range(ndims))
                        slc_dims.remove(l)
                        covMat_proj_all[r,i,j,k,l] = covMat_ijk[slc_dims][:,slc_dims]
                        _, _, axes_lengths, theta = covMat_to_ellParamsQ(covMat_proj_all[r,i,j,k,l])
                        params_proj_all[i,j,k,l,r] = np.hstack((grid_ijk[slc_dims],axes_lengths, theta))
                        #slices
                        _, _, axes_lengths_s, theta_s = covMat_to_ellParamsQ(covMat_slice_all[r,i,j,k,l])
                        params_slice_all[i,j,k,l,r] = np.hstack((grid_ijk[slc_dims],axes_lengths_s, theta_s))
                    
    #for each projection plane
    fitEll_min = np.full((num_grid_pts, num_grid_pts, num_grid_pts, ndims, 2, nTheta*2), np.nan)
    fitEll_max = np.full(fitEll_min.shape, np.nan)
    fitEll_s_min = np.full(fitEll_min.shape, np.nan)
    fitEll_s_max = np.full(fitEll_min.shape, np.nan)
    for l in range(ndims):
        for i in range(num_grid_pts):
            for j in range(num_grid_pts):
                for k in range(num_grid_pts):
                    params_ijkl = params_proj_all[i,j,k,l]
                    xu_ijkl, yu_ijkl, xi_ijkl, yi_ijkl = find_inner_outer_contours(params_ijkl)
                    idx_u = xu_ijkl.shape[0]
                    idx_i = xi_ijkl.shape[0]
                    fitEll_max[i,j,k,l,0,:idx_u] = xu_ijkl
                    fitEll_max[i,j,k,l,1,:idx_u] = yu_ijkl
                    fitEll_min[i,j,k,l,0,:idx_i] = xi_ijkl
                    fitEll_min[i,j,k,l,1,:idx_i] = yi_ijkl
                    
                    #do the same for slices
                    params_ijkl_s = params_slice_all[i,j,k,l]
                    xu_ijkl_s, yu_ijkl_s, xi_ijkl_s, yi_ijkl_s = \
                        find_inner_outer_contours(params_ijkl_s)
                    idx_u_s = xu_ijkl_s.shape[0]
                    idx_i_s = xi_ijkl_s.shape[0]
                    fitEll_s_max[i,j,k,l,0,:idx_u_s] = xu_ijkl_s
                    fitEll_s_max[i,j,k,l,1,:idx_u_s] = yu_ijkl_s
                    fitEll_s_min[i,j,k,l,0,:idx_i_s] = xi_ijkl_s
                    fitEll_s_min[i,j,k,l,1,:idx_i_s] = yi_ijkl_s                  
        
    #%%
    fig_name = f'Fitted{file_name[4:-4]}_wCI'
    wishart_pred_vis = WishartPredictionsVisualization(sim_trial_by_CIE,
                                                        model, 
                                                        model_pred_Wishart, 
                                                        color_thres_data,
                                                        fig_dir = figDir_fits + 'wCI/', 
                                                        save_fig = True,
                                                        save_gif = False)
    wishart_pred_vis.plot_3D(
        grid_trans, 
        grid_trans,
        gt_covMat_CIE, 
        gt_slice_2d_ellipse_CIE,
        visualize_model_pred = False,
        visualize_modelpred_CI = True,
        visualize_model_estimatedCov = False,
        visualize_gt_proj_slice_one_plot = True,
        modelpred_projection_CI = [np.transpose(fitEll_min,(1,0,2,3,4,5)),
                        np.transpose(fitEll_max,(1,0,2,3,4,5))],
        modelpred_slice_CI = [np.transpose(fitEll_s_min,(1,0,2,3,4,5)),
                        np.transpose(fitEll_s_max,(1,0,2,3,4,5))],
        fontsize = 12,
        gt_ls = '--',
        gt_lc = 'k',
        gt_lw = 1,
        gt_alpha = 0.85,
        gt_3Dproj_lc = [0.3,0.3,0.3],
        gt_3Dproj_lw = 1,
        gt_3Dproj_ls = '--',
        modelpred_alpha = 0.55,
        samples_alpha = 0.2,
        axes_samples = [0,1],
        fig_name = fig_name) 

else:
    #%%
    #three variables we need to define for loading the data
    plane_2D      = 'RB plane'
    plane_2D_dict = {'GB plane': 0, 'RB plane': 1, 'RG plane': 2}
    plane_2D_idx  = plane_2D_dict[plane_2D]
    # Create an instance of the class
    color_thres_data_2D = color_thresholds(2, base_dir + 'ELPS_analysis/', plane_2D = plane_2D)
    # Load Wishart model fits
    color_thres_data_2D.load_CIE_data()  
    
    #file_name_2D = f'Fitted_isothreshold_{plane_2D}_sim{nSims}perCond_samplingNearContour_'+\
    #    f'noFixedRef_jitter{jitter}'
    file_name_2D = f'Fitted_isothreshold_{plane_2D}_sim{nSims}perCond_samplingNearContour_'+\
        f'jitter{jitter}'
        
    params_all = np.full((num_grid_pts, num_grid_pts,len(rnd_seed_list), 5), np.nan)
    ell_all = np.full((num_grid_pts, num_grid_pts,2, nTheta, len(rnd_seed_list)), np.nan)
    unitC = UnitCircleGenerate(nTheta)
    for r in rnd_seed_list:
        file_name_r = file_name_2D + '_seed'+str(r) +'_oddity_indvEll.pkl'
        #file_name_r = file_name_2D + '_seed'+str(r) +'_bandwidth0.005_oddity.pkl'
        full_path = f"{fileDir_fits}{file_name_r}"
        with open(full_path, 'rb') as f:  vars_dict_2D = pickled.load(f)
        for var_name, var_value in vars_dict_2D.items():
            locals()[var_name] = var_value
        if fitting_method == 'indvEll':
            model = model_indvEll
            model_pred = model_pred_indvEll
            param_ell_r = model_pred_indvEll.params_ell
        else:
            model_pred = model_pred_Wishart
            param_ell_r = model_pred_Wishart.params_ell
        for i in range(num_grid_pts):
            for j in range(num_grid_pts):
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
                                                          model_pred, 
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
    # Removing 'seed9' and '.pkl'
    fig_name = file_name_r
    fig_name = fig_name.replace('_seed9', '').replace('.pkl', '')
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
        fig_name = fig_name +'_withSamples_withCI.pdf') 
            



