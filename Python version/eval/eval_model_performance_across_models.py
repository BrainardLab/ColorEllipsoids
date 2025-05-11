#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 20:31:02 2024

@author: fangfang
"""

import jax
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
import sys
import os
import dill as pickled
import numpy as np
import itertools
#sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids')
sys.path.append("/Users/fh862-adm/Documents/GitHub/ellipsoids/ellipsoids")
from analysis.ellipses_tools import ellParams_to_covMat, rotAngle_to_eigenvectors, \
    PointsOnEllipseQ, find_inner_outer_contours, convert_2Dcov_to_points_on_ellipse,\
    fit_2d_isothreshold_contour
from analysis.ellipsoids_tools import  eig_to_covMat
from plotting.wishart_predictions_plotting import WishartPredictionsVisualization
#sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/Python version')
sys.path.append("/Users/fh862-adm/Documents/GitHub/ColorEllipsoids/Python version")
from analysis.model_performance import ModelPerformance, PltBWDSettings
from analysis.utils_load import select_file_and_get_path

COLOR_DIMENSION = 3

#%% ------------------------------------------
# Load data simulated using CIELab
# ------------------------------------------
# List to store loaded simulation data
nFiles_load_indvEll = 4
data_load_indvEll, input_fileDir_all1, file_name_all1, trial_num_indvEll, numDatasets_indvEll = [],[],[],[],[]
grid_pts_desired = 5
min_grid_desired = -0.7
for _ in range(nFiles_load_indvEll): #it can be any number depending on how many files we want to compare
    #2D
    #META_analysis/ModelFitting_DataFiles/2dTask_indvEll/CIE/sub2/subset14014'
    #'IndvFitted_byWishart_Isoluminant plane_49interleaved2DExpt_10_10_10_330_AEPsychSampling_EAVC_sub2_subset14014.pkl'

    #or 3D
    #'META_analysis/ModelFitting_DataFiles/3dTask_indvEll/CIE/sub1/subset30000'
    # 'IndvFitted_byWishart_ellipsoids_3DExpt_30_30_30_550_AEPsychSampling_EAVC_decayRate0.4_nBasisDeg5_sub1_subset30000.pkl'
    input_fileDir_fits1, file_name1 = select_file_and_get_path()
    input_fileDir_all1.append(input_fileDir_fits1)
    file_name_all1.append(file_name1)
    
    #extract trial number
    trial_num = int(file_name1.split('subset')[1].split('.pkl')[0])
    trial_num_indvEll.append(trial_num)
    
    # Construct the full path to the selected file
    full_path1 = os.path.join(input_fileDir_fits1, file_name1)
    
    # Load the necessary variables from the file
    with open(full_path1, 'rb') as f:
        vars_dict1 = pickled.load(f)
    
    NUM_GRID_PTS = vars_dict1['NUM_GRID_PTS']
    if NUM_GRID_PTS == grid_pts_desired and np.min(vars_dict1['grid']) == min_grid_desired:
        prefix = f'model_pred_indv_subset{trial_num}'
        matching_keys = [key for key in vars_dict1.keys() if key.startswith(prefix)]
        numDatasets_indvEll.append(len(matching_keys))
        
        if COLOR_DIMENSION == 2:
            model_ellParams1 = np.full((len(matching_keys), grid_pts_desired**COLOR_DIMENSION,5), np.nan)
            for idx, m in enumerate(matching_keys):
                model_ellParams1_m = np.reshape(vars_dict1[m]['fitEll_params'],
                                             (grid_pts_desired, grid_pts_desired, -1))
                model_ellParams1_reshape = np.reshape(model_ellParams1_m, (grid_pts_desired**COLOR_DIMENSION, -1))
                model_ellParams1[idx] = model_ellParams1_reshape
            data_load_indvEll.append(model_ellParams1.tolist())
        else:
            model_ellParams1 = []
            for m in matching_keys:
                model_ellParams1_m = vars_dict1[m]['fitEll_params']
                model_ellParams1.append(model_ellParams1_m)
            data_load_indvEll.append([model_ellParams1])
    else:
        raise ValueError('Cannot find the model predictions at desired grid points.')

#%% load one from the Wishart fits
nFiles_load_Wishart = 2
data_load_Wishart, input_fileDir_all2, file_name_all2, numDatasets_Wishart = [],[],[],[]
for _ in range(nFiles_load_Wishart): #it can be any number depending on how many files we want to compare
    #'META_analysis/ModelFitting_DataFiles/4dTask/CIE/sub1/decayRate0.5'
    #'Fitted_byWishart_Isoluminant plane_4DExpt_300_300_300_5100_AEPsychSampling_EAVC_decayRate0.5_nBasisDeg5_sub1.pkl'

    #'META_analysis/ModelFitting_DataFiles/2dTask/CIE/sub2/subset6000'
    #'Fitted_byWishart_Isoluminant plane_49interleaved2DExpt_10_10_10_330_AEPsychSampling_EAVC_sub2.pkl'
    
    #or 3D
    #'META_analysis/ModelFitting_DataFiles/6dTask/CIE'
    #'Fitted_byWishart_ellipsoids_6DExpt_1500_1500_1500_25500_AEPsychSampling_EAVC_decayRate0.4_nBasisDeg5_sub1.pkl'
    #'META_analysis/ModelFitting_DataFiles/3dTask/CIE/sub1/subset30000'
    #'Fitted_byWishart_ellipsoids_3DExpt_30_30_30_550_AEPsychSampling_EAVC_decayRate0.4_nBasisDeg5_sub1_subset30000.pkl'
    input_fileDir_fits2, file_name2 = select_file_and_get_path()
    input_fileDir_all2.append(input_fileDir_fits2)
    file_name_all2.append(file_name2)
    
    #also include bootstrapped files
    matching_files = [f for f in os.listdir(input_fileDir_fits2) if f.startswith(file_name2[:-4])]
    numDatasets_Wishart.append(len(matching_files))
    
    if COLOR_DIMENSION == 2:
        model_ellParams2 = np.full((len(matching_files), grid_pts_desired**COLOR_DIMENSION,5), np.nan)
    else:
        model_ellParams2 = []
    for idx, m in enumerate(matching_files):
        # Construct the full path to the selected file
        full_path2 = os.path.join(input_fileDir_fits2, m)
        
        # Load the necessary variables from the file
        with open(full_path2, 'rb') as f:
            vars_dict2 = pickled.load(f)
            if vars_dict2['grid'].shape[0] == grid_pts_desired and np.min(vars_dict2['grid']) == min_grid_desired:
                    model_ellParams2_m = vars_dict2['model_pred_Wishart'].params_ell
            elif vars_dict2[f'grid{grid_pts_desired}'].shape[0] == grid_pts_desired and\
                np.min(vars_dict2[f'grid{grid_pts_desired}']) == min_grid_desired:
                    model_ellParams2_m = vars_dict2[f'model_pred_Wishart_grid{grid_pts_desired}'].params_ell
            else:
                raise ValueError('Cannot find the model predictions at desired grid points.')
        if COLOR_DIMENSION == 2:
            model_ellParams2_reshape = np.reshape(np.array(model_ellParams2_m), 
                                                  (grid_pts_desired**COLOR_DIMENSION, 5))
            model_ellParams2[idx] = model_ellParams2_reshape
        else:
            model_ellParams2_reshape = list(itertools.chain.from_iterable(itertools.chain.from_iterable(model_ellParams2_m)))    
            model_ellParams2.append(model_ellParams2_reshape)
            
    if COLOR_DIMENSION == 2:
        data_load_Wishart.append(model_ellParams2.tolist())
    else:
        data_load_Wishart.append([model_ellParams2])

#%% load ground truths
# - Transformation matrices for converting between DKL, RGB, and W spaces
color_thres_data = vars_dict1['color_thres_data']
grid = vars_dict1['grid']

gt_ellParams = vars_dict1['gt_Wishart'].params_ell

if COLOR_DIMENSION == 2:
    #repeat it to be in a consistent format as the ellParams loaded
    gt_ellParams_reshape = np.repeat(np.reshape(np.array(gt_ellParams),
                                                (grid_pts_desired**COLOR_DIMENSION,5))[np.newaxis], 
                                     numDatasets_Wishart[0], axis = 0)
    gt_ellParams_rep = gt_ellParams_reshape.tolist()
    
else:
    gt_ellParams_flat = list(itertools.chain.from_iterable(itertools.chain.from_iterable(gt_ellParams)))
    gt_ellParams_rep = [[gt_ellParams_flat for _ in range(numDatasets_Wishart[0])]]
    


#%%
# Create an instance of ModelPerformance to evaluate model predictions
model_perf = ModelPerformance(COLOR_DIMENSION, gt_ellParams_rep)
indices = [0, grid_pts_desired-1]

# Select specific points (corners) in the 2D or 3D space for comparison
if COLOR_DIMENSION == 2:
    # For 2D, select specific corner points in the plane
    idx_corner = [[i, j] for i in indices for j in indices]
    covMat_corner = []
    for i,j in idx_corner:
        _, _, a_ij, b_ij, R_ij = gt_ellParams[i][j]
        radii_ij = np.array([a_ij, b_ij])
        eigvec_ij = rotAngle_to_eigenvectors(R_ij)
        covMat_corner_ij = ellParams_to_covMat(radii_ij, eigvec_ij) 
        covMat_corner.append(covMat_corner_ij)
else:
    # For 3D, select corner points in the 3D space
    idx_corner = [[i, j, k] for i in indices for j in indices for k in indices]

    # # Retrieve covariance matrices at these corner points
    covMat_corner = [ellParams_to_covMat(gt_ellParams[i][j][k]['radii'],\
                    gt_ellParams[i][j][k]['evecs']) for i, j, k in idx_corner]

# Evaluate the model performance using the loaded data and corner points
BWD_indvEll = np.full((nFiles_load_indvEll, numDatasets_indvEll[0],
                                      grid_pts_desired**COLOR_DIMENSION), np.nan)
for i in range(nFiles_load_indvEll):
    model_perf.evaluate_model_performance([data_load_indvEll[i]],covMat_corner = covMat_corner)
    BWD_indvEll[i] = model_perf.BW_distance
    
# Evaluate the model performance using the loaded data and corner points
BWD_Wishart = np.full((nFiles_load_Wishart, numDatasets_Wishart[0],
                                      grid_pts_desired**COLOR_DIMENSION), np.nan)
for i in range(nFiles_load_Wishart):
    model_perf.evaluate_model_performance([data_load_Wishart[i]],covMat_corner = covMat_corner)
    BWD_Wishart[i] = model_perf.BW_distance

# %% ------------------------------------------
# Plot Bures-Wasserstein distance
# ------------------------------------------
# plotting starts from here
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams.update({'font.size': 10})
if COLOR_DIMENSION == 2:
    # 2D case: get colors for the corner points from the reference stimulus
    cmap_temp = np.hstack((np.vstack([grid[m, n] for m, n in idx_corner]), np.ones((len(idx_corner), 1))))
    cmap_BW = (color_thres_data.M_2DWToRGB @ cmap_temp.T).T

else:
    cmap_BW = color_thres_data.W_unit_to_N_unit(np.vstack([grid[m, n, o] for m, n, o in idx_corner]))

axis1_merge = (1,2)
BWD_median_indvEll = np.nanmedian(BWD_indvEll, axis = axis1_merge)
BWD_indvEll_merged = np.sort(np.reshape(BWD_indvEll, (nFiles_load_indvEll, -1)), axis = 1)
non_nan_counts_idx_indv = np.sum(~np.isnan(BWD_indvEll_merged), axis=1)
CI95_bds_idx_indv = np.round(np.tile(non_nan_counts_idx_indv[np.newaxis].T, 2) *
                             np.array([0.025, 0.975])).astype(int)

# Create the error for yerr as a tuple of arrays for asymmetrical error
CI95_indv = np.vstack((BWD_indvEll_merged[np.arange(nFiles_load_indvEll),CI95_bds_idx_indv[:,0]], 
                      BWD_indvEll_merged[np.arange(nFiles_load_indvEll),CI95_bds_idx_indv[:,1]]))
yerr_indvEll = np.array([BWD_median_indvEll - CI95_indv[0], 
                         CI95_indv[1] - BWD_median_indvEll])
    
BWD_median_Wishart = np.median(BWD_Wishart, axis = axis1_merge)
BWD_Wishart_merged = np.sort(np.reshape(BWD_Wishart, (nFiles_load_Wishart, -1)), axis = 1)
non_nan_counts_idx_Wishart = np.sum(~np.isnan(BWD_Wishart_merged), axis=1)
CI95_bds_idx_Wishart = np.round(np.tile(non_nan_counts_idx_Wishart[np.newaxis].T, 2) *
                             np.array([0.025, 0.975])).astype(int)

# Create the error for yerr as a tuple of arrays for asymmetrical error
CI95_Wishart = np.vstack((BWD_Wishart_merged[np.arange(nFiles_load_Wishart),CI95_bds_idx_Wishart[:,0]],
                          BWD_Wishart_merged[np.arange(nFiles_load_Wishart),CI95_bds_idx_Wishart[:,1]]))
yerr_Wishart = np.array([BWD_median_Wishart - CI95_Wishart[0], 
                         CI95_Wishart[1] - BWD_median_Wishart])

#%%plotting
output_figDir_fits = input_fileDir_all1[-1].replace('DataFiles', 'FigFiles')
os.makedirs(output_figDir_fits, exist_ok=True)

pltSt = PltBWDSettings()

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams.update({'font.size': 13})

fig1, ax1 = plt.subplots(1,1, figsize = pltSt.figsize, dpi = pltSt.dpi) #; format 1: (3.5, 2.2); format 2:  (3.2, 2.2)
y_ub = 0.16#0.24
flag_visualize_baseline = False
if flag_visualize_baseline:
    BW_distance_circle_median = np.median(model_perf.BW_distance_minEigval)
    BW_distance_corner_median = np.median(model_perf.BW_distance_corner, axis = axis1_merge[:-1])
    ax1.plot([-3, nFiles_load_indvEll+2], [BW_distance_circle_median, BW_distance_circle_median],
             c = 'k',ls = '-',lw = 2, alpha = 0.8)
    for i in range(len(covMat_corner)):
        ax1.plot([-3, nFiles_load_indvEll+2], np.array([1,1])*BW_distance_corner_median[i],
                 c = cmap_BW[i],ls = '-',lw = 2, alpha = 0.8)
#Wishart
for i in range(nFiles_load_indvEll):
    ax1.errorbar(i+1, BWD_median_indvEll[i], 
                 yerr=yerr_indvEll[:, i].reshape(2, 1),
                capsize=pltSt.errorbar_cs, c = pltSt.errorbar_c,  #np.array([0,0,0])+i*0.2
                marker = pltSt.errorbar_m, markersize = pltSt.errorbar_ms, 
                lw = pltSt.errorbar_lw)
for i in range(nFiles_load_Wishart):
    ax1.errorbar(-i-1, BWD_median_Wishart[i], 
                 yerr=yerr_Wishart[:, i].reshape(2,1),
                 capsize=pltSt.errorbar_cs, c = pltSt.errorbar_c,  #np.array([0,0,0])+i*0.2
                 marker = pltSt.errorbar_m, markersize = pltSt.errorbar_ms, 
                 lw = pltSt.errorbar_lw)
ax1.plot([0,0],[0,y_ub],ls = pltSt.dashed_ls, lw=pltSt.dashed_lw, c = pltSt.dashed_lc)
ax1.set_xticks([-2,-1] + list(range(1,nFiles_load_indvEll+1)))
ax1.set_xticklabels([f'30000\n({COLOR_DIMENSION}D)',f'30000\n({COLOR_DIMENSION*2}D)']+\
                    [str(n) for n in trial_num_indvEll], rotation= 30) #[6027, 10045, 14014, 17640]
ax1.set_xlabel('Total number of trial')
#ax1.set_title(slc_color)
ax1.grid(True, alpha = 0.3)
ax1.set_xlim([-3, nFiles_load_indvEll + 1])
ax1.set_yticks(np.around(np.linspace(0,y_ub,5),2))
ax1.set_ylim([0,y_ub])
ax1.set_ylabel('Bures-Wasserstein distance')
plt.tight_layout()
figName1 = f"ModelPerformance_BuresWassersteinDistance_Wishart_vs_IndvEll_trialNum[{trial_num_indvEll}].pdf"
fig1.savefig(os.path.join(output_figDir_fits, figName1))   

#%% plot baseline 
seed = 0
if COLOR_DIMENSION == 2:
    ellipse_gt = (0.1, 0.1,30, 0, 0)  # center (x,y), axes (a,b), angle in degrees
    target_bw = CI95_Wishart[1,0]
    min_len = 0.01
    max_len = 0.6
    
    ellipses, distances = ModelPerformance.generate_ellipses_within_BWdistance(
        ellipse_gt, target_bw, min_len, max_len, 
        max_trials=1e8, tol=1e-4, seed=seed, num_ellipses=10)
    
    for i, (ellipse, dist) in enumerate(zip(ellipses, distances)):
        print(f"Ellipse {i+1}: {ellipse}, BW distance: {dist}")
        
    #compute the confidence intervals
    xu, yu, xi, yi = find_inner_outer_contours(np.array(ellipses))
    ell_min = np.vstack((xi, yi))
    ell_max = np.vstack((xu, yu))
    
    #gt
    ell_x_gt, ell_y_gt = PointsOnEllipseQ(*ellipse_gt)
    
else:
    ellipsoid_gt= {'radii': np.array([0.1, 0.1, 0.1]),
                  'evecs': np.eye(3),
                  'center': np.array([0,0,0])}
    target_bw = CI95_indv[1,0]  #CI95_Wishart[1,0] CI95_Wishart[1,1] CI95_indv[1,-1] CI95_indv[1,0]
    min_len = 0.001
    max_len = 0.3
    varying_planes = np.array([1,2])
    ellipsoids, distances = ModelPerformance.generate_ellipsoids_within_BWdistance(
        ellipsoid_gt, target_bw, min_len, max_len, 
        max_trials=1e8, tol=1e-4, seed=seed, num_ellipsoids=10)    
    
    modelpred_proj_ell = np.full((10,5), np.nan)
    for p in range(len(ellipsoids)):
        cov_p  = eig_to_covMat(ellipsoids[p]['radii']**2, ellipsoids[p]['evecs'])
        cov_ijk = cov_p[varying_planes][:,varying_planes]
        centers = ellipsoid_gt['center'][varying_planes]
        
        x_val, y_val = convert_2Dcov_to_points_on_ellipse(cov_ijk,
            ref_x = centers[0], ref_y = centers[1])
        pts_on_ell = np.vstack((x_val, y_val))
        
        _, _, _, modelpred_proj_ell[p] = fit_2d_isothreshold_contour(\
            centers, [], comp_RGB = pts_on_ell)

    # now we repeat the calculations we did for the sliced elliposoids
    xu, yu, xi, yi = find_inner_outer_contours(modelpred_proj_ell)
    ell_min = np.vstack((xi, yi))
    ell_max = np.vstack((xu, yu))
    
    #gt
    ell_x_gt, ell_y_gt = PointsOnEllipseQ(0.1, 0.1, 0, 0,0)    
    
# plotting starts from here
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams.update({'font.size': 13})

fig2, ax2 = plt.subplots(1,1, figsize = (2,2), dpi = 1024)
WishartPredictionsVisualization.add_CI_ellipses(ell_min, ell_max, ax = ax2,
                                                alpha = 0.2)
ax2.plot(ell_x_gt, ell_y_gt, c = 'k', ls = '--',lw = 0.85)
ax2.set_xlim([-0.35, 0.35])
ax2.set_ylim([-0.35, 0.35])
figName2 = f"Ellipses_within_targetBuresWassersteinDistance_{target_bw:.4f}.pdf"
fig2.savefig(os.path.join(output_figDir_fits, figName2))   

    
    
    
    
    
    
    
    
    