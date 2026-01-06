#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 20:31:02 2024

@author: fangfang

This script compares the performance of the Wishart model and the 
independent-threshold model in predicting color discrimination data. 

Model performance is quantified using the Bures-Wasserstein distance (BWD) 
between model predictions and the ground truth, which is defined as the 
Wishart fit to simulated ΔE94 (CIELab 1994) data. Given this ground truth, 
we simulate trials using AEPsych, fit both models to those simulated trials, 
and evaluate their respective accuracy.

The script includes the following main sections:

1. Load independent-threshold model predictions  
   (multiple files may be loaded, each corresponding to a different trial count).

2. Load Wishart model predictions  
   (can include multiple files, e.g., from different task designs such as 
   interleaved 2D or full 4D experiments).

3. Load ground-truth Wishart model fit  
   (obtained from the simulated CIELab ΔE94 data).

4. Compute Bures-Wasserstein distances between model predictions and ground truth, 
   separately for the Wishart and independent-threshold models.

5. Visualize the comparison by plotting the median BWD along with the 95% 
   confidence intervals for each condition.

6. Generate synthetic ellipses/ellipsoids with specified BWDs relative to 
   a unit circle/sphere, to provide visual references for interpreting model performance.

7. Visualize the confidence ellipses/ellipsoids to illustrate variability 
   in model predictions.
   
"""

import jax
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
import sys
import os
import re
import dill as pickled
import numpy as np
import itertools
from dataclasses import replace
sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids')
#sys.path.append("/Users/fh862-adm/Documents/GitHub/ellipsoids/ellipsoids")
from analysis.ellipses_tools import ellParams_to_covMat, rotAngle_to_eigenvectors, \
    PointsOnEllipseQ, find_inner_outer_contours, convert_2Dcov_to_points_on_ellipse,\
    fit_2d_isothreshold_contour
from analysis.ellipsoids_tools import  eig_to_covMat
from plotting.wishart_predictions_plotting import WishartPredictionsVisualization
from analysis.model_performance import ModelPerformance, \
    compute_95CI_BWD_multipleConditions, generate_ellipses_within_BWdistance,\
    generate_ellipsoids_within_BWdistance
from plotting.wishart_plotting import PlotSettingsBase 
from plotting.modelperf_plotting import PltBWDSettings, ModelPerformanceVisualization
from analysis.utils_load import select_file_and_get_path

sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/Python version')
#sys.path.append("/Users/fh862-adm/Documents/GitHub/ColorEllipsoids/Python version")

##################################################
COLOR_DIMENSION = 3 #this can be modified, 2 or 3

#%% 
# --------------------------------------------------------
# Load the model fits by the independent threshold model
# ----------------------------------------------------------
nFiles_load_indvEll = 4   # Total number of files to load
grid_pts_desired = 5      # Grid size for reference stimulus (e.g., 7x7 for 2D, 5x5x5 for 3D)
min_grid_desired = -0.7   # Minimum grid value expected for the reference stimulus

# Initialize lists for storing loaded data and metadata
data_load_indvEll = []
input_fileDir_all1 = []
file_name_all1 = []
trial_num_indvEll = []
numDatasets_indvEll = []

for _ in range(nFiles_load_indvEll):
    #2D
    #META_analysis/ModelFitting_DataFiles/2dTask_indvEll/CIE/sub2/subset6027'
    #'IndvFitted_byWishart_Isoluminant plane_49interleaved2DExpt_10_10_10_330_AEPsychSampling_EAVC_sub2_subset6027.pkl'
    
    #or 3D
    #'META_analysis/ModelFitting_DataFiles/3dTask_indvEll/CIE/sub1/subset30000'
    # 'IndvFitted_byWishart_ellipsoids_3DExpt_30_30_30_550_AEPsychSampling_EAVC_decayRate0.4_nBasisDeg5_sub1_subset30000.pkl'
    
    # Select a model fitting result file (interactive or scripted selection)
    input_fileDir_fits1, file_name1 = select_file_and_get_path()
    input_fileDir_all1.append(input_fileDir_fits1)
    file_name_all1.append(file_name1)

    # Extract trial number from the filename (e.g., '..._subset6027.pkl' → 6027)
    trial_num = int(file_name1.split('subset')[1].split('.pkl')[0])
    trial_num_indvEll.append(trial_num)

    # Construct the full path to the file and load its contents
    full_path1 = os.path.join(input_fileDir_fits1, file_name1)
    with open(full_path1, 'rb') as f:
        vars_dict1 = pickled.load(f)

    # Check that the grid configuration matches the desired setup
    if vars_dict1['NUM_GRID_PTS'] == grid_pts_desired and np.min(vars_dict1['grid']) == min_grid_desired:
        prefix = f'model_pred_indv_subset{trial_num}'

        # Find all bootstrap datasets associated with this subset
        matching_keys = [key for key in vars_dict1 if key.startswith(prefix)]
        numDatasets_indvEll.append(len(matching_keys))

        # Load model-predicted ellipse parameters for each bootstrap dataset
        if COLOR_DIMENSION == 2:
            model_ellParams1 = np.full((len(matching_keys), grid_pts_desired**2, 5), np.nan)
            for idx, m in enumerate(matching_keys):
                model_ellParams1[idx] = vars_dict1[m]['fitEll_params']
            data_load_indvEll.append(model_ellParams1.tolist())
        else: 
            model_ellParams1 = []
            for m in matching_keys:
                model_ellParams1.append(vars_dict1[m]['fitEll_params'])
            data_load_indvEll.append(model_ellParams1)
    else:
        raise ValueError('The file does not match the desired grid configuration.')

#%% ------------------------------------------ 
# Load the model fits by the Wishart model
# -------------------------------------------- 
nFiles_load_Wishart = 2
# Initialize lists for storing loaded data and metadata
data_load_Wishart = []
input_fileDir_all2 = []
file_name_all2 = []
trial_num_Wishart = []
numDatasets_Wishart = [] 
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
        full_path2 = os.path.join(input_fileDir_fits2, m)
        
        # Load the necessary variables from the file
        with open(full_path2, 'rb') as f:
            vars_dict2 = pickled.load(f)
            #extract the number of trials
            if idx == 0: trial_num_Wishart.append(vars_dict2['data_AEPsych'][0].shape[0])
            
            # Check that the grid configuration matches the desired setup
            if vars_dict2['grid'].shape[0] == grid_pts_desired and\
                np.min(vars_dict2['grid']) == min_grid_desired:
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
        data_load_Wishart.append(model_ellParams2)

#store whether this is a interleaved 2D/3D or a full 4D/6D task
task_dims = [re.search(r'(\d+D)Expt', name).group(1) for name in file_name_all2]

#%% load ground truths
# - Transformation matrices for converting between DKL, RGB, and W spaces
color_thres_data = vars_dict1['color_thres_data']
grid = vars_dict1['grid']
gt_ellParams = vars_dict1['gt_Wishart'].params_ell

if COLOR_DIMENSION == 2:
    #repeat it to be in a consistent format as the ellParams loaded
    gt_ellParams_reshape = np.reshape(np.array(gt_ellParams),
                                      (grid_pts_desired**COLOR_DIMENSION,5))
    gt_ellParams_rep = gt_ellParams_reshape.tolist()
else:
    gt_ellParams_rep = list(itertools.chain.from_iterable(itertools.chain.from_iterable(gt_ellParams)))

#%% Compute model performance
#define the corners for computing the baseline
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

    # Retrieve covariance matrices at these corner points
    covMat_corner = [ellParams_to_covMat(gt_ellParams[i][j][k]['radii'],\
                    gt_ellParams[i][j][k]['evecs']) for i, j, k in idx_corner]

# Helper function to evaluate model performance
def compute_BWD(data_list, nFiles, numDatasets):
    BWD = np.full((nFiles, numDatasets[0], grid_pts_desired ** COLOR_DIMENSION), np.nan)
    for i in range(nFiles):
        perf = ModelPerformance(COLOR_DIMENSION, gt_ellParams_rep, nLevels=len(data_list[i]))
        perf.evaluate_model_performance(data_list[i], covMat_corner=covMat_corner)
        BWD[i] = perf.BW_distance
    return BWD, perf

# Compute BWD for individual ellipsoid fits and Wishart model fits
BWD_indvEll, model_perf_indvEll = compute_BWD(data_load_indvEll, nFiles_load_indvEll, numDatasets_indvEll)
BWD_Wishart, model_perf_Wishart = compute_BWD(data_load_Wishart, nFiles_load_Wishart, numDatasets_Wishart)

# compute the 95% confidence interval
#for the independent threshold model
CI95_indv, yerr_indvEll, BWD_median_indvEll = compute_95CI_BWD_multipleConditions(BWD_indvEll)

#for the WIshart model
CI95_Wishart, yerr_Wishart, BWD_median_Wishart = compute_95CI_BWD_multipleConditions(BWD_Wishart)

# %% ------------------------------------------
# Plot Bures-Wasserstein distance
# ------------------------------------------
if COLOR_DIMENSION == 2:
    # 2D case: get colors for the corner points from the reference stimulus
    cmap_temp = np.hstack((np.vstack([grid[m, n] for m, n in idx_corner]), np.ones((len(idx_corner), 1))))
    cmap_BW = (color_thres_data.M_2DWToRGB @ cmap_temp.T).T
    y_ub = 0.12
else:
    cmap_BW = color_thres_data.W_unit_to_N_unit(np.vstack([grid[m, n, o] for m, n, o in idx_corner]))
    y_ub = 0.16

output_figDir_fits = input_fileDir_all1[-1].replace('DataFiles', 'FigFiles')
os.makedirs(output_figDir_fits, exist_ok=True)

# Create base settings
pltSettings_base = PlotSettingsBase(fig_dir=output_figDir_fits, fontsize=14)

# Create instance and override fields
pltBWD_settings = replace(PltBWDSettings(), **pltSettings_base.__dict__)
pltBWD_settings = replace(
    pltBWD_settings,
    flag_visualize_baseline = False, #whether we want to viusalize the BWD baseline
    x_ticklabels =[f'{i}\n({j})' for i,j in zip(trial_num_Wishart, task_dims)] + \
        [str(n) for n in trial_num_indvEll],
    y_ub = y_ub,
    fig_name = "ModelPerformance_BuresWassersteinDistance_Wishart_vs_IndvEll_"+\
                f"trialNum[{trial_num_indvEll}].pdf")

#compare the BWD between the Wishart model and the independent threshold model
modelperf_plt = ModelPerformanceVisualization(model_perf_Wishart, pltBWD_settings, 
                                              save_fig=False, save_format = 'pdf')
modelperf_plt.plot_BWD_indvEll_vs_Wishart(nFiles_load_indvEll, 
                                          BWD_median_indvEll, 
                                          yerr_indvEll, 
                                          nFiles_load_Wishart, 
                                          BWD_median_Wishart, 
                                          yerr_Wishart, 
                                          pltBWD_settings,
                                          cmap_corner = cmap_BW)

#%% simulate ellipses with a target BWD
seed = 0
if COLOR_DIMENSION == 2:
    ellipse_gt = (0.1, 0.1,30, 0, 0)  # center (x,y), axes (a,b), angle in degrees
    target_bw = CI95_Wishart[1,0]
    min_len = 0.01
    max_len = 0.6
    
    ellipses, distances = generate_ellipses_within_BWdistance(
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
    target_bw = CI95_indv[1,1]  #CI95_Wishart; CI95_indv[0]: lower bound; CI95_indv[1]: upper bound
    min_len = 0.001
    max_len = 0.3
    varying_planes = np.array([1,2])
    ellipsoids, distances = generate_ellipsoids_within_BWdistance(
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
        
        _, _, modelpred_proj_ell[p], _ = fit_2d_isothreshold_contour(\
            centers, pts_on_ell)

    # now we repeat the calculations we did for the sliced elliposoids
    xu, yu, xi, yi = find_inner_outer_contours(modelpred_proj_ell)
    ell_min = np.vstack((xi, yi))
    ell_max = np.vstack((xu, yu))
    
    #gt
    ell_x_gt, ell_y_gt = PointsOnEllipseQ(0.1, 0.1, 0, 0,0)    
    
#%% visualize 
#the code is very simple, so no need to write it as a method
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams.update({'font.size': 13})

fig2, ax2 = plt.subplots(1,1, figsize = (2,2), dpi = 1024)
WishartPredictionsVisualization.add_CI_ellipses(ell_min, ell_max, ax = ax2,
                                                alpha = 0.2)
ax2.plot(ell_x_gt, ell_y_gt, c = 'k', ls = '--',lw = 0.85)
ax2.set_xlim([-0.35, 0.35])
ax2.set_ylim([-0.35, 0.35])
figName2 = f"Ellipses_within_targetBuresWassersteinDistance_{target_bw:.4f}.pdf"
#fig2.savefig(os.path.join(output_figDir_fits, figName2))   

    
    
    
    
    
    
    
    
    