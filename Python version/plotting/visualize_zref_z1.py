#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 21:16:16 2024

@author: fangfang
"""
#%% import modules
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import dill as pickled
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids")
from analysis.ellipses_tools import covMat_to_ellParamsQ, PointsOnEllipseQ
from plotting.zref_z0_z1 import ZrefZ1Visualization

base_dir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'
fig_outputDir = base_dir+ 'ELPS_analysis/SanityChecks_FigFiles/sampled_zref_z1/'

#%% -----------------------------------------------------------
# Load data simulated using CIELab and organize data
# -----------------------------------------------------------
#file 3: model fits
plane_2D = 'GB plane'
path_str2 = base_dir + 'ELPS_analysis/ModelFitting_DataFiles/2D_oddity_task/'  
file_fits = f'Fitted_isothreshold_{plane_2D}_sim240perCond_'+\
                'samplingNearContour_jitter0.3_seed0_bandwidth0.005_oddity.pkl' 
full_path3 = f"{path_str2}{file_fits}"
with open(full_path3, 'rb') as f:  D = pickled.load(f)

#%% three variables we need to define for loading the data
#indices for the selected reference stimulus
idx1 = 4 
idx2 = 4
#whether we want to use the ground truth W for computing Zref, Z1, Z0, pC and etc
flag_use_gt = False
#chromatic directions
nTheta = 16 
#finer grid for the chromatic directions
nTheta_finer = nTheta*10
#three different seeds pre-selected 
seed = 7

if seed == 0:
    scaler_W_prior = 1
    hist_bin_edges = np.linspace(0, 110, 31)
    hist_diff_bin_edges  = np.linspace(-110,20, 31)
    x_err_plt = 1
elif seed == 8:
    scaler_W_prior = 2.5
    hist_bin_edges = np.linspace(0, 60, 31)
    hist_diff_bin_edges  = np.linspace(-70,30, 31)
    x_err_plt = 2
else:
    scaler_W_prior = 2.5
    hist_bin_edges = np.linspace(0, 50, 31)
    hist_diff_bin_edges  = np.linspace(-60,20, 31)
    x_err_plt = 3
    
#%%
rgb_ref_s  = jnp.array(D['xref_raw'][:,idx1,idx2])
if flag_use_gt:
    W_init = D['model_pred_Wishart'].W_est
else:
    W_init = scaler_W_prior*D['model'].sample_W_prior(jax.random.PRNGKey(seed))
Sig_init = D['model'].compute_Sigmas(D['model'].compute_U(W_init, D['grid']))
num_grid_pts1, num_grid_pts2 = D['grid'].shape[0], D['grid'].shape[1]

#intialize
scaler_radii = 2.56
contours_W_est = np.full((num_grid_pts1, num_grid_pts2,2,nTheta), np.nan)
contours_W_est_finer = np.full((num_grid_pts1, num_grid_pts2,2,nTheta_finer), np.nan)

for i in range(num_grid_pts1):
    for j in range(num_grid_pts2): 
        _, _, ab_gt, theta_gt = covMat_to_ellParamsQ(D['model_pred_Wishart'].Sigmas_recover_grid[i,j])
        ab_scaled_gt = ab_gt*scaler_radii
        contours_W_est_finer[i,j,0], contours_W_est_finer[i,j,1] = \
            PointsOnEllipseQ(*ab_scaled_gt, 
                             theta_gt, 
                             *D['grid'][i,j], 
                             nTheta = nTheta_finer)       
            
        contours_W_est[i,j,0], contours_W_est[i,j,1] = \
            PointsOnEllipseQ(*ab_scaled_gt, 
                             theta_gt,
                             *D['grid'][i,j], 
                             nTheta=nTheta)

# Plot sampled data using custom plotting function.
slc_cDir = 0
rgb_comp_pts = contours_W_est[idx1, idx2]
rgb_comp_contour = contours_W_est_finer[idx1, idx2]
zref_z1_vis = ZrefZ1Visualization(W_init,
                                  D['model'], 
                                  rgb_ref_s, 
                                  rgb_comp_pts,
                                  rgb_comp_contour,
                                  fig_dir= fig_outputDir,
                                  save_fig = True)
        
#%% 
# Simulate Zref, Z0, Z1, and distances using the estimated parameters from the model.
Zref_all, Z0_all, Z1_all, Z0_to_zref_all, Z1_to_zref_all, Zdiff_all, pC = \
    zref_z1_vis.simulate_zref_z0_z1()

#%% visualization
# Construct strings for RGB reference values for use in filenames.
x_str, y_str = str(np.round(rgb_ref_s[0], 2)), str(np.round(rgb_ref_s[1], 2)) 
zref_z1_vis.plot_sampled_comp(gt = contours_W_est_finer[idx1, idx2],
                              figName = f'Comp_{plane_2D}_x{x_str[0:4]}_y{y_str[0:4]}')

#plot the probability of correct
zref_z1_vis.plot_probC(pC, figName = f'pC_{plane_2D}_x{x_str[0:4]}_y{y_str[0:4]}_Wseed{seed}')

#lgd = [str(np.round(items,2)) for items in np.rad2deg(grid_theta)]

contours_W_cand = np.full((num_grid_pts1, num_grid_pts2, 2, 200), np.nan)
for i in range(num_grid_pts1):
    for j in range(num_grid_pts2):
        _, _, ab, theta = covMat_to_ellParamsQ(Sig_init[i,j])
        ab_scaled = ab*scaler_radii
        x_ij, y_ij = PointsOnEllipseQ(*ab_scaled, theta, *D['grid'][i,j])
        contours_W_cand[i,j,0] = x_ij
        contours_W_cand[i,j,1] = y_ij
        
zref_z1_vis.plot_sampled_zref_z1(Zref_all[slc_cDir][np.newaxis,:,:],
                     Z0_all[slc_cDir][np.newaxis,:,:],
                     Z1_all[slc_cDir][np.newaxis,:,:], 
                     gt = D['model_pred_Wishart'].fitEll_unscaled[idx1, idx2],
                     sim = contours_W_cand[idx1, idx2],
                     max_dots = 100, 
                     figName = f'sampled_zref_z1_{plane_2D}_x{x_str[0:4]}_y{y_str[0:4]}_Wseed{seed}')

#%% Define histogram bin edges
slc_cDir_list = np.array(list(range(0,nTheta_finer,10)))
zref_z1_vis.plot_EuclideanDist_hist(Z0_to_zref_all[slc_cDir_list],
                                    Z1_to_zref_all[slc_cDir_list],
                                    hist_bin_edges,
                                    figName = f'sampled_EuclideanDist_{plane_2D}_'+\
                                        f'x{x_str[0:4]}_y{y_str[0:4]}_Wseed{seed}')

zref_z1_vis.plot_EuclieanDist_diff_hist(Zdiff_all[slc_cDir_list], 
                            hist_diff_bin_edges, 
                            pC = np.round(pC,3),
                            figName = f'sampled_EuclideanDist_diff_{plane_2D}_'+\
                                f'x{x_str[0:4]}_y{y_str[0:4]}_Wseed{seed}')

#%%
target_pC = 2/3
nIters    = 1000
nLL = zref_z1_vis.compute_nLL_w_simData(nIters, pC)
nLL_target_pC = zref_z1_vis.compute_nLL_w_simData(nIters, target_pC)
nLL_avg, nLL_lb, nLL_ub = zref_z1_vis.nLL_avg_95CI(nLL)
nLL_target_pC_avg, nLL_target_pC_lb, nLL_target_pC_ub = zref_z1_vis.nLL_avg_95CI(nLL_target_pC)

zref_z1_vis.plot_nLL(nLL_avg, nLL_lb, nLL_ub, nLL_target_pC_avg,
             nLL_target_pC_lb, nLL_target_pC_ub, x_err_plt = x_err_plt,
             figName = f'{fig_outputDir}nLL_{plane_2D}_x{x_str[0:4]}_y{y_str[0:4]}_Wseed{seed}')

