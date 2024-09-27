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
from scipy.optimize import minimize

sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids")
from analysis.ellipses_tools import covMat_to_ellParamsQ, PointsOnEllipseQ
sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/Python version')
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

#%% Define visualization parameters for generating figures
vis = {
    'idx1': 4,            # Index for the selected row of the reference stimulus
    'idx2': 4,            # Index for the selected column of the reference stimulus
    'flag_use_gt': True, # Flag to indicate whether to use ground truth W for computations
    'flag_fit_Weifull':False, #this is only applicable if flag_use_gt is set to True, 
    #then we can look at whether Weibull function is a good fit to simulated percent correct
    'nTheta': 16,         # Number of chromatic directions for comparison stimulus
    'skip_steps': 10      # Step size for refining the chromatic direction grid
}
# Define a finer grid for chromatic directions
vis['nTheta_finer'] = vis['nTheta'] * vis['skip_steps']

#################################################
# Configuration settings for different seeds
##################################################

# Define a dictionary that holds configurations for each seed scenario
seed_configs = {
    0: {
        'scaler_W_prior': 1,
        'hist_bin_edges': np.linspace(0, 110, 31),
        'hist_diff_bin_edges': np.linspace(-110, 20, 31),
        'x_err_plt': 1
    },
    8: {
        'scaler_W_prior': 2,
        'hist_bin_edges': np.linspace(0, 70, 31),
        'hist_diff_bin_edges': np.linspace(-70, 20, 31),
        'x_err_plt': 2
    },
    7: {
        'scaler_W_prior': 2.25,
        'hist_bin_edges': np.linspace(0, 50, 31),
        'hist_diff_bin_edges': np.linspace(-60, 20, 31),
        'x_err_plt': 3
    }
}

# Pre-selected seed (from a list [0, 8, 7])
seed = 7

# Retrieve the configuration for the current seed
config = seed_configs[seed]

# Extract parameters from the configuration
scaler_W_prior = config['scaler_W_prior']
hist_bin_edges = config['hist_bin_edges']
hist_diff_bin_edges = config['hist_diff_bin_edges']
x_err_plt = config['x_err_plt']

# Define chromatic directions array for the current scenario
slc_cDir = np.arange(vis['nTheta'])  # Chromatic directions for visualization 
    
#%%Load reference stimulus and prepare covariance matrices
# Retrieve the reference stimulus using row (idx1) and column (idx2) indices
rgb_ref_s  = jnp.array(D['xref_raw'][:, vis['idx1'], vis['idx2']])
# Determine if we use W sampled from the prior or the ground truth W
if vis['flag_use_gt']:
    W_init = D['model_pred_Wishart'].W_est
else:
    W_init = scaler_W_prior*D['model'].sample_W_prior(jax.random.PRNGKey(seed))
# Compute covariance matrix (Sig_init) using the hypothesized W for each grid point
Sig_init = D['model'].compute_Sigmas(D['model'].compute_U(W_init, D['grid']))
# Retrieve the grid points for visualization (i.e., reference stimulus positions)
num_grid_pts1, num_grid_pts2 = D['grid'].shape[0], D['grid'].shape[1]

# Scaler applied to the covariance matrix to approximate the 66.7% threshold contour
scaler_radii = 2.56
target_pC    = 2/3 # The probability corresponding to the threshold contour

# Initialize arrays for storing the contours of threshold ellipses
#the comparison stimuli that are sampled exactly from the 66.7% threshold contour
contours_W_est = np.full((num_grid_pts1, num_grid_pts2,2, vis['nTheta']), np.nan)
#same as above but with finer grid points
contours_W_est_finer = np.full((num_grid_pts1, num_grid_pts2,2, vis['nTheta_finer']), np.nan)
#the contours computed using hypothesized W matrix. What we are interested in showing
#is the covariance matrix, but for the purpose of visualization, we are going to enlarge it by a scaler 2.56
contours_W_cand = np.full((num_grid_pts1, num_grid_pts2, 2, vis['nTheta_finer']), np.nan)

#%% Compute and store threshold contours for ground truth and hypothesized W
for i in range(num_grid_pts1):
    for j in range(num_grid_pts2): 
        # Retrieve the ground truth covariance matrix (Sig_gt_ij) for this grid point
        Sig_gt_ij = D['model_pred_Wishart'].Sigmas_recover_grid[i, j]

        # Convert the covariance matrix to ellipse parameters (axis lengths and orientation)
        _, _, ab_gt, theta_gt = covMat_to_ellParamsQ(Sig_gt_ij)

        # Scale the axes of the ellipse by the specified factor (2.56)
        ab_scaled_gt = ab_gt * scaler_radii

        # Compute the finer grid points along the ellipse for visualization
        pts_x_finer, pts_y_finer = PointsOnEllipseQ(*ab_scaled_gt,
                                                    theta_gt,
                                                    *D['grid'][i, j],
                                                    nTheta=vis['nTheta_finer']+1)

        # Store the finer contours for the ground truth ellipse
        contours_W_est_finer[i, j, 0] = pts_x_finer[:vis['nTheta_finer']]
        contours_W_est_finer[i, j, 1] = pts_y_finer[:vis['nTheta_finer']]

        # Compute the coarse grid points along the ellipse
        pts_x, pts_y = PointsOnEllipseQ(*ab_scaled_gt,
                                        theta_gt,
                                        *D['grid'][i, j], 
                                        nTheta=vis['nTheta']+1)

        # Store the coarse grid points
        contours_W_est[i, j, 0] = pts_x[:vis['nTheta']]
        contours_W_est[i, j, 1] = pts_y[:vis['nTheta']]

        # Compute ellipse points for the hypothesized W
        _, _, ab_test, theta_test = covMat_to_ellParamsQ(Sig_init[i, j])
        ab_scaled = ab_test * scaler_radii

        # Compute the contour points for the hypothesized W
        pts_x_test, pts_y_test = PointsOnEllipseQ(*ab_scaled,
                                                  theta_test, 
                                                  *D['grid'][i, j], 
                                                  nTheta=vis['nTheta_finer']+1)

        # Store the hypothesized W contours
        contours_W_cand[i, j, 0] = pts_x_test[:vis['nTheta_finer']]
        contours_W_cand[i, j, 1] = pts_y_test[:vis['nTheta_finer']]

#%% Plot the sampled contours and stimuli using a custom visualization class
# Retrieve the coarse comparison stimuli for the selected reference stimulus
rgb_comp_pts = contours_W_est[vis['idx1'], vis['idx2']]
# Retrieve the finer comparison stimuli (contour points)
rgb_comp_contour = contours_W_est_finer[vis['idx1'], vis['idx2']]
# Initialize a custom visualization object for plotting Zref, Z0, Z1, and contours
zref_z1_vis = ZrefZ1Visualization(W_init,
                                  D['model'], 
                                  rgb_ref_s, 
                                  rgb_comp_pts[:, slc_cDir],
                                  rgb_comp_contour,
                                  fig_dir= fig_outputDir,
                                  save_fig = False)
         
# Simulate and plot Zref, Z0, Z1, and their respective distances using the model
Zref_all, Z0_all, Z1_all, Z0_to_zref_all, Z1_to_zref_all, Zdiff_all, pC = \
    zref_z1_vis.simulate_zref_z0_z1()

#%% visualization
# Construct strings for RGB reference values for use in filenames.
x_str, y_str = str(np.round(rgb_ref_s[0], 2)), str(np.round(rgb_ref_s[1], 2)) 
fig_name_ext1 = f'_{plane_2D}_x{x_str[0:4]}_y{y_str[0:4]}'
fig_name_ext2 = '_useGtModel' if vis['flag_use_gt'] else ''
# visualize the comparison stimuli selected based on the ground truth
zref_z1_vis.plot_sampled_comp(gt = contours_W_est_finer[vis['idx1'], vis['idx2']], 
                              figName = f'Comp_{fig_name_ext1}{fig_name_ext2}')

#plot the probability of correct
zref_z1_vis.plot_probC(pC, figName = f'pC_{fig_name_ext1}_Wseed{seed}{fig_name_ext2}')

#%% visualization part 2
slc_cDir_finer = [0]
zref_z1_vis.plot_sampled_zref_z1(Zref_all[slc_cDir_finer],
                     Z0_all[slc_cDir_finer],
                     Z1_all[slc_cDir_finer],
                     gt = D['model_pred_Wishart'].fitEll_unscaled[vis['idx1'], vis['idx2']], #ground truth enlarged cov matrix
                     sim = contours_W_cand[vis['idx1'], vis['idx2']], #enlarged cov matrix computed using hypothesized W
                     max_dots = 100, #alpha = 0.8,
                     figName = f'sampled_zref_z1_{fig_name_ext1}_Wseed{seed}{fig_name_ext2}')

#%% Define histogram bin edges
#we compute the squared M distance for a fine grid of chromatic direction
#but we only want to show a small subset of them, evenly pick 16 
slc_cDir_list = np.array(list(range(0, vis['nTheta_finer'], vis['skip_steps'])))
# visualize the mahalanobis distance between (z0, z1), (zref, z1) and (z0 and zref)
zref_z1_vis.plot_EuclideanDist_hist(Z0_to_zref_all[slc_cDir_list],
                                    Z1_to_zref_all[slc_cDir_list],
                                    hist_bin_edges,
                                    figName = f'sampled_EuclideanDist_{fig_name_ext1}_'+
                                    f'Wseed{seed}{fig_name_ext2}')

#visualize the signed difference of mahalanobis distance
zref_z1_vis.plot_EuclieanDist_diff_hist(Zdiff_all[slc_cDir_list], 
                            hist_diff_bin_edges, 
                            pC = np.round(pC[slc_cDir_list],3),
                            figName = f'sampled_EuclideanDist_diff_{fig_name_ext1}_'+\
                                f'Wseed{seed}{fig_name_ext2}')

#%% compute and visualize nLL
#number of simulated datasets
#data are simulated based on the ground truth pC, which is 66.7%
nIters    = 1000
#compute the negative log likelihood given pC, which is simulated based on hypothesized W matrix
nLL = zref_z1_vis.compute_nLL_w_simData(nIters, pC)
#compute the mean nLL and 95% CI
nLL_avg, nLL_lb, nLL_ub = zref_z1_vis.nLL_avg_95CI(nLL)
#compute the nLL given gt pC, which is 66.7% and 95% CI
nLL_target_pC = zref_z1_vis.compute_nLL_w_simData(nIters, target_pC)
nLL_target_pC_avg, nLL_target_pC_lb, nLL_target_pC_ub = zref_z1_vis.nLL_avg_95CI(nLL_target_pC)

#visualize
if vis['flag_use_gt']:
    zref_z1_vis.plot_nLL(None, None, None, nLL_avg, nLL_lb, nLL_ub, x_err_plt = x_err_plt,
                 figName = f'{fig_outputDir}nLL_{fig_name_ext1}_Wseed{seed}{fig_name_ext2}')
else:
    zref_z1_vis.plot_nLL(nLL_avg, nLL_lb, nLL_ub, nLL_target_pC_avg,
                 nLL_target_pC_lb, nLL_target_pC_ub, x_err_plt = x_err_plt,
                 figName = f'{fig_outputDir}nLL_{fig_name_ext1}_Wseed{seed}{fig_name_ext2}')        

#%%-------------------------------------------
# Visualize how we compute model predictions
#---------------------------------------------
#we only need to compute the model prediction if the hypothesized W is the ground truth
if vis['flag_use_gt']:
    #specify the number of vector length
    nRadiiScaler = 50
    #evenly sample from 0 to 6
    scaler_radii_list = np.linspace(0,6,nRadiiScaler) 
    #only select three points for plotting the three measurement clusters
    scaler_radii_list_plt_idx = [0, 21, 42]
    #select a specific chromatic direction
    cDir_coarse_unit, cDir_finer_unit = [12], [120]
    #color map
    cmap_slc_cDir = np.array([0.8078, 0.4275, 0.7412])
    
    #initialize
    Zref_all_list_plt = np.full((len(scaler_radii_list_plt_idx), vis['nTheta_finer'], *Zref_all.shape[-2:]), np.nan)
    Z0_all_list_plt = np.full(Zref_all_list_plt.shape, np.nan)
    Z1_all_list_plt = np.full(Zref_all_list_plt.shape, np.nan)
    pC_list = np.full((nRadiiScaler),np.nan)
    
    #loop through each vector length
    for l in range(nRadiiScaler):
        print(l)
        contours_W_est_l = np.full((num_grid_pts1, num_grid_pts2,2, vis['nTheta']), np.nan)
        contours_W_est_finer_l = np.full((num_grid_pts1, num_grid_pts2,2, vis['nTheta_finer']), np.nan)
        for i in range(num_grid_pts1):
            for j in range(num_grid_pts2): 
                _, _, ab_gt, theta_gt = covMat_to_ellParamsQ(D['model_pred_Wishart'].Sigmas_recover_grid[i,j])
                ab_scaled_gt = ab_gt*scaler_radii_list[l]
                s1, s2 = PointsOnEllipseQ(*ab_scaled_gt, 
                                            theta_gt, 
                                            *D['grid'][i,j], 
                                            nTheta = vis['nTheta_finer']+1)  
                contours_W_est_finer_l[i,j,0] = s1[:vis['nTheta_finer']]
                contours_W_est_finer_l[i,j,1] = s2[:vis['nTheta_finer']]   
                
                t1, t2 = PointsOnEllipseQ(*ab_scaled_gt, 
                                             theta_gt,
                                             *D['grid'][i,j], 
                                             nTheta= vis['nTheta']+1)
                contours_W_est_l[i,j,0] = t1[:vis['nTheta']]
                contours_W_est_l[i,j,1] = t2[:vis['nTheta']]
    
        rgb_comp_pts_l = contours_W_est_l[vis['idx1'], vis['idx2']]
        rgb_comp_contour_l = contours_W_est_finer_l[vis['idx1'], vis['idx2']]
        zref_z1_vis_l = ZrefZ1Visualization(W_init,
                                              D['model'], 
                                              rgb_ref_s, 
                                              rgb_comp_pts_l[:, cDir_coarse_unit],
                                              rgb_comp_contour_l,
                                              fig_dir= fig_outputDir,
                                              save_fig = False)
    
        # Simulate Zref, Z0, Z1, and distances using the estimated parameters from the model.
        Zref_all_l, Z0_all_l, Z1_all_l,_,_,_, pC_l = zref_z1_vis_l.simulate_zref_z0_z1()
        pC_list[l] = np.mean(pC_l)
    
        # Plot sampled data only for selected vector lengths
        if l in scaler_radii_list_plt_idx:
            l_idx = np.where(scaler_radii_list_plt_idx == l)
            Zref_all_list_plt[l_idx] = Zref_all_l
            Z0_all_list_plt[l_idx] = Z0_all_l
            Z1_all_list_plt[l_idx] = Z1_all_l
            zref_z1_vis_l.plot_sampled_zref_z1(Zref_all_l[cDir_finer_unit],
                                               Z0_all_l[cDir_finer_unit],
                                               Z1_all_l[cDir_finer_unit],
                                               max_dots = 100, 
                                               alpha = 0.8,
                                               color_comp = cmap_slc_cDir,
                                               figName = f'sampled_zref_z1_{fig_name_ext1}_Wseed{seed}{fig_name_ext2}')
    
    #%%
    if vis['flag_fit_Weifull']:
        lb_opt = np.array([0.5,0.5])
        ub_opt = np.array([4,4])
        pC_weibull = lambda d: (1 - target_pC*np.exp(- (scaler_radii_list/d[0])** d[1]))
        nLL_weibull = lambda d: -np.sum(pC_list*np.log(pC_weibull(d)) + (1-pC_list)*np.log(1-pC_weibull(d)))
                
        # Generate initial points for the optimization algorithm within the bounds.
        init = np.random.rand(2) * (ub_opt - lb_opt) + lb_opt
        # Set the options for the optimization algorithm.
        options = {'maxiter': 1e5, 'disp': False}
        res = minimize(nLL_weibull, init,method='SLSQP',
                       bounds=[(lb_opt[0], ub_opt[0]), (lb_opt[1], ub_opt[1])], options=options)
        # Store the result of each optimization run.
        print(res.x)
        pC_weibull_pred = pC_weibull(res.x)
        
    #%% plot the probability of correct responses as a function of vector length
    fig, ax = plt.subplots(1, 1, figsize = (4,1.6), dpi= 256)
    ax.plot(scaler_radii_list, pC_list, c= 'k', lw = 2)
    if vis['flag_fit_Weibull']:
        ax.plot(scaler_radii_list, pC_weibull_pred, c='r',alpha = 0.5)
    ax.scatter(scaler_radii_list[scaler_radii_list_plt_idx],
               pC_list[scaler_radii_list_plt_idx],
               c = 'k', edgecolor = 'white')
    ax.set_xticks([])
    ax.set_xlabel('Vector length along a chromatic direction')
    ax.set_yticks(np.around([1/3, 2/3, 1],3))
    ax.set_ylim([0.3,1.03])
    ax.set_ylabel('p(correct)')
    ax.grid(True)
    # plt.tight_layout()
    # plt.savefig(f'{fig_outputDir}pC_wIncreasingVecLength_wFixedChromaticDir.pdf')

