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

#%% three variables we need to define for loading the data
#plotting specifics for Figure 10
vis = {'idx1': 4,     #indices for the selected reference stimulus
       'idx2':4, 
       'flag_use_gt': False,  #whether we want to use the ground truth W for computing Zref, Z1, Z0, pC and etc
       'nTheta': 16,  #chromatic directions
       'skip_steps':10,
       }
#finer grid for the chromatic directions
vis['nTheta_finer'] = vis['nTheta']* vis['skip_steps']

#################################################
#three different seeds pre-selected 
##################################################
seed = 7 #[0,8,7]
slc_cDir = np.array(list(range(vis['nTheta'])))
if seed == 0:
    scaler_W_prior = 1
    hist_bin_edges = np.linspace(0, 110, 31)
    hist_diff_bin_edges = np.linspace(-110,20, 31)
    x_err_plt = 1
elif seed == 8:
    scaler_W_prior = 2
    hist_bin_edges = np.linspace(0, 70, 31)
    hist_diff_bin_edges  = np.linspace(-70,20, 31)
    x_err_plt = 2
else:
    scaler_W_prior = 2.25
    hist_bin_edges = np.linspace(0, 50, 31)
    hist_diff_bin_edges  = np.linspace(-60,20, 31)
    x_err_plt = 3  
    
if vis['flag_use_gt']:
    hist_bin_edges = np.linspace(0, 50, 31)
    hist_diff_bin_edges  = np.linspace(-60,30, 31)    
    
#%%
#retrieve the reference stimulus given the row and col indices
rgb_ref_s  = jnp.array(D['xref_raw'][:, vis['idx1'], vis['idx2']])
#whether we want to sample W from a prior distribution or use the ground truth
if vis['flag_use_gt']:
    W_init = D['model_pred_Wishart'].W_est
else:
    W_init = scaler_W_prior*D['model'].sample_W_prior(jax.random.PRNGKey(seed))
#given W, compute covariance matrix
Sig_init = D['model'].compute_Sigmas(D['model'].compute_U(W_init, D['grid']))
#retrieve the grid points
num_grid_pts1, num_grid_pts2 = D['grid'].shape[0], D['grid'].shape[1]

#scaler used on the covariance matrix so that it approximates the 66.7% threshold contour
scaler_radii = 2.56
target_pC    = 2/3
#initialize
#the comparison stimuli that are sampled exactly from the 66.7% threshold contour
contours_W_est = np.full((num_grid_pts1, num_grid_pts2,2, vis['nTheta']), np.nan)
#same as above but with finer grid points
contours_W_est_finer = np.full((num_grid_pts1, num_grid_pts2,2, vis['nTheta_finer']), np.nan)
#the contours computed using hypothesized W matrix. What we are interested in showing
#is the covariance matrix, but for the purpose of visualization, we are going to enlarge it by a scaler 2.56
contours_W_cand = np.full((num_grid_pts1, num_grid_pts2, 2, vis['nTheta_finer']), np.nan)

#for each reference stimulus
for i in range(num_grid_pts1):
    for j in range(num_grid_pts2): 
        #retrieve the ground truth covariance matrix
        Sig_gt_ij = D['model_pred_Wishart'].Sigmas_recover_grid[i,j]
        #convert covariance matrix to ellipse parameters
        _, _, ab_gt, theta_gt = covMat_to_ellParamsQ(Sig_gt_ij)
        #scale the radii by the scaler
        ab_scaled_gt = ab_gt*scaler_radii
        #derive the points on the enlarged ellipse
        pts_x_finer, pts_y_finer = PointsOnEllipseQ(*ab_scaled_gt, 
                                                     theta_gt, 
                                                     *D['grid'][i,j], 
                                                     nTheta = vis['nTheta_finer']+1)  
        #save the finer contours
        contours_W_est_finer[i,j,0] = pts_x_finer[:vis['nTheta_finer']]
        contours_W_est_finer[i,j,1] = pts_y_finer[:vis['nTheta_finer']]     
        
        #more coarse samples
        pts_x, pts_y = PointsOnEllipseQ(*ab_scaled_gt, 
                                     theta_gt,
                                     *D['grid'][i,j], 
                                     nTheta= vis['nTheta']+1)
        #save them
        contours_W_est[i,j,0] = pts_x[:vis['nTheta']]
        contours_W_est[i,j,1] = pts_y[:vis['nTheta']]
        
        #for hypothesized W
        _, _, ab_test, theta_test = covMat_to_ellParamsQ(Sig_init[i,j])
        ab_scaled = ab_test*scaler_radii
        pts_x_test, pts_y_test = PointsOnEllipseQ(*ab_scaled, 
                                                  theta_test,
                                                  *D['grid'][i,j],
                                                  nTheta = vis['nTheta_finer']+1)
        contours_W_cand[i,j,0] = pts_x_test[:vis['nTheta_finer']]
        contours_W_cand[i,j,1] = pts_y_test[:vis['nTheta_finer']]

#%% Plot sampled data using custom plotting function.
#the comparison stimuli (coarse sampling)
rgb_comp_pts = contours_W_est[vis['idx1'], vis['idx2']]
#the comparison stimuli (fine sampling)
rgb_comp_contour = contours_W_est_finer[vis['idx1'], vis['idx2']]
#define the object for plotting
zref_z1_vis = ZrefZ1Visualization(W_init,
                                  D['model'], 
                                  rgb_ref_s, 
                                  rgb_comp_pts[:, slc_cDir],
                                  rgb_comp_contour,
                                  fig_dir= fig_outputDir,
                                  save_fig = False)
         
# Simulate Zref, Z0, Z1, and distances using the estimated parameters from the model.
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
slc_cDir_finer = [0]#int(vis['skip_steps'] * (slc_cDir+1) - 3) #slc_cDir*10#
# default_cmap = plt.get_cmap('tab20b')
# values       = np.linspace(0, 1, 16)
# colors_array = default_cmap(np.append(values[3:], values[:3]))
# colors_array[slc_cDir]
zref_z1_vis.plot_sampled_zref_z1(Zref_all[slc_cDir_finer],
                     Z0_all[slc_cDir_finer],
                     Z1_all[slc_cDir_finer],
                     gt = D['model_pred_Wishart'].fitEll_unscaled[vis['idx1'], vis['idx2']], #ground truth enlarged cov matrix
                     sim = contours_W_cand[vis['idx1'], vis['idx2']], #enlarged cov matrix computed using hypothesized W
                     max_dots = 100, #alpha = 0.8,
                     figName = f'sampled_zref_z1_{fig_name_ext1}_Wseed{seed}{fig_name_ext2}')
                    #_pC{np.mean(pC):.3f}
#                     color_comp = np.array([0.80784314, 0.42745098, 0.74117647]),

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


#%%--------------------------
#NEW SECTION
#----------------------------
#intialize
nRadii = 50
scaler_radii_list = np.linspace(0,6,nRadii) 
pC_list = np.full((nRadii),np.nan)
for l in range(nRadii):
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
            contours_W_est_finer_l[i,j,0], contours_W_est_finer_l[i,j,1] = s1[:vis['nTheta_finer']], s2[:vis['nTheta_finer']]     
            
            t1, t2 = PointsOnEllipseQ(*ab_scaled_gt, 
                                         theta_gt,
                                         *D['grid'][i,j], 
                                         nTheta= vis['nTheta']+1)
            contours_W_est_l[i,j,0], contours_W_est_l[i,j,1] = t1[:vis['nTheta']], t2[:vis['nTheta']]

    # Plot sampled data using custom plotting function.
    rgb_comp_pts_l = contours_W_est_l[vis['idx1'], vis['idx2']]
    rgb_comp_contour_l = contours_W_est_finer_l[vis['idx1'], vis['idx2']]
    zref_z1_vis_l = ZrefZ1Visualization(W_init,
                                      D['model'], 
                                      rgb_ref_s, 
                                      rgb_comp_pts_l[:, slc_cDir],#[:,np.newaxis]
                                      rgb_comp_contour_l,
                                      fig_dir= fig_outputDir,
                                      save_fig = False)

    # Simulate Zref, Z0, Z1, and distances using the estimated parameters from the model.
    _,_,_,_,_,_, pC_l = zref_z1_vis_l.simulate_zref_z0_z1()
    pC_list[l] = np.mean(pC_l)

#%%
fig, ax = plt.subplots(1, 1, figsize = (4,1.6), dpi= 256)
ax.plot(scaler_radii_list, pC_list, c= 'k', lw = 2)
ax.set_xticks([])
ax.set_xlabel('Vector length along a chromatic direction')
ax.set_yticks(np.around([1/3, 2/3, 1],3))
ax.set_ylim([0.3,1.03])
ax.set_ylabel('p(correct)')
ax.grid(True)
plt.tight_layout()
plt.savefig(f'{fig_outputDir}pC_wIncreasingVecLength_wFixedChromaticDir.pdf')



