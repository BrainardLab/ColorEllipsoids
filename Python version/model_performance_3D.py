#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 00:39:55 2024

@author: fangfang
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.spatial.transform import Rotation as R

import sys
sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids')
from core import viz, utils, oddity_task, model_predictions, optim
from core.wishart_process import WishartProcessModel
sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/Python version')
from Simulate_probCorrectResp_3D import plot_3D_sampledComp
from Isothreshold_ellipsoids_CIELab import fit_3d_isothreshold_ellipsoid

#%% ------------------------------------------
# Load data simulated using CIELab
# ------------------------------------------
import os
import pickle
import numpy as np

nSims = 240
jitter = 0.3
file_name1 = 'Fitted_isothreshold_ellipsoids_sim'+str(nSims) +\
            'perCond_samplingNearContour_jitter'+str(jitter)+'_bandwidth0.005.pkl'
path_str1  = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
            'ELPS_analysis/ModelFitting_DataFiles/'
full_path1 = f"{path_str1}{file_name1}"
os.chdir(path_str1)

#load data    
with open(full_path1, 'rb') as f:
    # Load the object from the file
    data_load1 = pickle.load(f)
ellipsoidParams_est = data_load1['params_ellipsoids']

#%% load ground truths
path_str2 = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
        'ELPS_analysis/Simulation_DataFiles/'
file_name2 = 'Isothreshold_ellipsoid_CIELABderived.pkl'
full_path2 = f"{path_str2}{file_name2}"
os.chdir(path_str2)

#Here is what we do if we want to load the data
with open(full_path2, 'rb') as f:
    # Load the object from the file
    data_load2 = pickle.load(f)
param3D, stim3D, results3D = data_load2[0], data_load2[1], data_load2[2]

ellipsoidParams_gt = results3D['ellipsoidParams']
ref_size_dim1, ref_size_dim2, ref_size_dim3 = ellipsoidParams_gt.shape[0:3]  

#%%
fig, axes = plt.subplots(1, 3, figsize=(12,4))
plt.rcParams['figure.dpi'] = 250
radii_est = np.full((3, ref_size_dim1, ref_size_dim2, ref_size_dim3),np.nan)
radii_gt = np.full(radii_est.shape, np.nan)
radii_est_sort_idx = np.full(radii_est.shape,100, dtype=int)
radii_gt_sort_idx = np.full(radii_est.shape,100, dtype=int)

for l in range(3):
    axes[l].plot([0, 1], [0,1], color = 'k', linestyle = '--')
for i in range(ref_size_dim1):
    for j in range(ref_size_dim2):
        for k in range(ref_size_dim3):
            radii_est_ijk = ellipsoidParams_est[i][j][k]['radii']
            radii_est_sort_idx[:,i,j,k] = np.argsort(radii_est_ijk)
            radii_est_ijk = np.sort(radii_est_ijk)
            
            radii_gt_ijk = ellipsoidParams_gt[i,j,k]['radii']
            radii_gt_sort_idx[:,i,j,k] = np.argsort(radii_gt_ijk)
            radii_gt_ijk = np.sort(radii_gt_ijk)
            
            cmap_ijk = [stim3D['x_grid_ref'][i,j,k],stim3D['y_grid_ref'][i,j,k],stim3D['z_grid_ref'][i,j,k]]
            for l in range(3):
                radii_est[l,i,j,k] = radii_est_ijk[l]
                radii_gt[l,i,j,k] = radii_gt_ijk[l]
                axes[l].scatter(2*radii_gt[l,i,j,k],radii_est[l,i,j,k], s = 100,\
                                facecolor = cmap_ijk, alpha = 0.7, linewidth = 1, edgecolor = [1,1,1])
            
for l in range(3):
    x_lb_l = np.min(radii_est[l])*0.75
    x_ub_l = np.max(radii_est[l])*1.15
    axes[l].set_xlim([x_lb_l, x_ub_l])
    axes[l].set_ylim([x_lb_l, x_ub_l])
    axes[l].grid(True, alpha=0.3)
    axes[l].set_xticks(np.round(np.linspace(x_lb_l, x_ub_l,5),3))
    axes[l].set_yticks(np.round(np.linspace(x_lb_l, x_ub_l,5),3))
    axes[l].set_aspect('equal', adjustable='box')         
    if l == 1: axes[l].set_xlabel('Axis radii (ground truth)', fontsize = 12)
    if l == 0: axes[l].set_ylabel('Axis radii (model fits)', fontsize = 12)
    if l == 0: axes[l].set_title('Short axis', fontsize = 12)
    elif l == 1:axes[l].set_title('Medium axis')
    else: axes[l].set_title('Long axis')
fig.tight_layout()
plt.show()
fig_outputDir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                        'ELPS_analysis/ModelPerformance_FigFiles/'
fig_name = 'ModelPerformance_radii' + file_name1[6:-4] + '.png'
full_path = os.path.join(fig_outputDir,fig_name)
fig.savefig(full_path)    
            
#%%
fig, axes = plt.subplots(1, 3, figsize=(14,5))
plt.rcParams['figure.dpi'] = 250
rotAngle_est = np.full((3, ref_size_dim1, ref_size_dim2, ref_size_dim3),np.nan)
rotAngle_gt = np.full(radii_est.shape, np.nan)
for i in range(ref_size_dim1):
    for j in range(ref_size_dim2):
        for k in range(ref_size_dim3):
            evecs_est_ijk = ellipsoidParams_est[i][j][k]['evecs']
            evecs_est_ijk = evecs_est_ijk[:,radii_est_sort_idx[:,i,j,k]]
            # est_rotation = R.from_matrix(evecs_est_ijk)
            # est_angles = est_rotation.as_euler('XYZ', degrees = True)
            # for m in range(3):
            #     if est_angles[m] > 90:
            #         est_angles[m] = est_angles[m] - 180
            #     elif est_angles[m] < -90:
            #         est_angles[m] = est_angles[m] + 180
            
            
            evecs_gt_ijk = ellipsoidParams_gt[i,j,k]['evecs']
            evecs_gt_ijk = evecs_gt_ijk[:,radii_gt_sort_idx[:,i,j,k]]
            gt_rotation = R.from_matrix(evecs_gt_ijk)
            gt_angles = gt_rotation.as_euler('XYZ', degrees = True)
            for m in range(3):
                if gt_angles[m] > 90:
                    gt_angles[m] = gt_angles[m] - 180
                elif gt_angles[m] < -90:
                    gt_angles[m] = gt_angles[m] + 180
            
            cmap_ijk = [stim3D['x_grid_ref'][i,j,k],stim3D['y_grid_ref'][i,j,k],stim3D['z_grid_ref'][i,j,k]]
            for l in range(3):
                rotAngle_est[l,i,j,k] = est_angles[l]
                rotAngle_gt[l,i,j,k] = gt_angles[l]
                axes[l].scatter(rotAngle_gt[l,i,j,k],rotAngle_est[l,i,j,k],\
                                color = cmap_ijk)
            
for l in range(3):
    x_lb_l = np.min(rotAngle_est[l])*0.85
    x_ub_l = np.max(rotAngle_est[l])*1.15
    axes[l].plot([x_lb_l, x_ub_l], [x_lb_l, x_ub_l], color = 'k', linestyle = '--')
    axes[l].grid(True, alpha=0.3);
    axes[l].set_aspect('equal', adjustable='box')     

