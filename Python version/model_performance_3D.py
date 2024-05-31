#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 00:39:55 2024

@author: fangfang
"""

import jax
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids')
sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/Python version')

#%% ------------------------------------------
# Load data simulated using CIELab
# ------------------------------------------
import os
import pickle
import numpy as np

nSims = 240
jitter = 0.5
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
_, stim3D, results3D = data_load2[0], data_load2[1], data_load2[2]
            
#%% evalutate model performance
"""
calculate the average distance from the center to the surface of the ellipsoid 
using 20,000 evenly sampled vectors. Then, I compare these averages between the 
model predictions and the ground truths.
"""
#ground truth
#subtract the center for each ellipsoid
vec_gt_ellipsoids        = results3D['fitEllipsoid_unscaled'] -\
                            stim3D['ref_points'][:,:,:,:,None]
#compute L2 norm
vecLen_gt_ellipsoids     = np.linalg.norm(vec_gt_ellipsoids, axis = 3)
#compute the average across all the evenly spaced vectors
#We multiply by 2 here because results3D['fitEllipsoid_unscaled'] is in the RGB
#space within [0,1], after transformation, it's in the new space within [-1,1]
#i.e., each vector is stretched as twice as long
avg_vecLen_gt_ellipsoids = 2*np.sum(vecLen_gt_ellipsoids, axis = -1)\
                            /vecLen_gt_ellipsoids.shape[-1]

#model predictions
#data_load1['recover_fitEllipsoid_unscaled'] is already in the transformed space
#within [-1,1], so we first need to transform it back to be in the same RGB space
#as the reference points
vec_modelPred_ellipsoids        = (data_load1['recover_fitEllipsoid_unscaled']+1)/2 -\
                                             stim3D['ref_points'][:,:,:,:,None]
vecLen_modelPred_ellipsoids     = np.linalg.norm(vec_modelPred_ellipsoids, axis = 3)
avg_vecLen_modelPred_ellipsoids = 2* np.sum(vecLen_modelPred_ellipsoids,axis = -1)\
                                    /vecLen_modelPred_ellipsoids.shape[-1]

#diff_vecLen     = np.abs(vecLen_gt_ellipsoids - vecLen_modelPred_ellipsoids)
#avg_all_diff_vecLen = np.mean(diff_vecLen)
#std_all_diff_vecLen = np.std(diff_vecLen.flatten())
                                             
#%%
fig, ax = plt.subplots(1, 1, figsize=(5,5))
plt.rcParams['figure.dpi'] = 250
ax.plot([0, 1], [0,1], color = np.ones((1,3))*0.35, linestyle = '--')
ax.scatter(avg_vecLen_gt_ellipsoids.flatten(),\
           avg_vecLen_modelPred_ellipsoids.flatten(), s = 150,\
           facecolor = np.reshape(stim3D['ref_points'],(-1,3)),\
           alpha = 0.7, edgecolor = [1,1,1], linewidth = 1)
#for i in range(5):
#    for j in range(5):
#        for k in range(5):
#            ax.scatter(2*vecLen_gt_ellipsoids[i,j,k][0::1000],\
#                       2*vecLen_modelPred_ellipsoids[i,j,k][0::1000], s = 50,\
#                       facecolor = stim3D['ref_points'][i,j,k],\
#                       alpha = 0.7) #edgecolor = [1,1,1] linewidth = 1
            
x_lb_l = 0.01#np.min(avg_vecLen_modelPred_ellipsoids)*0.75
x_ub_l = 0.05#np.max(avg_vecLen_modelPred_ellipsoids)*1.15
ax.set_xlim([x_lb_l, x_ub_l])
ax.set_ylim([x_lb_l, x_ub_l])
ax.set_xlabel('Ground truths', fontsize = 12)
ax.set_ylabel('Model predictions', fontsize = 12)
ax.set_title('Average L2-norm from the\ncenter to the surface of ellipsoids')
ax.grid(True, alpha=0.3)
ax.set_xticks(np.round(np.linspace(x_lb_l, x_ub_l,5),3))
ax.set_yticks(np.round(np.linspace(x_lb_l, x_ub_l,5),3))
ax.set_aspect('equal', adjustable='box')         

fig.tight_layout()
plt.show()
fig_outputDir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                        'ELPS_analysis/ModelPerformance_FigFiles/'
fig_name = 'ModelPerformance_avgVecLength' + file_name1[6:-4] + '.png'
full_path = os.path.join(fig_outputDir,fig_name)
#fig.savefig(full_path)    
    
    
    
    
    

