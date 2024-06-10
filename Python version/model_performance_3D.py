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
import sys
sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids')
sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/Python version')
fig_outputDir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                        'ELPS_analysis/ModelPerformance_FigFiles/'

#%% ------------------------------------------
# Load data simulated using CIELab
# ------------------------------------------
import os
import pickle
import numpy as np
from scipy.linalg import sqrtm

nSims     = 240
bandwidth = 0.005
jitter    = [0.1, 0.3, 0.5]
path_str1 = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
            'ELPS_analysis/ModelFitting_DataFiles/'
data_load = []
for j in jitter:
    file_name_j = 'Fitted_isothreshold_ellipsoids_sim'+str(nSims) +\
                'perCond_samplingNearContour_jitter'+str(j)+'_bandwidth' +\
                str(bandwidth) + '.pkl'
    full_path_j = f"{path_str1}{file_name_j}"
    os.chdir(path_str1)
    #load data 
    with open(full_path_j, 'rb') as f:
        # Load the object from the file
        data_load_j = pickle.load(f)
        data_load.append(data_load_j)

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

#%% 
def plot_estimatedW_3D(poly_order, W, idx_slc, **kwargs):
    # Default parameters for ellipsoid fitting. Can be overridden by kwargs.
    pltP = {
        'marker_alpha':0.5,
        'marker_size':80,
        'marker_color_set1': "blue",
        'marker_color_set2':[0.5,0.5,0.5],
        'marker_edgecolor_set1':[1,1,1],
        'marker_edgecolor_set2':[1,1,1],
        'xbds':[-0.04, 0.04],
        'saveFig':False,
        'figDir':'',
        'figName':'ModelEstimatedW'} 
    pltP.update(kwargs)
    if idx_slc != W.shape[0]: jitter = 0.2; 
    else: jitter = 0
    
    fig, ax = plt.subplots(1, 1, figsize=(6,4))
    ax.scatter(poly_order[:idx_slc,:idx_slc,:idx_slc,:,:] - jitter,\
                W[:idx_slc,:idx_slc,:idx_slc,:,:], s = pltP['marker_size'],\
                edgecolor = pltP['marker_edgecolor_set1'], alpha = pltP['marker_alpha'])
    if idx_slc != W.shape[0]:
        ax.scatter(poly_order[idx_slc,:,:,:,:] + jitter,\
                    W[idx_slc,:,:,:,:], s = pltP['marker_size'],\
                    color = pltP['marker_color_set2'],\
                    edgecolor = pltP['marker_edgecolor_set2'],\
                    alpha = pltP['marker_alpha'])
        ax.scatter(poly_order[:idx_slc,idx_slc,:,:,:] + jitter,\
                    W[:idx_slc,idx_slc,:,:,:], color = pltP['marker_color_set2'],\
                    s = pltP['marker_size'],edgecolor = pltP['marker_edgecolor_set2'],\
                    alpha = pltP['marker_alpha'])
        ax.scatter(poly_order[:idx_slc,:idx_slc,idx_slc,:,:] + jitter,\
                    W[:idx_slc,:idx_slc,idx_slc,:,:], color = pltP['marker_color_set2'],\
                    s = pltP['marker_size'],edgecolor = pltP['marker_edgecolor_set2'],\
                    alpha = pltP['marker_alpha'])
    ax.plot([0,np.max(poly_order)],[0,0],color = [0.5,0.5,0.5],\
            linestyle = '--', linewidth = 1)
    ax.set_yticks(np.linspace(pltP['xbds'][0], pltP['xbds'][-1],5))
    ax.set_ylim(pltP['xbds'])
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('The order of 3D Chebyshev polynomial basis function')
    ax.set_ylabel('Model estimated weight')
    if pltP['saveFig'] and pltP['figDir'] != '':
        full_path = os.path.join(fig_outputDir, pltP['figName'])
        fig.savefig(full_path)    
        

#%%
def plot_L2norm_groundtruth_vs_modelpred(gt_L2norm, modelPred_L2norm, ref_points, **kwargs):
    # Default parameters for ellipsoid fitting. Can be overridden by kwargs.
    pltP = {
        'markersize':150,
        'alpha':0.7,
        'edgecolor':[1,1,1],
        'linewidth': 1,
        'bds':[0.01, 0.05],
        'fontsize':8,
        'saveFig':False,
        'figDir':'',
        'figName':'ModelPredictions'} 
    pltP.update(kwargs)
    
    fig, ax = plt.subplots(1, 1, figsize=(5,5))
    plt.rcParams['figure.dpi'] = 250
    ax.plot([0, 1], [0,1], color = np.ones((1,3))*0.35, linestyle = '--')
    ax.scatter(gt_L2norm.flatten(),modelPred_L2norm.flatten(), s = pltP['markersize'],\
               facecolor = np.reshape(ref_points,(-1,3)),\
               alpha = pltP['alpha'], edgecolor = pltP['edgecolor'],\
               linewidth = pltP['linewidth'])
    ax.set_xlim(pltP['bds'])
    ax.set_ylim(pltP['bds'])
    ax.set_xlabel('Ground truths', fontsize = 12)
    ax.set_ylabel('Model predictions', fontsize = 12)
    ax.set_title('Average L2-norm from the\ncenter to the surface of ellipsoids')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(np.round(np.linspace(pltP['bds'][0], pltP['bds'][1],5),3))
    ax.set_yticks(np.round(np.linspace(pltP['bds'][0], pltP['bds'][1],5),3))
    ax.set_aspect('equal', adjustable='box')         
    fig.tight_layout()
    plt.show()
    if pltP['saveFig'] and pltP['figDir'] != '':
        full_path = os.path.join(fig_outputDir, pltP['figName'])
        fig.savefig(full_path)    
        
#%%
model = data_load[0]['model']
W_est = data_load[0]['W_est']
basis_degrees = (
    jnp.arange(model.degree)[:, None, None] +
    jnp.arange(model.degree)[None, :, None] + 
    jnp.arange(model.degree)[None, None, :]
)
basis_degrees_rep = np.tile(basis_degrees,(3,4,1,1,1))
basis_degrees_rep = np.transpose(basis_degrees_rep, (2,3,4,0,1))

idx_slc = 4

plot_estimatedW_3D(basis_degrees_rep, W_est, 4, saveFig = True, figDir = fig_outputDir,\
                   figName = 'ModelEstimatedW_maxDeg'+ str(idx_slc)+\
                   '_fitted_isothreshold_ellipsoids_sim'+str(nSims) +\
                   'perCond_samplingNearContour_jitter'+str(jitter[0])+\
                    '_bandwidth' + str(bandwidth) +'.png')
            
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
vec_modelPred_ellipsoids        = np.full((len(jitter),)+ vec_gt_ellipsoids.shape, np.nan)
vecLen_modelPred_ellipsoids     = np.full((len(jitter),)+ vecLen_gt_ellipsoids.shape, np.nan)
avg_vecLen_modelPred_ellipsoids = np.full((len(jitter),)+ avg_vecLen_gt_ellipsoids.shape, np.nan)
covMat_gt                       = np.full((len(jitter),)+ \
                                          stim3D['ref_points'].shape[0:3]+(3,3,), np.nan)
covMat_modelPred                = np.full(covMat_gt.shape, np.nan)
BW_distance                     = np.full((len(jitter),)+\
                                          stim3D['ref_points'].shape[0:3], np.nan)
             
for j in range(len(jitter)):
    vec_modelPred_ellipsoids[j]     = (data_load[j]['recover_fitEllipsoid_unscaled']+1)/2 -\
                                                 stim3D['ref_points'][:,:,:,:,None]
    vecLen_modelPred_ellipsoids[j]     = np.linalg.norm(vec_modelPred_ellipsoids[j], axis = 3)
    avg_vecLen_modelPred_ellipsoids[j] = 2* np.sum(vecLen_modelPred_ellipsoids[j],axis = -1)\
                                        /vecLen_modelPred_ellipsoids[j].shape[-1]
                                        
    #stim3D['ref_points']
    plot_L2norm_groundtruth_vs_modelpred(avg_vecLen_gt_ellipsoids,\
        avg_vecLen_modelPred_ellipsoids[j], stim3D['ref_points'], saveFig = False,\
        figName = 'ModelPerformance_avgVecLength' + file_name_j[j][6:-4] + '.png')
        
    #% the Bures-Wasserstein distance
    for ii in range(stim3D['ref_points'].shape[0]):
        for jj in range(stim3D['ref_points'].shape[1]):
            for kk in range(stim3D['ref_points'].shape[2]):
                covMat_gt[j,ii,jj,kk] = results3D['ellipsoidParams'][ii,jj,kk]['evecs'] @\
                    np.diag(results3D['ellipsoidParams'][ii,jj,kk]['radii']*2) @\
                    results3D['ellipsoidParams'][ii,jj,kk]['evecs'].T
                    
                covMat_modelPred[j,ii,jj,kk] = data_load[j]['params_ellipsoids'][ii][jj][kk]['evecs'] @\
                    np.diag(data_load[j]['params_ellipsoids'][ii][jj][kk]['radii']) @\
                    data_load[j]['params_ellipsoids'][ii][jj][kk]['evecs'].T
                BW_distance[j,ii,jj,kk] = np.sqrt(np.trace(covMat_gt[j,ii,jj,kk]) +\
                    np.trace(covMat_modelPred[j,ii,jj,kk])-2 * \
                    np.trace(sqrtm(sqrtm(covMat_gt[j,ii,jj,kk]) @\
                    covMat_modelPred[j,ii,jj,kk] @ sqrtm(covMat_gt[j,ii,jj,kk]))))
                    
#%%
y_ub = 25
lgd_list = ['small', 'medium', 'large']
cmap = np.array([[247,152,29],[75,40,73],[107,142,35]])/255
fig, ax = plt.subplots(1, 1, figsize=(7,4.5))
plt.rcParams['figure.dpi'] = 250
for j in range(len(jitter)):
    ax.hist(BW_distance[j].flatten(), bins = 50, range = [0,0.05], color = cmap[j],\
            alpha = 0.6, edgecolor = [1,1,1], label = str(jitter[j]) + ' ('+\
            lgd_list[j]+' noise)')
    ax.plot([np.median(BW_distance[j].flatten()),np.median(BW_distance[j].flatten())],\
             [0,y_ub],color = cmap[j], linestyle = '--')
ax.grid(True, alpha=0.3)
ax.legend(title = 'Amount of jitter added to\nsampled comparison stimuli')
ax.set_ylim([0,y_ub])
ax.set_xlabel('The Bures-Wasserstein distance')
ax.set_ylabel('Frquency')
full_path = os.path.join(fig_outputDir, 'ModelPerformance_BuresWassersteinDistance_3Dellipsoids_jitters.png')
#fig.savefig(full_path)          







