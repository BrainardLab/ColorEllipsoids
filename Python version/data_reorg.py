#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 10:32:00 2024

@author: fangfang
"""
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids")
from core import oddity_task
from analysis.ellipses_tools import fit_2d_isothreshold_contour
from analysis.ellipsoids_tools import fit_3d_isothreshold_ellipsoid
from analysis.ellipses_tools import ellParams_to_covMat, covMat3D_to_2DsurfaceSlice
from analysis.color_thres import color_thresholds

#%%
def organize_data(ndims, sim, x1_scaler, **kwargs):
    # Define default parameters with options for ellipse resolution and scaling
    params = {
        'slc_idx':[], #if we do not want to include all the ref stimuli, we can select a subset, e.g., [0, 2, 4]
        'visualize_samples':False,
        'plane_2D':''
    }
    # Update default parameters with any additional keyword arguments provided
    params.update(kwargs)
    if len(params['slc_idx']) == 0: idx = list(range(sim['rgb_comp'].shape[0]))
    else: idx = params['slc_idx']
    
    if ndims == 2: #2d case 
        #comparisom stimulus; size: (5 x 5 x 3 x 240)
        #the comparison stimulus was sampled around 5 x 5 different ref stimuli
        #the original data ranges from 0 to 1; we scale it so that it fits the [-1, 1] cube
        x1_raw = sim['rgb_comp'][idx][:,idx][:,:,sim['varying_RGBplane'],:]
        x1_raw = color_thresholds.N_unit_to_W_unit(x1_raw)
        #number of reference stimuli (5,5)
        ref_size_dim1, ref_size_dim2 = x1_raw.shape[0: ndims]
        #reshape the array so the final size is (2000, 3)
        x1_temp_reshaped    = np.transpose(x1_raw, (0,1,3,2)).reshape(-1, ndims)
    
        #reference stimulus
        xref_raw            = sim['ref_points'][sim['varying_RGBplane']][:,idx][:,:,idx]
        xref_raw = color_thresholds.N_unit_to_W_unit(xref_raw)
        xref_temp_expanded  = np.expand_dims(np.transpose(xref_raw, (1,2,0)), axis=-1)
        xref_repeated       = np.tile(xref_temp_expanded, (1, 1, 1, sim['nSims']))
        xref_temp_reshaped  = np.stack((xref_repeated[:,:,0].ravel(),\
                                        xref_repeated[:,:,1].ravel()), axis=1)
        #binary responses 
        y_temp     = jnp.array(sim['resp_binary'][idx,:][:,idx], dtype = jnp.float64)
        
    elif ndims == 3:
        x1_raw = sim['rgb_comp'][idx][:,idx][:,:,idx][:,:,:,sim['varying_RGBplane']] 
        x1_raw = color_thresholds.N_unit_to_W_unit(x1_raw)
        ref_size_dim1, ref_size_dim2, ref_size_dim3 = x1_raw.shape[0:ndims]
        x1_temp_reshaped    = np.transpose(x1_raw, (0,1,2,4,3)).reshape(-1, ndims)
        xref_raw            = sim['ref_points'][idx][:,idx][:,:,idx][:,:,:,sim['varying_RGBplane']]
        xref_raw = color_thresholds.N_unit_to_W_unit(xref_raw)
        xref_temp_expanded  = np.expand_dims(xref_raw, axis=-1)
        xref_repeated       = np.tile(xref_temp_expanded, (1, 1, 1,1, sim['nSims']))
        xref_temp_reshaped  = np.stack((xref_repeated[:,:,:,0].ravel(),\
                                        xref_repeated[:,:,:,1].ravel(),\
                                        xref_repeated[:,:,:,2].ravel()), axis=1)
        #binary responses 
        y_temp     = jnp.array(sim['resp_binary'][idx][:,idx][:,:,idx],\
                               dtype = jnp.float64)
        
    x1_jnp     = jnp.array(x1_temp_reshaped, dtype=jnp.float64)
    xref_jnp   = jnp.array(xref_temp_reshaped, dtype = jnp.float64)
    #copy the reference stimulus
    x0_jnp     = jnp.copy(xref_jnp)
    x1_jnp     = (x1_jnp - x0_jnp)*x1_scaler + x0_jnp
    y_jnp      = y_temp.ravel()

    # Package data
    data = (y_jnp, xref_jnp, x0_jnp, x1_jnp)
    print("Proportion of correct trials:", jnp.mean(y_jnp))
    
    if params['visualize_samples'] and ndims == 2:
        #figure 1: comparison stimulus & reference stimulus
        #figure 2: comparison - reference, which should be scattered around 0
        fig, ax = plt.subplots(1,2,figsize = (8,4))
        plt.rcParams['figure.dpi'] = 250 
        for i in range(ref_size_dim1*ref_size_dim2):
            i_lb = i*sim['nSims']
            i_ub = (i+1)*sim['nSims']
            #define the color map, which is the RGB value of the reference stimulus
            cm = (xref_jnp[i_lb] + 1)/2
            cm = np.insert(cm, sim['slc_RGBplane'], 0.5);
            ax[0].scatter(x1_jnp[i_lb:i_ub,0], x1_jnp[i_lb:i_ub,1], c = cm,s = 1, alpha = 0.5)
            ax[0].scatter(xref_jnp[i_lb:i_ub,0], xref_jnp[i_lb:i_ub,1],\
                          c = cm,s = 30,marker = '+')
            ax[1].scatter(x1_jnp[i_lb:i_ub,0] - xref_jnp[i_lb:i_ub,0],\
                        x1_jnp[i_lb:i_ub,1] - xref_jnp[i_lb:i_ub,1],\
                        c = cm, s = 2, alpha = 0.5)
        ax[0].set_xticks(np.unique(xref_jnp)); ax[0].set_yticks(np.unique(xref_jnp))
        ax[0].set_aspect('equal', adjustable='box')
        ax[0].set_xlim([-1, 1]); ax[0].set_ylim([-1, 1])
        ax[0].grid(True, alpha=0.3)
        ax[0].set_xlim([-1, 1]); ax[0].set_ylim([-1, 1])
        if params['plane_2D'] != '':
            ax[0].set_xlabel(params['plane_2D'][0]);
            ax[0].set_ylabel(params['plane_2D'][1])
        ax[0].set_title('Comparison stimuli')
        ax[1].set_title('Comparison - Reference')
        ax[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    return data, x1_raw, xref_raw