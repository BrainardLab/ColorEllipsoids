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
from analysis.color_thres import color_thresholds

#%%
def organize_data_sim_isoluminant_plane(sim, flag_remove_filler_col = True):
    """
    Extract and organize simulated data on the isoluminant plane for model fitting.

    This function prepares simulated data to be used with the Wishart model, 
    assuming the data was generated using the script `Simulate_probCorrectResp.py`. 
    The simulated data includes binary responses (`resp_binary`), reference stimuli 
    (`ref_points`), and comparison stimuli (`comp`).

    Parameters
    ----------
    sim : dict
        Dictionary containing simulation outputs. Expected keys are:
        - 'resp_binary': array of binary responses (0 or 1)
        - 'ref_points': array of reference stimuli (3 x N)
        - 'comp': array of comparison stimuli (3 x N)

    flag_remove_filler_col : bool, optional (default=True)
        If True, removes the last filler dimension (typically a column of 1s) 
        from reference and comparison stimuli before organizing.

    Returns
    -------
    y_jnp : jax.numpy.ndarray
        Array of binary responses, shape (N,).
    xref_jnp : jax.numpy.ndarray
        Array of reference stimuli in JAX format, shape (N, 2) or (N, 3).
    x1_jnp : jax.numpy.ndarray
        Array of comparison stimuli in JAX format, shape (N, 2) or (N, 3).
    """
    if flag_remove_filler_col:
        idx = list(range(2))  # Keep only first two dimensions
    else:
        idx = list(range(3))  # Keep all three dimensions

    y_jnp = jnp.array(sim['resp_binary'])
    xref_jnp = jnp.array(sim['ref_points'][idx, :].T)
    x1_jnp = jnp.array(sim['comp'][idx, :].T)

    return y_jnp, xref_jnp, x1_jnp

def organize_data(ndims, sim, x1_scaler, **kwargs):
    # Define default parameters with options for ellipse resolution and scaling
    params = {
        'slc_idx':[], #if we do not want to include all the ref stimuli, we can select a subset, e.g., [0, 2, 4]
        'visualize_samples':False,
        'plane_2D':'',
        'flag_convert_to_W': True,
        'M_2DWToRGB': None,
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
        if params['flag_convert_to_W']:
            x1_raw = color_thresholds.N_unit_to_W_unit(x1_raw)
        #number of reference stimuli (5,5)
        ref_size_dim1, ref_size_dim2 = x1_raw.shape[0: ndims]
        #reshape the array so the final size is (2000, 3)
        x1_temp_reshaped    = np.transpose(x1_raw, (0,1,3,2)).reshape(-1, ndims)
    
        #reference stimulus
        xref_raw            = sim['ref_points'][sim['varying_RGBplane']][:,idx][:,:,idx]
        if params['flag_convert_to_W']:
            xref_raw = color_thresholds.N_unit_to_W_unit(xref_raw)
        xref_temp_expanded  = np.expand_dims(np.transpose(xref_raw, (1,2,0)), axis=-1)
        xref_repeated       = np.tile(xref_temp_expanded, (1, 1, 1, sim['nSims']))
        xref_temp_reshaped  = np.stack((xref_repeated[:,:,0].ravel(),\
                                        xref_repeated[:,:,1].ravel()), axis=1)
        #binary responses 
        y_temp     = jnp.array(sim['resp_binary'][idx,:][:,idx], dtype = jnp.float64)
        
    elif ndims == 3:
        x1_raw = sim['rgb_comp'][idx][:,idx][:,:,idx][:,:,:,sim['varying_RGBplane']] 
        if params['flag_convert_to_W']:
            x1_raw = color_thresholds.N_unit_to_W_unit(x1_raw)
        ref_size_dim1, ref_size_dim2, ref_size_dim3 = x1_raw.shape[0:ndims]
        x1_temp_reshaped    = np.transpose(x1_raw, (0,1,2,4,3)).reshape(-1, ndims)
        xref_raw            = sim['ref_points'][idx][:,idx][:,:,idx][:,:,:,sim['varying_RGBplane']]
        if params['flag_convert_to_W']:
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
            if params['flag_convert_to_W']:
                cm = color_thresholds.W_unit_to_N_unit(xref_jnp[i_lb])
                cm = np.insert(cm, sim['slc_RGBplane'], 0.5)
            else:
                cm = np.insert(xref_jnp[i_lb], sim['slc_RGBplane'], 1)
                cm = params['M_2DWToRGB'] @ cm
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
        if params['plane_2D'] in ['GB plane', 'RG plane', 'RB plane']:
            ax[0].set_xlabel(params['plane_2D'][0]);
            ax[0].set_ylabel(params['plane_2D'][1])
        else: #suggest this is the isoluminant plane
            ax[0].set_xlabel('Wishart space dimension 1')
            ax[0].set_ylabel('Wishart space dimension 2')
        ax[0].set_title('Comparison stimuli')
        ax[1].set_title('Comparison - Reference')
        ax[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    return data, x1_raw, xref_raw