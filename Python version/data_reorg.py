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
from dataclasses import replace

#sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids")
sys.path.append("/Users/fh862-adm/Documents/GitHub/ellipsoids/ellipsoids")
from analysis.color_thres import color_thresholds
from core import model_predictions
#sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/Python version/')
sys.path.append("/Users/fh862-adm/Documents/GitHub/ColorEllipsoids/Python version")
from plotting.wishart_plotting import PlotSettingsBase
from plotting.trial_placement_nonadaptive_plotting import TrialPlacementVisualization,\
        Plot3DSampledCompSettings

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

def organize_data_2d_fixed_grid(ndims, sim, x1_scaler, **kwargs):
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
        
    x1_jnp     = jnp.array(x1_temp_reshaped, dtype=jnp.float64)
    xref_jnp   = jnp.array(xref_temp_reshaped, dtype = jnp.float64)
    #copy the reference stimulus
    x0_jnp     = jnp.copy(xref_jnp)
    x1_jnp     = (x1_jnp - x0_jnp)*x1_scaler + x0_jnp
    y_jnp      = y_temp.ravel()

    # Package data
    data = (y_jnp, xref_jnp, x0_jnp, x1_jnp)
    print("Proportion of correct trials:", jnp.mean(y_jnp))
    
    if params['visualize_samples']:
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

def visualize_data_3d_fixed_grid(sim_trial, fixed_val = 0.5):
    pltSettings_base = PlotSettingsBase(fig_dir= None, fontsize = 12)
    plt3DSettings = replace(Plot3DSampledCompSettings(), **pltSettings_base.__dict__)
    sim_vis = TrialPlacementVisualization(
        sim_trial, settings = pltSettings_base, save_fig=False
    )
    for fixed_dim in 'RGB':
        ttl = 'RGB plane'
        ttl_new = ttl.replace(fixed_dim,'')
        #plot the transformation
        plt3DSettings = replace(plt3DSettings, 
                                bds= 0.12,
                                slc_grid_ref_dim1 = list(range(0,5,2)), #list(range(0,5,1))
                                slc_grid_ref_dim2 = list(range(0,5,2)),
                                title = ttl_new)
    
        sim_vis.plot_3D_sampledComp(sim_trial.gt_CIE_stim['grid_ref'], 
                                    sim_trial.gt_CIE_results['fitEllipsoid_unscaled'],\
                                    sim_trial.sim['comp'],
                                    fixed_dim, fixedPlaneVal= fixed_val,
                                    settings = plt3DSettings,
                                    save_fig= False)

def organize_data_3d_fixed_grid(sim, slc_idx=None):
    """ 
    Prepare 3D simulated data (from CIELab) for fitting the Wishart model.
    
    Input shapes:
        - sim['comp']        : (N, N, N, 3, M)
        - sim['ref_points']  : (N, N, N, 3)
        - sim['resp_binary'] : (N, N, N, M)
    
    Output shapes:
        - x1_jnp, xref_jnp, x0_jnp: ((K³ × M), 3)
        - y_jnp                  : (K³ × M,)
    
    slc_idx : optional list of indices to subsample the cube grid (0-based)
    """

    ndims = 3
    n_grid = sim['comp'].shape[0]

    # Determine which indices to slice (default: use full cube)
    idx = np.arange(n_grid) if slc_idx is None else np.array(slc_idx)
    if np.any(idx < 0) or np.any(idx >= n_grid):
        raise ValueError("slc_idx contains out-of-bound indices.")

    # Extract relevant slices from the 5D comparison stimuli and convert to W space
    x1 = sim['comp'][np.ix_(idx, idx, idx)]
    # Convert simulated data from Normalized space (bounded between 0 and 1)
    # to Wishart space (bounded between -1 and 1)
    x1 = color_thresholds.N_unit_to_W_unit(x1)
    x1_reshaped = np.transpose(x1, (0, 1, 2, 4, 3)).reshape(-1, ndims)

    # Extract and repeat the reference stimuli for each trial, convert to W space
    xref = sim['ref_points'][np.ix_(idx, idx, idx)]
    xref = color_thresholds.N_unit_to_W_unit(xref)
    xref_repeated = np.repeat(xref[..., np.newaxis], sim['nSims'], axis=-1)
    xref_reshaped = np.transpose(xref_repeated, (0, 1, 2, 4, 3)).reshape(-1, ndims)

    # Flatten the binary responses
    y = sim['resp_binary'][np.ix_(idx, idx, idx)].ravel()

    # Convert to JAX arrays
    x1_jnp   = jnp.array(x1_reshaped, dtype=jnp.float64)
    xref_jnp = jnp.array(xref_reshaped, dtype=jnp.float64)
    x0_jnp   = jnp.copy(xref_jnp)
    y_jnp    = jnp.array(y, dtype=jnp.float64)

    print("Proportion of correct trials:", jnp.mean(y_jnp))

    return (y_jnp, xref_jnp, x0_jnp, x1_jnp), x1, xref

def derive_gt_slice_2d_ellipse_CIE(num_grid_pts, CIE_results_3D, flag_convert_to_W = True):
    # Initialize 3D covariance matrices for ground truth and predictions
    gt_covMat_CIE   = np.full((num_grid_pts, num_grid_pts, num_grid_pts, 3, 3), np.nan)
    
    if flag_convert_to_W: scaler = 2
    else: scaler = 1
    # Loop through each reference color in the 3D space
    for g1 in range(num_grid_pts):
        for g2 in range(num_grid_pts):
            for g3 in range(num_grid_pts):
                #Convert the ellipsoid parameters to covariance matrices for the 
                #ground truth
                gt_covMat_CIE[g1,g2,g3] = model_predictions.ellParams_to_covMat(\
                                scaler * CIE_results_3D['ellipsoidParams'][g1,g2,g3]['radii'],\
                                CIE_results_3D['ellipsoidParams'][g1,g2,g3]['evecs'])
    # Compute the 2D ellipse slices from the 3D covariance matrices for both ground 
    #truth and predictions
    gt_slice_2d_ellipse_CIE = model_predictions.covMat3D_to_2DsurfaceSlice(gt_covMat_CIE)
    
    return gt_slice_2d_ellipse_CIE, gt_covMat_CIE

def group_trials_by_grid(grid, y_jnp, xref_jnp, x1_jnp):
    """
    Groups trials by grid point and reshapes outputs.

    Args:
        grid: (num_x, num_y, 2) array of grid coordinates (x, y per grid point)
        y_jnp: (N,) array of binary responses
        xref_jnp: (N, 2) array of reference stimuli per trial
        x1_jnp: (N, 2) array of comparison stimuli per trial

    Returns:
        y_org:      (num_x, num_y, N_per) grouped responses
        xref_org:   (num_x, num_y, N_per, 2) grouped reference stimuli
        x1_org:     (num_x, num_y, N_per, 2) grouped comparison stimuli
        y_flat:     (num_x*num_y, N_per) flattened responses
        xref_flat:  (num_x*num_y, N_per, 2) flattened references
        x1_flat:    (num_x*num_y, N_per, 2) flattened comparisons
    """
    num_x, num_y, _ = grid.shape
    N = y_jnp.shape[0]
    N_per = N // (num_x * num_y)

    # Preallocate output arrays (NumPy, will convert to jnp later)
    xref_org = np.full((num_x, num_y, N_per, 2), np.nan)
    x1_org   = np.full_like(xref_org, np.nan)
    y_org    = np.full((num_x, num_y, N_per), np.nan)

    for i in range(num_x):
        for j in range(num_y):
            grid_pt = grid[i, j]
            # Find indices where reference matches grid point (use isclose for float precision)
            match = jnp.all(jnp.isclose(xref_jnp, grid_pt), axis=1)
            idxs = np.where(match)[0]
            # Fill preallocated arrays (up to available matches)
            xref_org[i, j, :len(idxs)] = np.asarray(xref_jnp[idxs])
            x1_org[i, j, :len(idxs)] = np.asarray(x1_jnp[idxs])
            y_org[i, j, :len(idxs)] = np.asarray(y_jnp[idxs])

    # Flatten arrays along grid dimensions
    xref_flat = np.reshape(xref_org, (num_x * num_y, N_per, 2))
    x1_flat   = np.reshape(x1_org, (num_x * num_y, N_per, 2))
    y_flat    = np.reshape(y_org, (num_x * num_y, N_per))

    # Convert outputs to jax.numpy arrays
    return (jnp.array(y_org), jnp.array(xref_org), jnp.array(x1_org)),\
           (jnp.array(y_flat), jnp.array(xref_flat), jnp.array(x1_flat))
