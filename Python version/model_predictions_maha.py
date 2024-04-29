#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 15:23:13 2024

@author: fangfang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 11:25:41 2024

@author: fangfang
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import warnings
import sys

sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids")
from core import oddity_task_mahalanobis

sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/Python version/")
from simulations_CIELab import fit_2d_isothreshold_contour


def convert_Sig_2DisothresholdContour_oddity(rgb_ref, varying_RGBplane,
                                             grid_theta_xy, vecLength,
                                             pC_threshold, W, model, scaler_x1, **kwargs):
    """
    This function simulates an oddity task based on chromatic directions and 
    computes the resulting isothreshold contour ellipse fit for comparison 
    stimuli. It evaluates and estimates the chromatic difference that reaches 
    a perceptual threshold.

    Parameters
    ----------
    rgb_ref (size: (3,)): the RGB values of the reference stimulus.

    varying_RGBplane (list of int): A list of two integers between 0 and 2 
        inclusive, specifying the RGB planes that are varying in the simulations.
    
    grid_theta_xy (size: (2, numDirPts)): the chromatic directions in which the
        RGB values are varied.
        
    vecLength: A 1D array with length equal to nSteps_bruteforce, representing 
        the magnitudes of the vector lengths to test in each chromatic direction.
        
    pC_threshold: A float specifying the threshold probability at which the 
        oddity task changes its decision criterion.
        
    W (size: (# Degree of the polynomial basis functions 
            x # Degree of the polynomial basis functions 
            x # stimulus dimensions
            x # Number of extra inner dimensions in `U`.)): estimated weigths

    Returns
    -------
    fitEllipse_scaled: A 2D array (2, nThetaEllipse) containing the coordinates 
        of the scaled fitted ellipse.
        
    fitEllipse_unscaled: A 2D array (2, nThetaEllipse) containing the 
        coordinates of the unscaled fitted ellipse.
    
    rgb_comp_scaled: A 2D array (2, numDirPts) containing the scaled RGB 
        components of the comparison stimuli.
    
    rgb_contour_cov: A 2x2 matrix representing the covariance of the RGB contour.
    
    [xCenter, yCenter, majorAxis, minorAxis, theta]: parameters of ellipses
    
    """
    
    # Define default parameters with options for ellipse resolution and scaling
    params = {
        'nThetaEllipse': 200, # Number of angular steps to compute the ellipse
        'contour_scaler': 1/scaler_x1, # Scale factor for determining contour size
        'opt_key':jax.random.PRNGKey(444),
        'mc_samples': 1e3,
        'bandwidth':1e-3,
    }
    # Update default parameters with any additional keyword arguments provided
    params.update(kwargs)
    
    # Number of brute force steps based on vector length
    nSteps_bruteforce    = len(vecLength)
    # Number of direction points in theta grid
    numDirPts            = grid_theta_xy.shape[1]
    
    # Extract and reshape the reference RGB values of the varying plane for processing
    rgb_ref_s = jnp.array(rgb_ref[varying_RGBplane]).reshape((1,2))
    Uref      = model.compute_U(W, rgb_ref_s)
    U0        = model.compute_U(W, rgb_ref_s)
    
    #initialize
    recover_vecLength    = np.full((numDirPts), np.nan)
    recover_rgb_comp_est = np.full((numDirPts, 3), np.nan)
    
    #for each chromatic direction
    for i in range(numDirPts):
        #determine the chromatic direction we are going
        vecDir = jnp.array(grid_theta_xy[:,i]).reshape((1,2))
        # Initialize probability of choosing x1 (the comparison stimulus
        # that is actually different from xref)
        pChoosingX1 = np.full((nSteps_bruteforce), np.nan)
        for x in range(nSteps_bruteforce):
            # Calculate RGB composition for current vector length
            rgb_comp = rgb_ref_s + vecDir * vecLength[x]
            U1 = model.compute_U(W, rgb_comp)
            # Simulate the oddity task trial and compute the signed difference
            signed_diff = oddity_task_mahalanobis.simulate_oddity_one_trial(\
                (rgb_ref_s[0], rgb_ref_s[0], rgb_comp[0], Uref[0],\
                 U0[0], U1[0]), params['opt_key'], params['mc_samples'],\
                 model, W)
            # Approximate the cumulative distribution function for the trial
            pChoosingX1[x] = oddity_task_mahalanobis.approx_cdf_one_trial(\
                0.0, signed_diff, params['bandwidth'])
            
        # find the index that corresponds to the minimum 
        #|pChoosingX1 - pC_threshold|
        min_idx = np.argmin(np.abs(pChoosingX1 - pC_threshold))
        # Warn and exit if the minimal index is at the bounds of the search 
        #range
        if min_idx in [0, nSteps_bruteforce-1]:
            print(min_idx)
            warnings.warn('Expand the range for grid search!')
            return
        # Store the vector length corresponding to the minimal index
        recover_vecLength[i] = vecLength[min_idx]
        # Compute and store the comparison RGB component estimate
        recover_rgb_comp_est[i, varying_RGBplane] = rgb_ref_s + \
            params['contour_scaler'] * vecDir * recover_vecLength[i]
    
    #fit an ellipse to the estimated comparison stimuli
    fitEllipse_scaled, fitEllipse_unscaled, rgb_comp_scaled, \
        rgb_contour_cov, [xCenter, yCenter, majorAxis, minorAxis, theta] = \
            fit_2d_isothreshold_contour(rgb_ref, [], grid_theta_xy, \
                vecLength = recover_vecLength, varyingRGBplan = varying_RGBplane)

    return fitEllipse_scaled, fitEllipse_unscaled, rgb_comp_scaled, \
        rgb_contour_cov,[xCenter, yCenter, majorAxis, minorAxis, theta]   
            
def convert_Sig_2DisothresholdContour_oddity_batch(rgb_ref, varying_RGBplane,
                                                   grid_theta_xy, target_pC,
                                                   W, model,opt_vecLen, **kwargs):
    """ 
    This function converts the estimated covariance matrix of the Wishart distribution 
    to the parameters of the isothreshold contour ellipse for the oddity task. It 
    computes the fitted ellipse for each comparison stimulus and returns the 
    parameters of the ellipse.

    Args:
        rgb_ref (array-like): The RGB values of the reference stimulus.
        varying_RGBplane (list of int): A list of two integers between 0 and 2 inclusive, 
            specifying the RGB planes that are varying in the simulations.
        grid_theta_xy (array-like): The chromatic directions in which the RGB values are varied.
        target_pC (float): The threshold probability at which the oddity task changes its decision criterion.
        W (array-like): The estimated weights of the model.
        model (object): The model object for computing U.
        opt_vecLen (array-like): The optimal vector length for each comparison stimulus.

    Returns:
        fitEllipse_scaled (array-like): The coordinates of the scaled fitted ellipse.
        fitEllipse_unscaled (array-like): The coordinates of the unscaled fitted ellipse.
        rgb_comp_scaled (array-like): The scaled RGB components of the comparison stimuli.
        rgb_contour_cov (array-like): The covariance matrix of the RGB contour.
        [xCenter, yCenter, majorAxis, minorAxis, theta] (array-like): The parameters of the ellipses.
    """
    # Define default parameters with options for ellipse resolution and scaling
    params = {
        'nTheta': 200, # Number of angular steps to compute the ellipse
        'scaler_x1': 5, # Scale factor for determining contour size
        'ngrid_bruteforce': 500, # Number of brute force steps based on vector length
        'scaler_bds_bruteforce': [0.5, 3], # Bounds for vector length search
        'mc_samples': 1e3, # Number of Monte Carlo samples for approximating CDF
        'bandwidth':1e-3, # Bandwidth for approximating CDF
        'opt_key':jax.random.PRNGKey(444), # Random key for optimization
    }
    # Update default parameters with any additional keyword arguments provided
    params.update(kwargs)
    
    num_grid_pts1, num_grid_pts2 = rgb_ref.shape[1], rgb_ref.shape[2]
    fixing_RGBplane = list(range(3))
    for items in varying_RGBplane: fixing_RGBplane.remove(items)
    params_ellipses             = [[]]*num_grid_pts1
    recover_fitEllipse_scaled   = np.full((num_grid_pts1, num_grid_pts2, 2, params['nTheta']),\
                                        np.nan)
    recover_fitEllipse_unscaled = np.full(recover_fitEllipse_scaled.shape, np.nan)
    recover_rgb_comp_scaled     = np.full((num_grid_pts1, num_grid_pts2, 2,\
                                        grid_theta_xy.shape[-1]), np.nan)
    recover_rgb_contour_cov     = np.full((num_grid_pts1, num_grid_pts2, 2, 2), np.nan)

    #for each reference stimulus
    for i in range(num_grid_pts1):
        print(i)
        params_ellipses[i] = [[]]*num_grid_pts2
        for j in range(num_grid_pts2):
            #first grab the reference stimulus' RGB
            rgb_ref_scaled_ij = rgb_ref[:,i,j]
            #insert the fixed R/G/B value to the corresponding plane
            rgb_ref_scaled_ij = np.insert(rgb_ref_scaled_ij, fixing_RGBplane[0], 0)
            #from previous results, get the optimal vector length
            vecLength_ij = opt_vecLen[varying_RGBplane,i,j]
            #use brute force to find the optimal vector length
            vecLength_test = np.linspace(\
                np.min(vecLength_ij)*params['scaler_x1']*params['scaler_bds_bruteforce'][0],\
                np.max(vecLength_ij)*params['scaler_x1']*params['scaler_bds_bruteforce'][1],\
                params['ngrid_bruteforce']) 
            
            #fit an ellipse to the estimated comparison stimuli
            recover_fitEllipse_scaled[i,j], recover_fitEllipse_unscaled[i,j],\
                recover_rgb_comp_scaled[i,j], recover_rgb_contour_cov[i,j],\
                params_ellipses[i][j] = convert_Sig_2DisothresholdContour_oddity(\
                rgb_ref_scaled_ij, varying_RGBplane,grid_theta_xy,\
                vecLength_test, target_pC, W, model, params['scaler_x1'],\
                nThetaEllipse = params['nTheta'], opt_key = params['opt_key'],
                mc_samples = params['mc_samples'], bandwidth = params['bandwidth'])
    return recover_fitEllipse_scaled, recover_fitEllipse_unscaled,\
        recover_rgb_comp_scaled, recover_rgb_contour_cov, params_ellipses  