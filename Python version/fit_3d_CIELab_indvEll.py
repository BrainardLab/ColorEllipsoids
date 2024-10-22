#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 16:54:24 2024

@author: fangfang
"""

#%% import modules
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import dill as pickle
import sys
import numpy as np

sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids")
from analysis.color_thres import color_thresholds
from plotting.wishart_predictions_plotting import WishartPredictionsVisualization
# Import functions and classes from your project
from core.probability_surface import IndividualProbSurfaceModel, optimize_nloglikelihood
from analysis.ellipses_tools import covMat3D_to_2DsurfaceSlice, ellParams_to_covMat
from analysis.ellipsoids_tools import UnitCircleGenerate_3D, PointsOnEllipsoid, \
    rotation_angles_to_eigenvectors,fit_3d_isothreshold_ellipsoid

sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/Python version')
from data_reorg import organize_data
from plotting.trial_placement_nonadaptive_plotting import TrialPlacementVisualization

#%% three variables we need to define for loading the data
for rr in range(9,10):
    rnd_seed  = rr
    nSims     = 4800
    jitter    = 0.3
    
    base_dir = '/Volumes/T9/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'
    output_figDir_fits = base_dir +'ELPS_analysis/ModelFitting_FigFiles/Python_version/3D_oddity_task_indvEll/'
    output_fileDir = base_dir + 'ELPS_analysis/ModelFitting_DataFiles/3D_oddity_task_indvEll/'
    
    #% ------------------------------------------
    # Load data simulated using CIELab
    # ------------------------------------------
    #file 1
    COLOR_DIMENSION = 3
    path_str = base_dir + 'ELPS_analysis/Simulation_DataFiles/'
    # Create an instance of the class
    color_thres_data = color_thresholds(COLOR_DIMENSION, base_dir + 'ELPS_analysis/')
    # Load Wishart model fits
    color_thres_data.load_CIE_data()  
    stim3D = color_thres_data.get_data('stim3D', dataset='CIE_data')
    results3D = color_thres_data.get_data('results3D', dataset='CIE_data')
    
    #simulation files
    file_sim = f'Sims_isothreshold_ellipsoids_sim{nSims}perCond_samplingNearContour_'+\
                f'jitter{jitter}_seed{rnd_seed}.pkl'
    full_path = f"{path_str}{file_sim}"
    with open(full_path, 'rb') as f: data_load = pickle.load(f)
    sim = data_load[0]
    
    """
    Fitting would be easier if we first scale things up, and then scale the model 
    predictions back down
    """
    idx_trim = list(range(5))
    scaler_x1  = 5
    #x1_raw is unscaled
    data, x1_raw, xref_raw = organize_data(COLOR_DIMENSION, sim,\
            scaler_x1, slc_idx = idx_trim, visualize_samples = False)
    # unpackage data
    ref_size_dim1, ref_size_dim2, ref_size_dim3 = x1_raw.shape[0:3]
    # if we run oddity task with the reference stimulus fixed at the top
    y_jnp_flat, xref_jnp_flat, x0_jnp_flat, x1_jnp_flat = data 
    nRefs    = y_jnp_flat.shape[0]//nSims
    y_jnp    = jnp.reshape(y_jnp_flat, (nRefs, nSims))
    xref_jnp = jnp.reshape(xref_jnp_flat, (nRefs, nSims, COLOR_DIMENSION))
    x1_jnp   = jnp.reshape(x1_jnp_flat, (nRefs, nSims,COLOR_DIMENSION))
    data_new = (y_jnp, xref_jnp[:,0,:], x1_jnp)
    
    #% Visualize the simulated data again
    fixedRGB_val_full = np.linspace(0.2,0.8,5)
    fixedRGB_val = fixedRGB_val_full[idx_trim]
    
    for fixedPlane, varyingPlanes in zip(['R','G','B'], ['GB','RB','RG']):
        for val in fixedRGB_val:
            TrialPlacementVisualization.plot_3D_sampledComp(stim3D['grid_ref'][idx_trim]*2-1, 
                results3D['fitEllipsoid_unscaled'][idx_trim][:,idx_trim][:,:,idx_trim]*2-1,
                x1_raw, fixedPlane, val*2-1, slc_grid_ref_dim1 = [0,1,2], 
                slc_grid_ref_dim2 = [0,1,2], surf_alpha =  0.1, 
                samples_alpha = 0.1,scaled_neg12pos1 = True,
                bds = 0.05,title = varyingPlanes+' plane',
                saveFig = False, figDir = path_str[0:-10] + 'FigFiles/',\
                figName =f"{file_sim}_{varyingPlanes}plane_fixedVal{val}")
    
    #%--------------------------------------------
    # Fit the independent threshold contour model
    # ---------------------------------------------
    NUM_GRID_PTS = len(idx_trim)
    # Initialize an instance of IndividualProbSurfaceModel
    model_indvEll = IndividualProbSurfaceModel(NUM_GRID_PTS,  # grid points
                                            [1e-2,0.3],       # Bounds for radii
                                            [0, 2*jnp.pi],    # Bounds for angle in radians
                                            [0.5, 5],         # Bounds for Weibull parameter 'a'
                                            [0.1, 5],         # Bounds for Weibull parameter 'b'
                                            ndims = COLOR_DIMENSION) #color dimension
    
    # Fix parameters for the Weibull function for fitting stability
    weibull_params = jnp.array([sim['alpha'], sim['beta']])  
    #a = 1.17; b = 2.33    
    
    nReps = 3
    KEY_list = np.array(list(range(nReps)))+rnd_seed*100
    total_steps = 50000
    objhist = np.full((total_steps,), 1)
    for k in KEY_list:    
        print(f'Reptition {k}:')       
        # Initialize random key                          
        KEY_k = jax.random.PRNGKey(k)  
        # Sample initial parameters for the model
        init_params = model_indvEll.sample_params_prior(KEY_k)  
        # fix weibull parameters
        init_params = init_params.at[:,-2:].set(weibull_params[:,np.newaxis].T)
        
        # Run optimization to recover the best-fit parameters
        params_recover_k, iters, objhist_k = optimize_nloglikelihood(
            init_params, 
            data_new, 
            ndims = COLOR_DIMENSION,
            total_steps=total_steps,              # Number of total optimization steps
            save_every=10,                        # Save the objective value every 10 steps
            fixed_weibull_params=weibull_params,  # Fix the Weibull parameters during optimization
            bds_radii = model_indvEll.bds_radii,  # Boundaries for radii (ellipse/ellipsoid size parameters)
            bds_angle = model_indvEll.bds_angle,  # Boundaries for angles (rotation in radians)
            learning_rate = 5e-1,                 # Set the learning rate for the optimizer
            show_progress =True                   # Show progress using tqdm
        )
        if objhist_k[-1] < objhist[-1]: 
            objhist = objhist_k
            params_recover = params_recover_k
    
    # Plot the optimization history (objective value vs iterations)
    fig, ax = plt.subplots(1, 1)
    ax.plot(iters, objhist)  # Plot iterations vs objective history
    fig.tight_layout()
    
    #%
    # -----------------------------
    # Compute model predictions
    # -----------------------------
    # Recover ellipses from the optimized parameters
    nTheta = 200
    nPhi = 100
    #create a unit sphere
    unitEll_finer = UnitCircleGenerate_3D(nTheta, nPhi)
    #initialize recovered eilipsoid
    xyz_recover = np.full((nRefs, COLOR_DIMENSION, nTheta*nPhi), np.nan)
    for i in range(nRefs):
        # use rotation angles to compute eigenvectors
        eigvec_i = rotation_angles_to_eigenvectors(*params_recover[i,3:6])
        # Reconstruct the recovered ellipses using the optimized parameters
        xyz_recover_i = PointsOnEllipsoid(params_recover[i,0:3],  # Semi-major and semi-minor axes
                                         xref_jnp[i,0][:,np.newaxis],# Center of ellipsoid
                                         eigvec_i, 
                                         unitEll_finer)       
        xyz_recover[i] = xyz_recover_i
    #reshape
    fitEll_unscaled = np.reshape(xyz_recover,(NUM_GRID_PTS, NUM_GRID_PTS, NUM_GRID_PTS,
                                              COLOR_DIMENSION, nTheta*nPhi))    
    
    #fit an ellipse, get paramQ
    grid_1d = jnp.linspace(jnp.min(xref_jnp), jnp.max(xref_jnp), NUM_GRID_PTS)
    grid = jnp.stack(jnp.meshgrid(*[grid_1d for _ in range(COLOR_DIMENSION)]), axis=-1)
    grid_trans = np.transpose(grid,(1,0,2,3))
    
    #In this section, we want to 
    ell_paramsQ = []
    fitEll_scaled = np.full(fitEll_unscaled.shape, np.nan)
    covMat_recover_grid = np.full((NUM_GRID_PTS, NUM_GRID_PTS, NUM_GRID_PTS,
                                   COLOR_DIMENSION, COLOR_DIMENSION),np.nan)
    for i in range(NUM_GRID_PTS):
        ell_paramsQ_i = []
        for j in range(NUM_GRID_PTS):
            ell_paramsQ_j = []
            for k in range(NUM_GRID_PTS):
                rgb_comp_ijk = np.reshape(np.transpose(fitEll_unscaled[i,j,k],(1,0)),
                                          (nPhi, nTheta, COLOR_DIMENSION))
                fitEll_scaled_ijk, _, _, _, ellP_ijk = \
                    fit_3d_isothreshold_ellipsoid(grid[i,j,k], [], rgb_comp_ijk, 
                                                scaler_x1= 1/scaler_x1)
                fitEll_scaled[i,j,k] = fitEll_scaled_ijk
                covMat_recover_grid_ijk = ellParams_to_covMat(ellP_ijk['radii'],
                                                              ellP_ijk['evecs'])
                covMat_recover_grid[i,j,k] = covMat_recover_grid_ijk
                ell_paramsQ_j.append(ellP_ijk)
            ell_paramsQ_i.append(ell_paramsQ_j)
        ell_paramsQ.append(ell_paramsQ_i)
        
    #class for model prediction
    class model_pred:
        def __init__(self, M, fitEll_unscaled, fitEll_scaled, params_ell, target_pC = 2/3):
            self.fitM = M
            self.fitEll_unscaled = fitEll_unscaled
            self.fitEll_scaled = fitEll_scaled
            self.params_ell = params_ell
            self.target_pC = target_pC
    model_pred_indvEll = model_pred(params_recover, fitEll_unscaled, fitEll_scaled, ell_paramsQ)
    model_pred_indvEll.covMat_recover_grid = covMat_recover_grid
    
    #%
    # -----------------------------
    # Retrieve ground truth
    # -----------------------------
    #ground truth ellipses
    gt_covMat_CIE = color_thresholds.N_unit_to_W_unit(results3D['fitEllipsoid_scaled'])
    
    class sim_data:
        def __init__(self, xref_all, x1_all):
            self.xref_all = xref_all
            self.x1_all = x1_all
    sim_trial_by_CIE = sim_data(xref_jnp_flat, x1_jnp_flat)
    
    #% derive 2D slices
    # Initialize 3D covariance matrices for ground truth and predictions
    gt_covMat_CIE   = np.full((NUM_GRID_PTS, NUM_GRID_PTS, NUM_GRID_PTS, 3, 3), np.nan)
    
    # Loop through each reference color in the 3D space
    for g1 in range(NUM_GRID_PTS):
        for g2 in range(NUM_GRID_PTS):
            for g3 in range(NUM_GRID_PTS):
                #Convert the ellipsoid parameters to covariance matrices for the 
                #ground truth
                gt_covMat_CIE[g1,g2,g3] = (scaler_x1*2)**2*ellParams_to_covMat(\
                                results3D['ellipsoidParams'][g1,g2,g3]['radii'],\
                                results3D['ellipsoidParams'][g1,g2,g3]['evecs'])
    # Compute the 2D ellipse slices from the 3D covariance matrices for both ground 
    #truth and predictions
    gt_slice_2d_ellipse_CIE = covMat3D_to_2DsurfaceSlice(gt_covMat_CIE)
    # compute the slice of model-estimated cov matrix
    model_pred_slice_2d_ellipse = covMat3D_to_2DsurfaceSlice(model_pred_indvEll.covMat_recover_grid)
    model_pred_indvEll.pred_slice_2d_ellipse =model_pred_slice_2d_ellipse
    
    #%
    # -----------------------------
    # Visualize model predictions
    # -----------------------------
    wishart_pred_vis = WishartPredictionsVisualization(sim_trial_by_CIE,
                                                         model_indvEll, 
                                                         model_pred_indvEll, 
                                                         color_thres_data,
                                                         fig_dir = output_figDir_fits, 
                                                         save_fig = False)
    #specify figure name and path
    fig_name_part1 = 'Fitted' + file_sim[4:-4]
    
    wishart_pred_vis.plot_3D(
        grid_trans, 
        grid_trans[np.array(idx_trim)][:,np.array(idx_trim)][:,:,np.array(idx_trim)],
        gt_covMat_CIE, 
        gt_slice_2d_ellipse_CIE,
        visualize_samples= True,
        visualize_gt = True,
        visualize_model_estimatedCov = False,
        samples_alpha = 0.1,
        samples_s = 1,
        modelpred_ls = '-',
        modelpred_lc = 'g',
        modelpred_lw = 2,
        modelpred_alpha = 0.5,
        gt_lw= 2,
        gt_lc ='r',
        gt_ls = '--',
        fig_name = fig_name_part1+'_indvEll_withSamples.pdf') 
            
    #% save data
    output_file = fig_name_part1 + "_oddity_indvEll.pkl"
    full_path = f"{output_fileDir}{output_file}"
    
    variable_names = ['nSims', 'jitter','data_new', 'x1_raw', 'xref_raw','weibull_params',
                      'sim_trial_by_CIE', 'grid_1d', 'grid','grid_trans','iters', 'objhist',
                      'nReps', 'KEY_list','total_steps','model_indvEll', 'model_pred_indvEll',
                      'gt_covMat_CIE','gt_slice_2d_ellipse_CIE']
    vars_dict = {}
    for i in variable_names: vars_dict[i] = eval(i)
    
    # Write the list of dictionaries to a file using pickle
    with open(full_path, 'wb') as f:
        pickle.dump(vars_dict, f)