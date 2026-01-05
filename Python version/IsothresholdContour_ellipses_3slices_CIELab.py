# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 00:03:09 2025

@author: fangfang

This script derives 2D isothreshold contours in CIELab space from RGB stimuli.

Overview
--------
The script computes color-discrimination isothreshold contours by converting RGB
stimuli to CIELab space using calibrated monitor and visual system models, then
identifying comparison stimuli that produce a fixed perceptual color difference
(ΔE) from a reference.

General procedure
-----------------
1. For each selected reference stimulus, RGB values are converted to CIELab space.
   This conversion uses:
     - The monitor’s spectral power distribution (SPD),
     - The background RGB (adaptation) color,
     - Stockman–Sharpe 2° cone fundamentals,
     - A calibrated transformation from LMS cone responses to CIEXYZ
       (M_LMS_TO_XYZ).

   The monitor calibration data and transformation matrices were output from
   MATLAB code (`t_WFromPlanarGamut.m`).

2. A set of chromatic directions is defined in each 2D color plane. For each
   direction, we search along that direction to find the RGB value of a comparison
   stimulus that produces a target color difference (ΔE) relative to the
   reference:
     - ΔE = 5 for CIE1976,
     - ΔE = 2.5 for CIE1994 and CIE2000.

3. The resulting discrete isothreshold contour points are fit with an ellipse,
   with the ellipse center constrained to coincide with the reference stimulus.

Scope of the script
-------------------
The script loops over:
  - A grid of reference stimuli,
  - Three 2D color planes (RG, RB, GB) defined by fixing one RGB component,
  - Three CIE color-difference metrics (CIE1976, CIE1994, CIE2000).

The outputs include discrete isothreshold contours, fitted ellipse parameters,
and rendered ellipse curves for visualization and further analysis.

"""

import sys
import numpy as np
from dataclasses import replace
import dill as pickled
import os
from tqdm import tqdm
sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids")
from analysis.ellipses_tools import fit_2d_isothreshold_contour, UnitCircleGenerate
from analysis.simulations_CIELab import SimThresCIELab, strip_trailing_zeros
from plotting.sim_CIELab_plotting import CIELabVisualization, Plot2DSinglePlaneSettings
from plotting.wishart_plotting import PlotSettingsBase 

base_dir = '/Volumes/T9/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'
output_figDir = os.path.join(base_dir, 'ELPS_analysis','Simulation_FigFiles',
                             'Python_version','CIE')
output_fileDir = os.path.join(base_dir, 'ELPS_analysis','Simulation_DataFiles')
pltSettings_base = PlotSettingsBase(fig_dir=output_figDir, fontsize = 16)

#%% Some constants
# Background RGB (linear, in [0, 1]) used as the adaptation/whitepoint for Lab conversion
background_RGB = np.array([0.5, 0.5, 0.5])

# Bounds for the dense grid used only for visualizing the full RGB planes
lb_RGB_fine = 0
ub_RGB_fine = 1
nGridPts_ref_fine = 100  # resolution of the fine visualization grid

# Bounds for the coarse reference grid used for threshold simulations / ellipse fitting
lb_RGB_grid = 0.15  # corresponds to -0.7 in model space
ub_RGB_grid = 0.85  # corresponds to +0.7 in model space

# Number of chromatic directions around each reference (uniformly spaced in angle)
numDirPts = 16

# Number of samples used to render each fitted ellipse contour
nTheta = 200

# Optional scaling applied to the fitted ellipse about the reference
scaler = 1
    
#%%
def run_one_setting(fixed_RGBvec, nGridPts_ref, color_diff_algorithm):    
    # Instantiate simulator for computing isothreshold contours in CIELab space
    # (evaluated on three 2D planes: GB, RB, RG)
    sim_thres_CIELab = SimThresCIELab(background_RGB, 
                                      plane_2D_list=['GB plane', 'RB plane', 'RG plane']
                                      )

    # Build RGB planes for visualization (fine grid over full [0, 1] range)
    # plane_points shape: (nPlanes, 3, nGridPts_ref_fine, nGridPts_ref_fine)
    plane_points, *_ = sim_thres_CIELab.get_planes(lb_RGB_fine, 
                                                   ub_RGB_fine,
                                                   num_grid_pts= nGridPts_ref_fine,
                                                   fixed_val = fixed_RGBvec
                                                   )
    
    # Build coarse reference grid used for threshold computations
    # ref_points shape: (nPlanes, 3, nGridPts_ref, nGridPts_ref)
    ref_points, grid_ref, X, Y = sim_thres_CIELab.get_planes(lb_RGB_grid, 
                                                             ub_RGB_grid,
                                                             num_grid_pts = nGridPts_ref,
                                                             fixed_val = fixed_RGBvec
                                                             )

    # Unit vectors on the 2D plane (2 × numDirPts), evenly spaced around the circle
    grid_theta_xy = UnitCircleGenerate(numDirPts) 
    
    # Target ΔE defining the "threshold" contour (algorithm-dependent scaling)
    # (You can think of these as approximate 1-JND settings per ΔE metric.)
    deltaE_1JND = 5 if color_diff_algorithm == 'CIE1976' else 2.5
    
    # Allocate arrays for outputs (one entry per plane × ref grid point × direction)
    ssize = (sim_thres_CIELab.nPlanes, nGridPts_ref, nGridPts_ref)
    opt_vecLen              = np.full(ssize + (numDirPts,), np.nan)
    fitEllipse_scaled       = np.full(ssize + (2, nTheta,),np.nan)
    fitEllipse_unscaled     = np.full(ssize + (2, nTheta,),np.nan)
    rgb_comp_contour_scaled = np.full(ssize + (2, numDirPts, ),  np.nan)
    ellParams               = np.full(ssize + (5,),  np.nan) #5 free parameters for the ellipse
    
    # Loop over planes and reference locations, then find contour points
    for p in range(sim_thres_CIELab.nPlanes):
        # Select which two RGB dimensions vary in this plane:
        # GB plane: vary [G, B] (fix R), RB plane: vary [R, B] (fix G), RG plane: vary [R, G] (fix B)
        idx_varyingDim = list(range(sim_thres_CIELab.nPlanes))
        idx_varyingDim.remove(p)
        
        # Direction vector in full 3D RGB space; only the two varying dims are used
        vecDir = np.zeros((sim_thres_CIELab.nPlanes))
        
        #for each reference stimulus
        for i in range(nGridPts_ref):
            for j in range(nGridPts_ref):
                #grab the reference stimulus' RGB
                rgb_ref_pij = ref_points[p,:,i,j]
                
                #for each chromatic direction
                for k in range(numDirPts):      
                    #determine the direction we are varying
                    vecDir[idx_varyingDim] = grid_theta_xy[:,k]
                    #fun minimize to search for the magnitude of vector that 
                    #leads to a pre-determined deltaE
                    opt_vecLen[p,i,j,k] = sim_thres_CIELab.find_vecLen(rgb_ref_pij,
                                                                       vecDir,
                                                                       deltaE_1JND,
                                                                       coloralg = color_diff_algorithm
                                                                       )
                # derive the comparison stimuli
                rgb_comp_contour_scaled[p,i,j] = grid_theta_xy  * opt_vecLen[p,i,j][None] + \
                    rgb_ref_pij[idx_varyingDim, None]
                    
                #fit an ellipse
                fitEllipse_scaled[p,i,j],fitEllipse_unscaled[p,i,j], ellParams[p,i,j] =\
                    fit_2d_isothreshold_contour(rgb_ref_pij[idx_varyingDim], 
                                                rgb_comp_contour_scaled[p,i,j],
                                                nTheta = nTheta, 
                                                flag_force_centered_ref = True,
                                                ellipse_scaler = scaler
                                                )
    
    # PLOTTING
    fixedVal_s = strip_trailing_zeros(f'{fixed_RGBvec:.6f}')
    sim_CIE_vis = CIELabVisualization(sim_thres_CIELab,
                                      settings = pltSettings_base,
                                      save_fig= True
                                      )
    
    plt2D_settings = replace(Plot2DSinglePlaneSettings(), **pltSettings_base.__dict__)
    plt2D_settings = replace(plt2D_settings, 
                             visualize_raw_data = True,
                             ell_lc = [1,1,1],
                             ref_mc = [1,1,1],
                             rgb_background = np.transpose(plane_points,(0,2,3,1)),
                             fig_name = f'Isothreshold_contour_{color_diff_algorithm}'+\
                                 f'_fixedVal{fixedVal_s}.pdf')
    
    grid_est = np.stack((X,Y), axis = 2)
    sim_CIE_vis.plot_2D_all_planes(grid_est, 
                                   fitEllipse_scaled, 
                                   settings = plt2D_settings,
                                   rawData= rgb_comp_contour_scaled
                                   )
    
    # SAVING DATA
    file_name = f'Isothreshold_ellipses_3slices_{color_diff_algorithm}.pkl'
    full_path = os.path.join(output_fileDir, file_name)
    
    #save all the stim info
    stim_keys = ['fixed_RGBvec', 'plane_points', 'grid_ref', 'nGridPts_ref', 
                 'ref_points', 'background_RGB','numDirPts', 'grid_theta_xy', 'deltaE_1JND']
    stim = {}
    for i in stim_keys: stim[i] = eval(i)
    
    #save the results
    results_keys = ['opt_vecLen', 'fitEllipse_scaled', 'fitEllipse_unscaled',\
                    'rgb_comp_contour_scaled', 'ellParams']
    results = {}
    for i in results_keys: results[i] = eval(i)
    
    #% check if there is existing file
    ext_str = f'_grid{nGridPts_ref}_fixedVal{fixedVal_s}'
    if os.path.exists(full_path):
        # Load existing pickle file and check whether `nGridPts_ref` matches
        with open(full_path, 'rb') as f:
            existing_dict = pickled.load(f)    
        flag_match_grid_pts = (f'stim{ext_str}' in existing_dict)
        
        if flag_match_grid_pts:
            # If grid points match, ask whether to overwrite the existing file
            flag_overwrite = input(f"The file '{file_name}' already exists. Enter 'y' to overwrite: ")
            #if yes, overwrite
            if flag_overwrite.lower() == 'y':
                # Overwrite the file with new data
                with open(full_path, 'wb') as f:
                    pickled.dump({
                        f'sim_thres_CIELab{ext_str}': sim_thres_CIELab,
                        f'stim{ext_str}': stim,
                        f'results{ext_str}': results
                    }, f)
            else:
                print("File not overwritten.")
                    
        else: #append the data
            # Construct a new dictionary with grid-specific keys
            data_dict_append = {
                f'sim_thres_CIELab{ext_str}': sim_thres_CIELab,
                f'stim{ext_str}': stim,
                f'results{ext_str}': results
            }
    
            # Add new entries to the existing dictionary
            existing_dict.update(data_dict_append)
    
            # Save updated dictionary back to file
            with open(full_path, 'wb') as f:
                pickled.dump(existing_dict, f)
    else:
        data_dict = {
            f'sim_thres_CIELab{ext_str}': sim_thres_CIELab,
            f'stim{ext_str}': stim,
            f'results{ext_str}': results
            }
        
        # If file doesn't exist, create it and save the current data
        with open(full_path, 'wb') as f:
            pickled.dump(data_dict, f)
        
#%%
# Number of reference points along the "fixed RGB" axis (coarse grids)
nGridPts_ref_list = [5, 7]

# Color-difference metrics to evaluate
color_diff_algorithm_list = ["CIE1976", "CIE1994", "CIE2000"]

# Progress bar setup: total runs = (algorithms) × (grid sizes) × (fixed values per grid)
total_runs = 0
for n in nGridPts_ref_list:
    total_runs += len(np.linspace(lb_RGB_grid, ub_RGB_grid, n)) * len(color_diff_algorithm_list)

pbar = tqdm(total=total_runs, desc="Sweeping settings", unit="run")

for c in color_diff_algorithm_list:
    for n in nGridPts_ref_list:
        # Fixed value for the dimension held constant in each 2D plane 
        fixed_RGBvec_array = np.linspace(lb_RGB_grid, ub_RGB_grid, n)

        for f in fixed_RGBvec_array:
            # Run one configuration: (fixed value, grid resolution, ΔE algorithm)
            run_one_setting(f, n, c)
            pbar.update(1)

pbar.close()
