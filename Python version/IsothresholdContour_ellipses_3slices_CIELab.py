# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sys
import numpy as np
from dataclasses import replace
import dill as pickled
import os
sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/Python version")
from analysis.simulations_CIELab import SimThresCIELab
from plotting.sim_CIELab_plotting import CIELabVisualization, Plot2DSinglePlaneSettings
sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids")
from analysis.ellipses_tools import fit_2d_isothreshold_contour
from plotting.wishart_plotting import PlotSettingsBase 
base_dir = '/Volumes/T9/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'
output_figDir = os.path.join(base_dir, 'ELPS_analysis','Simulation_FigFiles',
                             'Python_version','CIE')
output_fileDir = os.path.join(base_dir, 'ELPS_analysis','Simulation_DataFiles')
pltSettings_base = PlotSettingsBase(fig_dir=output_figDir, fontsize = 16)

#%% Some constants
# Set the background RGB color values used for normalization (in this case, a neutral gray)
background_RGB = np.array([0.5,0.5,0.5])

# Define the lower and upper bounds for the varying color dimension in the fine grid
lb_RGBvec_fine = 0
ub_RGBvec_fine = 1
# Set the number of grid points for the reference stimuli in the fine grid
nGridPts_ref_fine = 100

# Define the lower and upper bounds for the varying color dimension in the coarse grid
lb_RGBvec = 0.15
ub_RGBvec = 0.85

# Sample a total of 16 chromatic directions uniformly distributed from 0 to 360 degrees
numDirPts = 16

#the raw isothreshold contou is very tiny, we can amplify it by 5 times for the purpose of visualization
nThetaEllipse  = 200
    
def run_one_setting(fixed_RGBvec, nGridPts_ref, color_diff_algorithm):    
    # Initialize the SimThresCIELab class with the path and background RGB values
    sim_thres_CIELab = SimThresCIELab(background_RGB, 
                                      plane_2D_list=['GB plane', 'RB plane', 'RG plane'])

    # Generate 3 slices of 2D planes (GB, RB, RG) in the RGB color space
    # 'plane_points' has the shape (3 slices x 3 RGB values x 100 x 100 grid points)
    plane_points, _,_,_ = sim_thres_CIELab.get_planes(lb_RGBvec_fine, 
                                                      ub_RGBvec_fine,
                                                      num_grid_pts= nGridPts_ref_fine,
                                                      fixed_val = fixed_RGBvec)
    
    # Generate 3 slices of 2D planes (GB, RB, RG) in the RGB color space with the coarse grid
    ref_points, grid_ref, X, Y = sim_thres_CIELab.get_planes(lb_RGBvec, 
                                                             ub_RGBvec,
                                                             num_grid_pts= nGridPts_ref,
                                                             fixed_val = fixed_RGBvec)

    grid_theta_xy = sim_thres_CIELab.set_chromatic_directions(num_dir_pts= numDirPts) 
    
    # Define the threshold value for deltaE (color difference in CIELab space) to be 1
    deltaE_1JND = 5 if color_diff_algorithm == 'CIE1976' else 2.5
    
    #%%make a finer grid for the direction (just for the purpose of visualization)   
    ssize = (sim_thres_CIELab.nPlanes, nGridPts_ref, nGridPts_ref)
    opt_vecLen              = np.full(ssize + (numDirPts,), np.nan)
    fitEllipse_scaled       = np.full(ssize + (2,nThetaEllipse,),np.nan)
    fitEllipse_unscaled     = np.full(ssize + (2,nThetaEllipse,),np.nan)
    rgb_comp_contour_scaled = np.full(ssize + (2, numDirPts, ),  np.nan)
    ellParams               = np.full(ssize + (5,),  np.nan) #5 free parameters for the ellipse
    
    #% for each fixed R / G / B value in the BG / RB / RG plane
    for p in range(sim_thres_CIELab.nPlanes):
        #indices for the varying chromatic directions 
        #GB plane: [1,2]; RB plane: [0,2]; RG plane: [0,1]
        idx_varyingDim = list(range(sim_thres_CIELab.nPlanes))
        idx_varyingDim.remove(p)
        
        #vecDir is a vector that tells us how far we move along a specific direction 
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
                                                                       coloralg = color_diff_algorithm)
                #fit an ellipse
                fit_results = fit_2d_isothreshold_contour(
                        rgb_ref_pij[idx_varyingDim], grid_theta_xy,
                        vecLength = opt_vecLen[p,i,j],
                        nThetaEllipse = nThetaEllipse)
                fitEllipse_scaled[p,i,j],fitEllipse_unscaled[p,i,j],\
                rgb_comp_contour_scaled[p,i,j], ellParams[p,i,j,:] = fit_results
                                                        
    
    #%% PLOTTING AND SAVING DATA
    sim_CIE_vis = CIELabVisualization(sim_thres_CIELab,
                                      settings = pltSettings_base,
                                      save_fig= False)
    plt2D_settings = replace(Plot2DSinglePlaneSettings(), **pltSettings_base.__dict__)
    plt2D_settings = replace(plt2D_settings, 
                             visualize_raw_data = True,
                             ell_lc = [1,1,1],
                             ref_mc = [1,1,1],
                             rgb_background = np.transpose(plane_points,(0,2,3,1)),
                             fig_name = f'Isothreshold_contour_2D{color_diff_algorithm}.pdf')
    
    grid_est = np.stack((X,Y), axis = 2)
    sim_CIE_vis.plot_2D_all_planes(grid_est, 
                                   fitEllipse_scaled, 
                                   settings = plt2D_settings,
                                   rawData= rgb_comp_contour_scaled)
    
    #%%save to pkl
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
    ext_str = f'_grid{nGridPts_ref}_fixedVal{fixed_RGBvec}'
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
# Set the number of grid points for the reference stimuli in the coarse grid
nGridPts_ref_list = [5, 7]
# For RG, RB, and GB planes, fix the B, G, and R values respectively to be 0.5
fixed_RGBvec_array = np.linspace(0.15, 0.85, 5)
    
#define the algorithm for computing color difference
color_diff_algorithm_list = ['CIE1994'] #['CIE1976', 'CIE1994', 'CIE2000']

for c in color_diff_algorithm_list:
    for n in nGridPts_ref_list:
        for f in fixed_RGBvec_array:
            run_one_setting(f, n, c)

