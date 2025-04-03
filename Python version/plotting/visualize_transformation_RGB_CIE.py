#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 12:20:12 2025

@author: fangfang
"""
import sys
import numpy as np
import pickle
sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/"+\
                "Python version")
from analysis.simulations_CIELab import SimThresCIELab
from plotting.sim_CIELab_plotting import CIELabVisualization
sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids")
from analysis.ellipses_tools import fit_2d_isothreshold_contour

base_dir = '/Volumes/T9/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'
output_figDir = base_dir+ 'ELPS_analysis/Simulation_FigFiles/Python_version/CIE/'
input_fileDir = base_dir+ 'ELPS_analysis/Simulation_DataFiles/'

#Here is what we do if we want to load the data
file_name   = 'Isothreshold_contour_CIELABderived_fixedVal0.5_CIE2000.pkl'
full_path   = f"{input_fileDir}{file_name}"
with open(full_path, 'rb') as f:
    # Load the object from the file
    data_load = pickle.load(f)

#%% visualize RGB color space and CIELab color space
sim_CIE_vis = CIELabVisualization(sim_thres_CIELab,
                                  fig_dir=output_figDir, 
                                  save_fig= False)

viewing_angle = [[30,-25],[30,-15], [30,-75]]
lab_comp = np.full(plane_points.shape,np.nan)
for p in range(sim_thres_CIELab.nPlanes):
    #for each chromatic direction
    for i in range(nGridPts_ref_fine):
        for j in range(nGridPts_ref_fine):
            rgb_comp_slc_ij = plane_points[p,:,i,j]
            lab_comp[p,:,i,j],_,_ = sim_thres_CIELab.convert_rgb_lab(rgb_comp_slc_ij)
    sim_CIE_vis.plot_RGB_to_LAB(plane_points[p], 
                                lab_comp[p], 
                                lab_viewing_angle = viewing_angle[p],
                                fig_name = f'RGB_to_CIELab2000_conversion{p}.pdf')

#%% visualize primaries of the monitor
rgb_s_green = np.array([0.5, 0.8, 0.2])
rgb_instances = np.stack((sim_thres_CIELab.background_rgb,rgb_s_green), 
                         axis = 1)
sim_CIE_vis.plot_primaries(rgb = rgb_instances, 
                           figsize = (3,3),
                           visualize_primaries = False,
                           fig_name = 'spd_background.pdf')

#compute the CIELab with provided background and stimulus rgb values
color_CIE_background, color_XYZ_background, color_LMS_background = \
    sim_thres_CIELab.convert_rgb_lab(sim_thres_CIELab.background_rgb)

color_CIE_eg, color_XYZ_eg, color_LMS_eg = \
    sim_thres_CIELab.convert_rgb_lab(rgb_s_green)

print(color_CIE_background)
print(color_CIE_eg)

#%% visualize color patches at the threshold
# Define an RGB color array for blue with shape (3, 1). This is just an example
rgb_s_blue = np.array([[0.65],[0.8],[0.8]])

# Extract the scaled threshold value for blue from the rgb_comp_contour_scaled array
rgb_s_blue_thres_scaled = rgb_comp_contour_scaled[0,-1,-1]

# Unscale the threshold value by applying the contour_scaler and adjusting with 
#the original RGB values
#rgb_s_blue_thres_unscaled = (rgb_s_blue_thres_scaled - rgb_s_blue[1:])/contour_scaler +\
#    rgb_s_blue[1:]
    
# Combine the unscaled threshold value with the fixed R value (first row) and 
# stack to form a (3, numDirPts) array
rgb_s_blue_thres = np.vstack((np.full((1, numDirPts), rgb_s_blue[0]), 
                              rgb_s_blue_thres_scaled))

# Visualize the stimuli at threshold using the provided visualization method
sim_CIE_vis.visualize_stimuli_at_thres(rgb_s_blue_thres,
                                       save_fig = False,
                                       fig_dir = output_figDir,
                                       fig_name = f'color_patches_thres_r{rgb_s_blue[0]}'+\
                                           f'_g{rgb_s_blue[1]}_b{rgb_s_blue[2]}.pdf')

# Define the upper bounds for the blue color using grid_theta_xy, scaled by 0.2
rgb_s_blue_ub = np.vstack((np.full((1, numDirPts), 0), 
                        grid_theta_xy*0.2)) + rgb_s_blue
# Visualize the stimuli at the upper bounds using the provided visualization method
sim_CIE_vis.visualize_stimuli_at_thres(rgb_s_blue_ub,
                                       save_fig = False,
                                       fig_dir = output_figDir,
                                       fig_name = f'color_patches_ub_r{rgb_s_blue[0]}'+\
                                           f'_g{rgb_s_blue[1]}_b{rgb_s_blue[2]}.pdf')

#%%
slc_dir = grid_theta_xy[:,2] #2 or 14
# Tile the array to have a shape of (2, 20)
num_pts_path = 10
path_points_blue_tile = np.tile(slc_dir[:, np.newaxis], (1, num_pts_path))*\
    np.linspace(0,0.2,num_pts_path).reshape(1,num_pts_path)
path_points_blue = np.vstack((np.full((1, num_pts_path), 0), path_points_blue_tile)) + rgb_s_blue
comp_CIE_blue = np.full(path_points_blue.shape, np.nan)
deltaE_blue = np.full((num_pts_path),np.nan)
for idx in range(num_pts_path):
    comp_CIE_blue[:,idx], _, _ = sim_thres_CIELab.convert_rgb_lab(path_points_blue[:,idx])
    deltaE_blue[idx] = sim_thres_CIELab.compute_deltaE(path_points_blue[:,0],[],[],
                                                       comp_RGB = path_points_blue[:,idx])
    
sim_CIE_vis.plot_RGB_to_LAB(path_points_blue[:,:,np.newaxis], 
                            comp_CIE_blue[:,:,np.newaxis], 
                            lab_viewing_angle = viewing_angle[0],
                            lab_xylim = [-15,0],
                            lab_zlim = [118,130],
                            lab_scatter_ms = 50,
                            lab_ticks = [-20,-10,0],
                            lab_scatter_alpha = 1,
                            lab_scatter_edgecolor = 'k',
                            fontsize = 15,
                            fig_name = f'RGB_to_CIELab_conversion{p}_1path_cDir'+\
                                f'_{slc_dir[0]:.2f}_{slc_dir[1]:.2f}.pdf')
    
# sim_CIE_vis.plot_deltaE(deltaE_blue, 
#                         np.transpose(path_points_blue,(1,0)),
#                         save_fig = True,
#                         fig_dir = output_figDir,
#                         fig_name = 'deltaE_1path_cDir'+\
#                             f'_{slc_dir[0]:.2f}_{slc_dir[1]:.2f}.pdf')