# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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

output_figDir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                'ELPS_analysis/Simulation_FigFiles/Python_version/CIE/'
output_fileDir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                'ELPS_analysis/Simulation_DataFiles/'

#%% LOAD DATA WE NEED
# Define the path to the directory containing the necessary files for the simulation
path_str = "/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/"+\
                "FilesFromPsychtoolbox/"
# Set the background RGB color values used for normalization (in this case, a neutral gray)
background_RGB = np.array([0.5,0.5,0.5])

# Initialize the SimThresCIELab class with the path and background RGB values
sim_thres_CIELab = SimThresCIELab(path_str, background_RGB)
# For RG, RB, and GB planes, fix the B, G, and R values respectively to be 0.5
fixed_RGBvec = 0.5
# Define the lower and upper bounds for the varying color dimension in the fine grid
lb_RGBvec_fine = 0
ub_RGBvec_fine = 1
# Set the number of grid points for the reference stimuli in the fine grid
nGridPts_ref_fine = 100
# Generate 3 slices of 2D planes (GB, RB, RG) in the RGB color space
# 'plane_points' has the shape (3 slices x 3 RGB values x 100 x 100 grid points)
plane_points, _,_,_ = sim_thres_CIELab.get_planes(lb_RGBvec_fine, 
                                                  ub_RGBvec_fine,
                                                  num_grid_pts= nGridPts_ref_fine,
                                                  fixed_val = fixed_RGBvec)

# Define the lower and upper bounds for the varying color dimension in the coarse grid
lb_RGBvec = 0.2
ub_RGBvec = 0.8
# Set the number of grid points for the reference stimuli in the coarse grid
nGridPts_ref= 5
# Generate 3 slices of 2D planes (GB, RB, RG) in the RGB color space with the coarse grid
ref_points, grid_ref, X, Y = sim_thres_CIELab.get_planes(lb_RGBvec, 
                                                         ub_RGBvec,
                                                         num_grid_pts= nGridPts_ref,
                                                         fixed_val = fixed_RGBvec)

# Sample a total of 16 chromatic directions uniformly distributed from 0 to 360 degrees
numDirPts = 16
grid_theta_xy = sim_thres_CIELab.set_chromatic_directions(num_dir_pts= numDirPts) 

# Define the threshold value for deltaE (color difference in CIELab space) to be 1
deltaE_1JND   = 1

#%%make a finer grid for the direction (just for the purpose of visualization)
#the raw isothreshold contou is very tiny, we can amplify it by 5 times for the purpose of visualization
contour_scaler = 5
nThetaEllipse  = 200

ssize = (sim_thres_CIELab.nPlanes, nGridPts_ref, nGridPts_ref)
opt_vecLen              = np.full(ssize + (numDirPts,), np.nan)
fitEllipse_scaled       = np.full(ssize + (2,nThetaEllipse,),np.nan)
fitEllipse_unscaled     = np.full(ssize + (2,nThetaEllipse,),np.nan)
rgb_comp_contour_scaled = np.full(ssize + (2, numDirPts, ),  np.nan)
rgb_comp_contour_cov    = np.full(ssize + (2, 2,),np.nan)
ellParams               = np.full(ssize + (5,),  np.nan) #5 free parameters for the ellipse

#%% for each fixed R / G / B value in the BG / RB / RG plane
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
                                                                   deltaE_1JND)
            #fit an ellipse
            fit_results = fit_2d_isothreshold_contour(
                    rgb_ref_pij, grid_theta_xy,
                    vecLength = opt_vecLen[p,i,j],
                    varyingRGBplan = idx_varyingDim,
                    nThetaEllipse = nThetaEllipse,
                    ellipse_scaler = contour_scaler)
            fitEllipse_scaled[p,i,j],fitEllipse_unscaled[p,i,j],\
            rgb_comp_contour_scaled[p,i,j],rgb_comp_contour_cov[p,i,j],\
            ellParams[p,i,j,:] = fit_results
                                                    

#%% PLOTTING AND SAVING DATA
sim_CIE_vis = CIELabVisualization(sim_thres_CIELab,
                                  fig_dir=output_figDir, 
                                  save_fig= True)

grid_est = np.stack((X,Y), axis = 2)
sim_CIE_vis.plot_2D(grid_est, 
                    fitEllipse_scaled, 
                    visualize_raw_data = True,
                    rawData= rgb_comp_contour_scaled, 
                    ell_lc = [1,1,1],
                    ref_mc = [1,1,1],
                    rgb_background = np.transpose(plane_points,(0,2,3,1)),
                    fig_name = 'Isothreshold_contour_2D.pdf')

#%% visualize RGB color space and CIELab color space
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
                                fig_name = f'RGB_to_CIELab_conversion{p}.pdf')

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
rgb_s_blue = np.array([[0.5],[0.8],[0.8]])

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
                                       save_fig = True,
                                       fig_dir = output_figDir,
                                       fig_name = f'color_patches_thres_r{rgb_s_blue[0]}'+\
                                           f'_g{rgb_s_blue[1]}_b{rgb_s_blue[2]}.pdf')

# Define the upper bounds for the blue color using grid_theta_xy, scaled by 0.2
rgb_s_blue_ub = np.vstack((np.full((1, numDirPts), 0), 
                        grid_theta_xy*0.2)) + rgb_s_blue
# Visualize the stimuli at the upper bounds using the provided visualization method
sim_CIE_vis.visualize_stimuli_at_thres(rgb_s_blue_ub,
                                       save_fig = True,
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

#%%save to CSV
file_name   = f'Isothreshold_contour_CIELABderived_fixedVal{fixed_RGBvec}.pkl'
full_path   = f"{output_fileDir}{file_name}"

#save all the stim info
stim_keys = ['fixed_RGBvec', 'plane_points', 'grid_ref', 'nGridPts_ref', \
             'x_grid_ref', 'y_grid_ref', 'ref_points', 'background_RGB', \
             'numDirPts', 'grid_theta', 'grid_theta_xy', 'deltaE_1JND']
stim = {}
for i in stim_keys: stim[i] = eval(i)

#save the results
results_keys = ['opt_vecLen', 'fitEllipse_scaled', 'fitEllipse_unscaled',\
                'rgb_comp_contour_scaled', 'rgb_comp_contour_cov', 'ellParams']
results = {}
for i in results_keys: results[i] = eval(i)
    
# Write the list of dictionaries to a file using pickle
with open(full_path, 'wb') as f:
    pickle.dump([sim_thres_CIELab, stim, results], f)

#Here is what we do if we want to load the data
# with open(full_path, 'rb') as f:
#     # Load the object from the file
#     data_load = pickle.load(f)



