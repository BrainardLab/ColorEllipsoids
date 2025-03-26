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

base_dir = '/Volumes/T9/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'
output_figDir = base_dir+ 'ELPS_analysis/Simulation_FigFiles/Python_version/CIE/'
output_fileDir = base_dir+ 'ELPS_analysis/Simulation_DataFiles/'

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
deltaE_1JND   = 2.5

#define the algorithm for computing color difference
color_diff_algorithm = 'CIE2000' #or 'CIE2000', 'CIE1994', 'CIE1976' (default)
str_append = '' if color_diff_algorithm == 'CIE1976' else '_'+color_diff_algorithm

#%%make a finer grid for the direction (just for the purpose of visualization)
#the raw isothreshold contou is very tiny, we can amplify it by 5 times for the purpose of visualization
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
                                  fig_dir=output_figDir, 
                                  save_fig= False)

grid_est = np.stack((X,Y), axis = 2)
sim_CIE_vis.plot_2D_all_planes(grid_est, 
                    fitEllipse_scaled, 
                    visualize_raw_data = True,
                    rawData= rgb_comp_contour_scaled, 
                    ell_lc = [1,1,1],
                    ref_mc = [1,1,1],
                    rgb_background = np.transpose(plane_points,(0,2,3,1)),
                    fig_name = f'Isothreshold_contour_2D{str_append}.pdf')

#%%save to CSV
# file_name   = f'Isothreshold_contour_CIELABderived_fixedVal{fixed_RGBvec}{str_append}.pkl'
# full_path   = f"{output_fileDir}{file_name}"

# #save all the stim info
# stim_keys = ['fixed_RGBvec', 'plane_points', 'grid_ref', 'nGridPts_ref', 
#              'ref_points', 'background_RGB','numDirPts', 'grid_theta_xy', 'deltaE_1JND']
# stim = {}
# for i in stim_keys: stim[i] = eval(i)

# #save the results
# results_keys = ['opt_vecLen', 'fitEllipse_scaled', 'fitEllipse_unscaled',\
#                 'rgb_comp_contour_scaled', 'rgb_comp_contour_cov', 'ellParams']
# results = {}
# for i in results_keys: results[i] = eval(i)
    
# # Write the list of dictionaries to a file using pickle
# with open(full_path, 'wb') as f:
#     pickle.dump([sim_thres_CIELab, stim, results], f)



