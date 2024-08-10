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

output_fileDir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                'ELPS_analysis/Simulation_DataFiles/'

#%% LOAD DATA WE NEED
path_str = "/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/"+\
                "FilesFromPsychtoolbox/"
#set the background RGB
background_RGB = np.array([0.5,0.5,0.5])
sim_thres_CIELab = SimThresCIELab(path_str, background_RGB)
#for RG / RB / GB plane, we fix the B / G / R value to be one of the following
fixed_RGBvec = 0.5
#First create a cube and select the RG, the RB and the GB planes
nGridPts_ref_fine = 100
plane_points, _,_,_ = sim_thres_CIELab.get_planes(0, 1,
                                           num_grid_pts= nGridPts_ref_fine,
                                           fixed_val = fixed_RGBvec)
#get the grid points for the reference stimuli of each plane
nGridPts_ref= 5
ref_points, grid_ref, X, Y = sim_thres_CIELab.get_planes(0.2, 0.8,
                                                         num_grid_pts= nGridPts_ref,
                                                         fixed_val = fixed_RGBvec)

#sample total of 16 directions (0 to 360 deg) 
numDirPts = 16
grid_theta_xy = sim_thres_CIELab.set_chromatic_directions(num_dir_pts= numDirPts) 

#define threshold as deltaE = 1
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
                                  fig_dir='', 
                                  save_fig=False, 
                                  save_gif=False)

grid_est = np.stack((X,Y), axis = 2)
sim_CIE_vis.plot_2D(grid_est, fitEllipse_scaled, visualize_raw_data = True, 
                    rawData= rgb_comp_contour_scaled, ell_lc = [1,1,1],
                    ref_mc = [1,1,1],
                    rgb_background = np.transpose(plane_points,(0,2,3,1)))

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



