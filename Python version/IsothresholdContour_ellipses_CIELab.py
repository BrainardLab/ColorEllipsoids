# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from scipy.io import loadmat
import sys
import math
import numpy as np
import pickle

sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/"+\
                "FilesFromPsychtoolbox")
sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/"+\
                "Python version")
import simulations_CIELab

#%% LOAD DATA WE NEED
#load data
T_cones_mat    = loadmat('T_cones.mat')
T_cones        = T_cones_mat['T_cones'] #size: (3, 61)

B_monitor_mat  = loadmat('B_monitor.mat')
B_monitor      = B_monitor_mat['B_monitor'] #size: (61, 3)

M_LMSToXYZ_mat = loadmat('M_LMSToXYZ.mat')
M_LMSToXYZ     = M_LMSToXYZ_mat['M_LMSToXYZ'] #size: (3, 3)

#First create a cube and select the RG, the RB and the GB planes
nGridPts       = 100
grid           = np.linspace(0,1,nGridPts)
x_grid, y_grid = np.meshgrid(grid, grid)

#number of selected planes
nPlanes        = 3

#%% DEINE STIMULUS PROPERTIES AND PLOTTING SPECIFICS
#for RG / RB / GB plane, we fix the B / G / R value to be one of the following
fixed_RGBvec = 0.5

#get the grid points for those three planes with one dimension having a specific fixed value
plane_points = simulations_CIELab.get_gridPts(x_grid,y_grid,np.full(3, fixed_RGBvec))

#set a grid for the reference stimulus
#pick 5 x 5 reference points 
grid_ref     = np.arange(0.2, 0.8, 0.15)
nGridPts_ref = len(grid_ref)
x_grid_ref,y_grid_ref = np.meshgrid(grid_ref,grid_ref)

#get the grid points for the reference stimuli of each plane
ref_points     = simulations_CIELab.get_gridPts(x_grid_ref,y_grid_ref,np.full(3,\
                                                fixed_RGBvec))
    
#compute iso-threshold contour
#set the background RGB
background_RGB = fixed_RGBvec * np.ones((nPlanes,1))

#sample total of 16 directions (0 to 360 deg) 
numDirPts     = 16
grid_theta    = np.linspace(0,2*math.pi-math.pi/8,numDirPts)
grid_theta_xy = np.stack((np.cos(grid_theta),np.sin(grid_theta)),axis = 0)

#define threshold as deltaE = 0.5
deltaE_1JND   = 1

#%%make a finer grid for the direction (just for the purpose of visualization)
#the raw isothreshold contou is very tiny, we can amplify it by 5 times for the purpose of visualization
contour_scaler = 5
nThetaEllipse  = 200
colorMatrix    = ref_points
circleIn2D     = simulations_CIELab.UnitCircleGenerate(nThetaEllipse)
subTitles      = ['GB plane', 'RB plane', 'RG plane']    

ref_Lab                 = np.full((nPlanes, nGridPts_ref, nGridPts_ref, 3), np.nan)
opt_vecLen              = np.full((nPlanes, nGridPts_ref, nGridPts_ref, numDirPts), np.nan)
fitEllipse_scaled       = np.full((nPlanes, nGridPts_ref, nGridPts_ref, 2,  nThetaEllipse),np.nan)
fitEllipse_unscaled     = np.full(fitEllipse_scaled.shape, np.nan)
rgb_comp_contour_scaled = np.full((nPlanes, nGridPts_ref, nGridPts_ref, 2, numDirPts),  np.nan)
rgb_comp_contour_cov    = np.full((nPlanes, nGridPts_ref, nGridPts_ref, 2, 2),np.nan)
ellParams               = np.full((nPlanes, nGridPts_ref, nGridPts_ref, 5),  np.nan) #5 free parameters for the ellipse

#%% for each fixed R / G / B value in the BG / RB / RG plane
for p in range(nPlanes):
    #vecDir is a vector that tells us how far we move along a specific direction 
    vecDir = np.zeros((1,nPlanes))
    #indices for the varying chromatic directions 
    #GB plane: [1,2]; RB plane: [0,2]; RG plane: [0,1]
    idx_varyingDim_full = np.arange(0,nPlanes)
    idx_varyingDim = idx_varyingDim_full[idx_varyingDim_full != p]
    
    #for each reference stimulus
    for i in range(nGridPts_ref):
        for j in range(nGridPts_ref):
            #grab the reference stimulus' RGB
            rgb_ref_pij = ref_points[p,:,i,j]
            #convert it to Lab
            Lab_ref_pij,_,_ = simulations_CIELab.convert_rgb_lab(B_monitor,\
                              background_RGB, rgb_ref_pij)
            ref_Lab[p,i,j,:] = Lab_ref_pij
            
            #for each chromatic direction
            for k in range(numDirPts):
                #determine the direction we are varying
                vecDir[0][idx_varyingDim] = grid_theta_xy[:,k]
                
                #fun minimize to search for the magnitude of vector that 
                #leads to a pre-determined deltaE
                opt_vecLen[p,i,j,k] = simulations_CIELab.find_vecLen(background_RGB,\
                                                            rgb_ref_pij, Lab_ref_pij,\
                                                            vecDir,deltaE_1JND)
            
            #fit an ellipse
            fitEllipse_scaled[p,i,j,:,:],fitEllipse_unscaled[p,i,j,:,:],\
                rgb_comp_contour_scaled[p,i,j,:,:],rgb_comp_contour_cov[p,i,j,:,:],\
                ellParams[p,i,j,:] = simulations_CIELab.fit_2d_isothreshold_contour(rgb_ref_pij, [], \
                grid_theta_xy,vecLength = opt_vecLen[p,i,j,:],varyingRGBplan = \
                    idx_varyingDim,nThetaEllipse = nThetaEllipse,\
                    ellipse_scaler = contour_scaler)
                                                    

#%% PLOTTING AND SAVING DATA
simulations_CIELab.plot_2D_isothreshold_contour(x_grid_ref, y_grid_ref,\
                             fitEllipse_scaled, [], visualizeRawData = True,\
                             rgb_contour = rgb_comp_contour_scaled,\
                             EllipsesLine = '-', fontsize =12)    

#%%save to CSV
file_name   = 'Isothreshold_contour_CIELABderived.pkl'
path_output = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                'ELPS_analysis/Simulation_DataFiles/'
full_path   = f"{path_output}{file_name}"

#save all the parameters
param_keys  = ['T_cones', 'B_monitor', 'M_LMSToXYZ', 'nGridPts', 'grid', 'x_grid',\
              'y_grid', 'nPlanes']
param = {}
for i in param_keys: param[i] = eval(i)

#save all the stim info
stim_keys = ['fixed_RGBvec', 'plane_points', 'grid_ref', 'nGridPts_ref', \
             'x_grid_ref', 'y_grid_ref', 'ref_points', 'background_RGB', \
             'numDirPts', 'grid_theta', 'grid_theta_xy', 'deltaE_1JND']
stim = {}
for i in stim_keys: stim[i] = eval(i)

#save all the plotting specifics
plt_specifics_keys = ['contour_scaler', 'nThetaEllipse', 'colorMatrix', \
                      'circleIn2D', 'subTitles']
plt_specifics = {}
for i in plt_specifics_keys: plt_specifics[i] = eval(i)

#save the results
results_keys = ['ref_Lab', 'opt_vecLen', 'fitEllipse_scaled', 'fitEllipse_unscaled',\
                'rgb_comp_contour_scaled', 'rgb_comp_contour_cov', 'ellParams']
results = {}
for i in results_keys: results[i] = eval(i)
    
# Write the list of dictionaries to a file using pickle
with open(full_path, 'wb') as f:
    pickle.dump([param, stim, results, plt_specifics], f)

#Here is what we do if we want to load the data
# with open(full_path, 'rb') as f:
#     # Load the object from the file
#     data_load = pickle.load(f)



