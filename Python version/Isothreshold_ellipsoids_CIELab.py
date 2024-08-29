#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 21:02:33 2024

@author: fangfang
"""

import sys
from scipy.io import loadmat
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import pickle

folder_path = '/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/'+\
    'Python version/efit-python'
sys.path.append(folder_path)
from ellipsoid_fit import ellipsoid_fit
from plotting.sim_CIELab_plotting import CIELabVisualization
sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids")
from analysis.simulations_CIELab import UnitCircleGenerate_3D, fit_3d_isothreshold_ellipsoid


#import functions from the other script
func_path = '/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/Python version/'
os.chdir(func_path)
from simulations_CIELab import convert_rgb_lab, find_vecLen


numRef = input('How many reference stimuli per color dimension (default: 5): ') 
# first see if the script has been run before and we already have the data saved
if numRef == '5' or '':
    file_name = 'Isothreshold_ellipsoid_CIELABderived.pkl'
else:
    str_ext = '_numRef'+numRef
    file_name = 'Isothreshold_ellipsoid_CIELABderived'+str_ext+'.pkl'
numRef_int = int(numRef)
path_str = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
        'ELPS_analysis/Simulation_DataFiles/'
full_path = f"{path_str}{file_name}"
os.chdir(path_str)

#Here is what we do if we want to load the data
try:
    with open(full_path, 'rb') as f:
        # Load the object from the file
        data_load = pickle.load(f)
    param, stim, results, plt_specifics = data_load[0], data_load[1], \
        data_load[2], data_load[3]
except:
    #%% LOAD DATA WE NEED
    path_str = "/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/"+\
        "FilesFromPsychtoolbox"
    os.chdir(path_str)
    #load data
    param = {}
    T_cones_mat = loadmat('T_cones.mat')
    param['T_cones'] = T_cones_mat['T_cones'] #size: (3, 61)

    B_monitor_mat = loadmat('B_monitor.mat')
    param['B_monitor'] = B_monitor_mat['B_monitor'] #size: (61, 3)

    M_LMSToXYZ_mat = loadmat('M_LMSToXYZ.mat')
    param['M_LMSToXYZ'] = M_LMSToXYZ_mat['M_LMSToXYZ'] #size: (3, 3)

    #%% DEINE STIMULUS PROPERTIES AND PLOTTING SPECIFICS
    stim, results, plt_specifics = {},{},{}
    #define a 5 x 5 x 5 grid of RGB values as reference stimuli
    stim['nGridPts_ref'] = numRef_int
    #define grid points from 0.2 to 0.8 in each dimension
    stim['grid_ref'] = np.linspace(0.2,0.8,stim['nGridPts_ref']);
    
    #generate 3D grid
    stim['x_grid_ref'], stim['y_grid_ref'], stim['z_grid_ref'] = \
        np.meshgrid(stim['grid_ref'],stim['grid_ref'],stim['grid_ref'],indexing = 'ij')
    
    #Concatenate grids to form reference points matrix
    stim['ref_points'] = np.stack((stim['x_grid_ref'], stim['y_grid_ref'],\
                                         stim['z_grid_ref']), axis = 3)
        
    #Define a neutral background RGB value for the simulations
    stim['background_RGB'] = np.ones((3,1))*0.5
    
    #sample total of 16 directions (0 to 360 deg) 
    stim['numDirPts_xy'] = 16
    #Sample directions along Z (polar), fewer due to spherical geometry
    stim['numDirPts_z'] = int(np.ceil(stim['numDirPts_xy']/2))+1
    #Azimuthal angle, 0 to 360 degrees
    stim['grid_theta'] = np.linspace(0,2*math.pi - math.pi/8,stim['numDirPts_xy'])
    #Polar angle, 0 to 180 degrees
    stim['grid_phi'] = np.linspace(0, np.pi, stim['numDirPts_z'])
    #Create a grid of angles, excluding the redundant final theta
    stim['grid_THETA'], stim['grid_PHI'] = np.meshgrid(stim['grid_theta'], stim['grid_phi'])
    
    #Calculate Cartesian coordinates for direction vectors on a unit sphere
    stim['grid_x'] = np.sin(stim['grid_PHI']) * np.cos(stim['grid_THETA'])
    stim['grid_y'] = np.sin(stim['grid_PHI']) * np.sin(stim['grid_THETA'])
    stim['grid_z'] = np.cos(stim['grid_PHI'])
    stim['grid_xyz'] = np.stack((stim['grid_x'], stim['grid_y'], stim['grid_z']), axis = 2)
    
    #define threshold as deltaE = 0.5
    stim['deltaE_1JND'] = 1
    
    #the raw isothreshold contour is very tiny, we can amplify it by 5 times
    #for the purpose of visualization
    results['ellipsoid_scaler'] = 5
    
    #make a finer grid for the direction (just for the purpose of visualization)
    plt_specifics['nThetaEllipsoid'] = 200
    plt_specifics['nPhiEllipsoid'] = 100
    plt_specifics['circleIn3D'] = UnitCircleGenerate_3D(plt_specifics['nThetaEllipsoid'],\
                                                        plt_specifics['nPhiEllipsoid'])
    
    #initialization
    results['ref_Lab'] = np.full(stim['ref_points'].shape, np.nan)
    results['opt_vecLen'] = np.full((stim['nGridPts_ref'],stim['nGridPts_ref'],\
                                    stim['nGridPts_ref'],stim['numDirPts_z'],\
                                    stim['numDirPts_xy']),np.nan)
    results['fitEllipsoid_scaled'] = np.full((stim['nGridPts_ref'],stim['nGridPts_ref'],\
                                      stim['nGridPts_ref'],3,plt_specifics['nThetaEllipsoid']*\
                                          plt_specifics['nPhiEllipsoid']),np.nan)
    results['fitEllipsoid_unscaled'] = np.full(results['fitEllipsoid_scaled'].shape,np.nan)
    results['rgb_surface_scaled'] = np.full((stim['nGridPts_ref'],stim['nGridPts_ref'],\
                                    stim['nGridPts_ref'],stim['numDirPts_z'],\
                                    stim['numDirPts_xy'],3),np.nan)
    results['rgb_surface_cov'] = np.full((stim['nGridPts_ref'],stim['nGridPts_ref'],\
                                    stim['nGridPts_ref'],3,3),np.nan)
    results['ellipsoidParams'] = np.full((stim['nGridPts_ref'],stim['nGridPts_ref'],\
                                          stim['nGridPts_ref']),{})
    
    #%%Fitting starts from here
    #for each reference stimulus
    for i in range(stim['nGridPts_ref']):
        print(i)
        for j in range(stim['nGridPts_ref']):
            for k in range(stim['nGridPts_ref']):
                #grab the reference stimulus' RGB
                rgb_ref_ijk = stim['ref_points'][i,j,k]
                
                # Convert the reference RGB values to Lab color space.
                ref_Lab_ijk, _, _ = convert_rgb_lab(param['B_monitor'],\
                                                stim['background_RGB'],\
                                                rgb_ref_ijk)
                results['ref_Lab'][i,j,k] = ref_Lab_ijk
                
                #for each chromatic direction
                for l in range(stim['numDirPts_z']):
                    for m in range(stim['numDirPts_xy']):
                        #determine the direction we are going
                        vecDir = np.array([[stim['grid_x'][l,m],\
                                           stim['grid_y'][l,m],\
                                           stim['grid_z'][l,m]]])
                        
                        #run minimize to search for the magnitude of vector that
                        #leads to a pre-determined deltaE
                        results['opt_vecLen'][i,j,k,l,m] = \
                            find_vecLen(stim['background_RGB'],rgb_ref_ijk,\
                                        ref_Lab_ijk, vecDir,stim['deltaE_1JND'])
                        
                #fit an ellipsoid 
                results['fitEllipsoid_scaled'][i,j,k],\
                    results['fitEllipsoid_unscaled'][i,j,k],\
                    results['rgb_surface_scaled'][i,j,k],\
                    results['rgb_surface_cov'][i,j,k],\
                    results['ellipsoidParams'][i,j,k] = \
                    fit_3d_isothreshold_ellipsoid(rgb_ref_ijk, [], stim['grid_xyz'],\
                        vecLen = results['opt_vecLen'][i,j,k],\
                        nThetaEllipsoid=plt_specifics['nThetaEllipsoid'],\
                        nPhiEllipsoid = plt_specifics['nPhiEllipsoid'],\
                        ellipsoid_scaler = results['ellipsoid_scaler'])
    #save the data
    path_output = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
        'ELPS_analysis/Simulation_DataFiles/'
    full_path = f"{path_output}{file_name}"
        
    # Write the list of dictionaries to a file using pickle
    with open(full_path, 'wb') as f:
        pickle.dump([param, stim, results, plt_specifics], f)
                        
#%%visualize ellipsoids
CIELabVisualization.plot_3D_isothreshold_ellipsoid(stim['grid_ref'], stim['grid_ref'],
                               stim['grid_ref'], results['fitEllipsoid_scaled'],
                               plt_specifics['nThetaEllipsoid'], 
                               plt_specifics['nPhiEllipsoid'],
                               slc_x_grid_ref = np.arange(0,5,2),
                               slc_y_grid_ref = np.arange(0,5,2),
                               slc_z_grid_ref = np.arange(0,5,2),
                               visualize_thresholdPoints = True,
                               threshold_points = results['rgb_surface_scaled'],
                               color_ref_rgb = np.array([0.2,0.2,0.2]),
                               color_surf = np.array([0.8,0.8,0.8]),
                               color_threshold = [])
                                    
    
    
    
    
    
    