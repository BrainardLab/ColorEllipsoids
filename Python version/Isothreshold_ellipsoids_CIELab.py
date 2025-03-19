#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 21:02:33 2024

@author: fangfang
"""

import sys
import os
import math
import numpy as np
import pickle
import dill as pickled

sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/Python version")
from analysis.simulations_CIELab import SimThresCIELab
from plotting.sim_CIELab_plotting import CIELabVisualization
sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids")
from analysis.ellipsoids_tools import UnitCircleGenerate_3D, fit_3d_isothreshold_ellipsoid
                
base_dir = '/Volumes/T9/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'
output_figDir = os.path.join(base_dir,'ELPS_analysis','Simulation_FigFiles', 'Python_version','CIE')
output_fileDir = os.path.join(base_dir, 'ELPS_analysis','Simulation_DataFiles')

#%%import functions from the other script
# Define the path to the directory containing the necessary files for the simulation
path_str = "/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/FilesFromPsychtoolbox/"
# Set the background RGB color values used for normalization (in this case, a neutral gray)
background_RGB = np.array([0.5,0.5,0.5])

# Initialize the SimThresCIELab class with the path and background RGB values
sim_thres_CIELab = SimThresCIELab(path_str, background_RGB)

#define the algorithm for computing color difference
color_diff_algorithm = 'CIE2000' #or 'CIE2000', 'CIE1994', 'CIE1976' (default)
str_append = '' if color_diff_algorithm == 'CIE1976' else '_'+color_diff_algorithm

#%%
file_name = f'Isothreshold_ellipsoid_CIELABderived{str_append}.pkl'
path_str = os.path.join(base_dir, 'ELPS_analysis','Simulation_DataFiles')
full_path = f"{path_str}/{file_name}"
os.chdir(path_str)

#Here is what we do if we want to load the data
try:
    try: #in older version, the saved data are all dictionaries
        with open(full_path, 'rb') as f:
            data_load = pickle.load(f)
    except: 
        # in newer version, some of the saved data are object/classes, 
        #so we have to use dill for loading
        with open(full_path, 'rb') as f:
            data_load = pickled.load(f)        
    stim, results, plt_specifics = data_load[1], data_load[2], data_load[3]
except:
    # DEINE STIMULUS PROPERTIES AND PLOTTING SPECIFICS
    ndims = 3 #color dimensions
    #define a 5 x 5 x 5 grid of RGB values as reference stimuli
    nGridPts_ref = 5
    #define grid points from 0.2 to 0.8 in each dimension
    grid_ref = np.linspace(0.2,0.8, nGridPts_ref);
    
    #generate 3D grid
    x_grid_ref, y_grid_ref, z_grid_ref = \
        np.meshgrid(grid_ref, grid_ref, grid_ref,indexing = 'ij')
    
    #Concatenate grids to form reference points matrix
    ref_points = np.stack((x_grid_ref, y_grid_ref,  z_grid_ref), axis = 3)
    
    #sample total of 16 directions (0 to 360 deg) 
    numDirPts_xy = 16
    #Sample directions along Z (polar), fewer due to spherical geometry
    numDirPts_z = int(np.ceil(numDirPts_xy/2))+1
    #Azimuthal angle, 0 to 360 degrees
    grid_theta = np.linspace(0,2*math.pi - math.pi/8, numDirPts_xy)
    #Polar angle, 0 to 180 degrees
    grid_phi = np.linspace(0, np.pi, numDirPts_z)
    #Create a grid of angles, excluding the redundant final theta
    grid_THETA, grid_PHI = np.meshgrid(grid_theta, grid_phi)
    
    #Calculate Cartesian coordinates for direction vectors on a unit sphere
    grid_x = np.sin(grid_PHI) * np.cos(grid_THETA)
    grid_y = np.sin(grid_PHI) * np.sin(grid_THETA)
    grid_z = np.cos(grid_PHI)
    grid_xyz = np.stack((grid_x, grid_y, grid_z), axis = 2)
    
    #define threshold as deltaE = 0.5
    deltaE_1JND = 1
    
    #the raw isothreshold contour is very tiny, we can amplify it by 5 times
    #for the purpose of visualization
    ellipsoid_scaler = 2.5
    
    #make a finer grid for the direction (just for the purpose of visualization)
    nThetaEllipsoid = 200
    nPhiEllipsoid   = 100
    circleIn3D = UnitCircleGenerate_3D(nThetaEllipsoid, nPhiEllipsoid)
    
    #initialization
    base_shape1 = (nGridPts_ref, nGridPts_ref, nGridPts_ref)
    base_shape2 = (numDirPts_z, numDirPts_xy)
    
    ref_Lab               = np.full(ref_points.shape, np.nan)
    opt_vecLen            = np.full(base_shape1 + base_shape2, np.nan)
    fitEllipsoid_scaled   = np.full(base_shape1 + (ndims, nThetaEllipsoid* nPhiEllipsoid),np.nan)
    fitEllipsoid_unscaled = np.full(base_shape1 + (ndims, nThetaEllipsoid* nPhiEllipsoid),np.nan)
    rgb_surface_scaled    = np.full(base_shape1 + base_shape2 + (ndims,),np.nan)
    rgb_surface_cov       = np.full(base_shape1 + (ndims, ndims),np.nan)
    ellipsoidParams       = np.full(base_shape1,{})
    
    #Fitting starts from here
    #for each reference stimulus
    for i in range(nGridPts_ref):
        print(i)
        for j in range(nGridPts_ref):
            for k in range(nGridPts_ref):
                #grab the reference stimulus' RGB
                rgb_ref_ijk = ref_points[i,j,k]
                        
                #for each chromatic direction
                for l in range(numDirPts_z):
                    for m in range(numDirPts_xy):
                        #determine the direction we are going
                        vecDir = np.array([grid_x[l,m], grid_y[l,m], grid_z[l,m]])
                        
                        #run minimize to search for the magnitude of vector that
                        #leads to a pre-determined deltaE
                        opt_vecLen[i,j,k,l,m] = sim_thres_CIELab.find_vecLen(rgb_ref_ijk,
                                                                             vecDir,
                                                                             deltaE_1JND,
                                                                             coloralg=color_diff_algorithm)
                        
                #fit an ellipsoid 
                fit_results = fit_3d_isothreshold_ellipsoid(rgb_ref_ijk, 
                                                            grid_xyz,
                                                            vecLen = opt_vecLen[i,j,k],
                                                            nThetaEllipsoid=nThetaEllipsoid,
                                                            nPhiEllipsoid = nPhiEllipsoid,
                                                            ellipsoid_scaler = ellipsoid_scaler)
                
                fitEllipsoid_scaled[i,j,k],fitEllipsoid_unscaled[i,j,k],\
                    rgb_surface_scaled[i,j,k], rgb_surface_cov[i,j,k],\
                    ellipsoidParams[i,j,k] = fit_results
                        
    #%%
    #save all the stim info
    stim_keys = ['nGridPts_ref', 'grid_ref', 'x_grid_ref', 'y_grid_ref',
                 'z_grid_ref', 'ref_points', 'background_RGB', 'numDirPts_xy',
                 'numDirPts_z', 'grid_theta', 'grid_phi', 'grid_THETA', 'grid_PHI',
                 'grid_x', 'grid_y', 'grid_z', 'grid_xyz', 'deltaE_1JND']
    stim = {}
    for i in stim_keys: stim[i] = eval(i)
    
    results_keys = ['ellipsoid_scaler', 'ref_Lab', 'opt_vecLen', 
                    'fitEllipsoid_scaled', 'fitEllipsoid_unscaled',
                    'rgb_surface_scaled', 'rgb_surface_cov', 'ellipsoidParams']
    results = {}
    for i in results_keys: results[i] = eval(i)    
    
    plt_specifics_keys = ['nThetaEllipsoid', 'nPhiEllipsoid', 'circleIn3D']
    plt_specifics = {}
    for i in  plt_specifics_keys: plt_specifics[i] = eval(i)     
    
    full_path = f"{output_fileDir}/{file_name}"
        
    # Write the list of dictionaries to a file using pickle
    with open(full_path, 'wb') as f:
        pickled.dump([sim_thres_CIELab, stim, results, plt_specifics], f)
                        
#%%visualize ellipsoids
sim_CIE_vis = CIELabVisualization(sim_thres_CIELab,
                                  fig_dir=output_figDir, 
                                  save_fig= False)
ndims = 3
sim_CIE_vis.plot_3D(np.reshape(stim['ref_points'],(stim['nGridPts_ref']**ndims,-1)), 
                            np.reshape(results['fitEllipsoid_scaled'],(stim['nGridPts_ref']**ndims,ndims,-1)),
                            visualize_thresholdPoints = True,
                            threshold_points = np.reshape(results['rgb_surface_scaled'],\
                                                          (stim['nGridPts_ref']**ndims,\
                                                           stim['numDirPts_z'],stim['numDirPts_xy'],ndims)),
                            color_ref_rgb = np.array([0.2,0.2,0.2]),
                            color_surf = np.array([0.8,0.8,0.8]),
                            color_threshold = [],
                            fig_name =file_name[:-4]+'.pdf')
                                    
    
    
    
    
    
    