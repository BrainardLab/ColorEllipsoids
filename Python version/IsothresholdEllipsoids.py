#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 21:02:33 2024

@author: fangfang
"""

import sys
from scipy.io import loadmat
import os
import colour
import math
import numpy as np
from scipy.optimize import minimize
from skimage.measure import EllipseModel
import matplotlib.pyplot as plt
import pickle

folder_path = '/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/'+\
    'Python version/efit-python'
sys.path.append(folder_path)
from ellipsoid_fit import ellipsoid_fit

#import functions from the other script
func_path = '/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/Python version/'
os.chdir(func_path)
from IsothresholdContour_RGBcube import convert_rgb_lab, find_vecLen

path_str = "/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/FilesFromPsychtoolbox"
os.chdir(path_str)

#%% LOAD DATA WE NEED
#load data
param = {}
T_cones_mat = loadmat('T_cones.mat')
param['T_cones'] = T_cones_mat['T_cones'] #size: (3, 61)

B_monitor_mat = loadmat('B_monitor.mat')
param['B_monitor'] = B_monitor_mat['B_monitor'] #size: (61, 3)

M_LMSToXYZ_mat = loadmat('M_LMSToXYZ.mat')
param['M_LMSToXYZ'] = M_LMSToXYZ_mat['M_LMSToXYZ'] #size: (3, 3)

#%% FUNCTIONS
def UnitCircleGenerate_3D(nTheta, nPhi):
    theta = np.linspace(0, 2*np.pi, nTheta)
    phi = np.linspace(0, np.pi, nPhi)
    
    THETA, PHI = np.meshgrid(theta, phi)
    xCoords = np.sin(PHI) * np.cos(THETA)
    yCoords = np.sin(PHI) * np.sin(THETA)
    zCoords = np.cos(PHI)
    
    ellipsoids = np.full((nPhi, nTheta,3), np.nan)
    ellipsoids[:,:,0] = xCoords
    ellipsoids[:,:,1] = yCoords
    ellipsoids[:,:,2] = zCoords
    
    return ellipsoids

def PointsOnEllipsoid(radii, center, eigenVectors, unitEllipsoid):
    x_Ellipsoid = unitEllipsoid[:,:,0]
    y_Ellipsoid = unitEllipsoid[:,:,1]
    z_Ellipsoid = unitEllipsoid[:,:,2]
    
    x_stretched = x_Ellipsoid * radii[0]
    y_stretched = y_Ellipsoid * radii[1]
    z_stretched = z_Ellipsoid * radii[2]
    
    xyz = np.vstack(x_stretched, y_stretched, z_stretched)
    
    xyz_rotated = eigenVectors @ xyz
    
    ellipsoid = xyz_rotated + center
    
    return ellipsoid

def fit_3d_isothreshold_ellipsoid(rgb_ref, rgb_comp, grid_xyz, **kwargs):
    ellP = {
        'vecLen':[],
        'nThetaEllipsoid':200,
        'nPhilEllipsoid':100,
        'ellipsoid_scaler':1}
    ellP.update(kwargs)
    
    #generate a unit circle for Ellipsoid fitting
    circleIn3D = UnitCircleGenerate_3D(ellP['nThetaEllipsoid'], \
                                       ellP['nPhiEllipsoid'])
    #compute a unit circle for ellipsoid fitting
    if rgb_comp == []:
        #compute the comparison stimuli if not provided
        rgb_comp_unscaled = np.reshape(rgb_ref,(1,1,3)) +\
            np.tile(ellP['vecLen'],(1,1,3)) *grid_xyz
        rgb_comp_scaled = np.reshape(rgb_ref,(1,1,3)) + \
            np.tile(ellP['vecLen'],(1,1,3)) * ellP['ellipsoid_scaler'] * grid_xyz
    else:
        rgb_comp_unscaled = rgb_comp
        rgb_comp_scaled = rgb_ref + (rgb_comp_unscaled - rgb_ref)*\
            ellP['ellipsoid_scaler']
    
    #compute covariance of the unscaled comparison stimuli
    if len(rgb_comp_scaled.shape) == 3:
        rgb_comp_unscaled_reshape = rgb_comp_unscaled.reshape(-1,3)
    else:
        rgb_comp_unscaled_reshape = rgb_comp_unscaled
    rgb_contour_cov = np.cov(rgb_comp_unscaled_reshape)
    
    #fit an ellipsoid
    ellFits = {}
    ellFits['center'], ellFits['radii'], ellFits['evecs'],\
        ellFits['v'], ellFits['chi2'] = ellipsoid_fit(rgb_comp_unscaled_reshape)
    
    #adjust the fitted
    fitEllipsoid_unscaled = PointsOnEllipsoid(ellFits['radii'], ellFits['center'],\
                                              ellFits['evecs'], ellFits['circleIn3D'])
    #scale the fitted ellipsoid
    fitEllipsoid_scaled = (fitEllipsoid_unscaled - rgb_ref) * \
        ellP['ellipsoid_scaler'] + rgb_ref
        
    return fitEllipsoid_scaled, fitEllipsoid_unscaled, rgb_comp_scaled,\
        rgb_contour_cov, ellFits
    

#%%
def main():
    #%% DEINE STIMULUS PROPERTIES AND PLOTTING SPECIFICS
    stim, results, plt = {},{},{}
    #define a 5 x 5 x 5 grid of RGB values as reference stimuli
    stim['nGridPts_ref'] = 5
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
    stim['numDirPts_z'] = int(np.ceil(stim['numDirPts_xy']/2))
    #Azimuthal angle, 0 to 360 degrees
    stim['grid_theta'] = np.linspace(0,2*math.pi,stim['numDirPts_xy'])
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
    plt['nThetaEllipsoid'] = 200
    plt['nPhiEllipsoid'] = 100
    plt['circleIn3D'] = UnitCircleGenerate_3D(plt['nThetaEllipsoid'], plt['nPhiEllipsoid'])
    
    #initialization
    results['ref_Lab'] = np.full(stim['ref_points'].shape, np.nan)
    results['opt_vecLen'] = np.full((stim['nGridPts_ref'],stim['nGridPts_ref'],\
                                    stim['nGridPts_ref'],stim['numDirPts_z'],\
                                    stim['numDirPts_xy']),np.nan)
    results['fitEllipsoid_scaled'] = np.full((stim['nGridPts_ref'],stim['nGridPts_ref'],\
                                      stim['nGridPts_ref'],plt['nThetaEllipsoid']*\
                                          plt['nPhiEllipsoid'],3),np.nan)
    results['fitEllipsoid_unscaled'] = np.full(results['fitEllipsoid_scaled'].shape,np.nan)
    results['rgb_surface_scaled'] = np.full((stim['nGridPts_ref'],stim['nGridPts_ref'],\
                                    stim['nGridPts_ref'],stim['numDirPts_z'],\
                                    stim['numDirPts_xy'],3),np.nan)
    results['rgb_surface_cov'] = np.full((stim['nGridPts_ref'],stim['nGridPts_ref'],\
                                    stim['nGridPts_ref'],3,3),np.nan)
    results['ellipsoidParams'] = []
    
    #Fitting starts from here
    #for each reference stimulus
    for i in range(stim['nGridPts_ref']):
        for j in range(stim['nGridPts_ref']):
            for k in range(stim['nGridPts_ref']):
                #grab the reference stimulus' RGB
                rgb_ref_ijk = stim['ref_points'][i,j,k,:]
                
                # Convert the reference RGB values to Lab color space.
                ref_Lab_ijk, _, _ = convert_rgb_lab(param['B_monitor'],\
                                                stim['background_RGB'],\
                                                rgb_ref_ijk)
                results['ref_Lab'][i,j,k,:] = ref_Lab_ijk
                
                #for each chromatic direction
                for l in range(stim['numDirPts_z']):
                    for m in range(stim['numDirPts_xy']):
                        #determine the direction we are going
                        vecDir = np.array([stim['grid_x'][l,m],\
                                           stim['grid_y'][l,m],\
                                           stim['grid_z'][l,m]])
                        
                        #run minimize to search for the magnitude of vector that
                        #leads to a pre-determined deltaE
                        results['opt_vecLen'][i,j,k,l,m] = \
                            find_vecLen(stim['background_RGB'],rgb_ref_ijk,\
                                        ref_Lab_ijk, vecDir,stim['deltaE_1JND'])
                        
                #fit an ellipsoid 
                
    
    
    
    
    
    
    
    