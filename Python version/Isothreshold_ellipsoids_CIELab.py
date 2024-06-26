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

#import functions from the other script
func_path = '/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/Python version/'
os.chdir(func_path)
from simulations_CIELab import convert_rgb_lab, find_vecLen

#%% FUNCTIONS
def UnitCircleGenerate_3D(nTheta, nPhi):
    """
    Generates points on the surface of a unit sphere (3D ellipsoid with equal radii) 
    by sampling angles theta and phi in spherical coordinates.
    
    Parameters:
    - nTheta (int): The number of points to sample along the theta dimension.
    - nPhi (int): The number of points to sample along the phi dimension.
            Determines the resolution from top (north pole) to bottom (south pole).
            
    Returns:
    - ellipsoids: A 3D numpy array of shape (nPhi, nTheta, 3), where each "slice" 
        of the array ([..., 0], [..., 1], [..., 2]) corresponds to the x, y, and z 
        coordinates of points on the unit sphere. The first two dimensions 
        correspond to the grid defined by the phi and theta angles, and the 
        third dimension corresponds to the Cartesian coordinates.
    """
    # Generate linearly spaced angles for theta and phi
    theta = np.linspace(0, 2*np.pi, nTheta)
    phi = np.linspace(0, np.pi, nPhi)
    
    # Create 2D grids for theta and phi using meshgrid.
    # THETA and PHI arrays have shapes (nPhi, nTheta) and contain all combinations
    # of phi and theta values.
    THETA, PHI = np.meshgrid(theta, phi)
    
    # Calculate the Cartesian coordinates for points on the unit sphere surface
    # using the spherical to Cartesian coordinate transformation.
    xCoords = np.sin(PHI) * np.cos(THETA)
    yCoords = np.sin(PHI) * np.sin(THETA)
    zCoords = np.cos(PHI)
    
    # Initialize an array to hold the Cartesian coordinates of the points on 
    # the unit sphere. The array is initially filled with NaNs and has the shape 
    #(nPhi, nTheta, 3).
    ellipsoids = np.full((nPhi, nTheta,3), np.nan)
    
    # Assign the calculated Cartesian coordinates to the corresponding "slices" of the
    # ellipsoids array.
    ellipsoids[:,:,0] = xCoords
    ellipsoids[:,:,1] = yCoords
    ellipsoids[:,:,2] = zCoords
    
    return ellipsoids

def PointsOnEllipsoid(radii, center, eigenVectors, unitEllipsoid):
    """
    This function computes points on the surface of an ellipsoid given its 
    radii, center, orientation (via eigenVectors), and a unit ellipsoid 
    (unit sphere mapped to an ellipsoid).
    
    Parameters:
    - radii (array; (3,)): Radii of the ellipsoid along the x, y, and z axes.
    - center (array; (3,1)): Center of the ellipsoid in 3D space.
    - eigenVectors (array; (3,3)): Rotation matrix representing the orientation 
        of the ellipsoid.
    - unitEllipsoid (array; (nPhi, nTheta,3)): Points on a unit ellipsoid, 
        which is a unit sphere scaled according to the ellipsoid's radii.
                     
    Returns:
    - ellipsoid: A 2D array of size (3, N) containing the 3D coordinates of 
        N points on the ellipsoid's surface. The first dimension corresponds 
        to the x, y, z coordinates, and the second dimension corresponds to 
        the sampled grid points.

    """
    # Extract the x, y, and z coordinates from the unit ellipsoid's surface points.
    x_Ellipsoid = unitEllipsoid[:,:,0]
    y_Ellipsoid = unitEllipsoid[:,:,1]
    z_Ellipsoid = unitEllipsoid[:,:,2]
    
    # Stretch the unit ellipsoid points by the ellipsoid radii to get the 
    # ellipsoid's shape in its principal axis frame.
    x_stretched = x_Ellipsoid * radii[0]
    y_stretched = y_Ellipsoid * radii[1]
    z_stretched = z_Ellipsoid * radii[2]
    
    # Stack the stretched coordinates and flatten them to create a 2D array of size (3, N),
    # where N = Theta * Phi. This step prepares the coordinates for rotation.
    xyz = np.vstack((x_stretched.flatten(), y_stretched.flatten(), z_stretched.flatten()))
    
    # Rotate the stretched ellipsoid points to align with the ellipsoid's actual 
    # orientation in 3D space using the eigenVectors rotation matrix.
    # The resulting xyz_rotated array has size (3, N).
    xyz_rotated = eigenVectors @ xyz
    
    # Translate the rotated points by the ellipsoid's center to position the ellipsoid
    # correctly in 3D space. The size of the ellipsoid array remains (3, N).
    ellipsoid = xyz_rotated + center
    
    return ellipsoid

def fit_3d_isothreshold_ellipsoid(rgb_ref, rgb_comp, grid_xyz, **kwargs):
    """
    Fits a 3D ellipsoid to a set of RGB color stimuli and adjusts the fit based
    on specified parameters and scaling factors.
    
    Parameters:
    - rgb_ref: The reference RGB stimulus.
    - rgb_comp: The comparison RGB stimuli to which the ellipsoid is fitted. 
        If empty, it will be computed based on the reference stimulus, vector 
        length, and grid_xyz.
    - grid_xyz: A grid of XYZ coordinates representing the direction and 
        magnitude of comparison stimuli from the reference stimulus.
    - kwargs: Additional keyword arguments to specify or override ellipsoid 
        fitting parameters.
    
    Returns:
    - fitEllipsoid_scaled: The scaled fitted ellipsoid coordinates in RGB space.
    - fitEllipsoid_unscaled: The unscaled fitted ellipsoid coordinates in RGB space.
    - rgb_comp_scaled: The scaled comparison RGB stimuli.
    - rgb_contour_cov: The covariance matrix of the unscaled comparison stimuli.
    - ellFits: A dictionary containing the fitted ellipsoid parameters: center, radii,
               eigenvectors (evecs), v (the algebraic parameters of the ellipsoid),
               and chi2 (the goodness of fit measure).
    """
    # Default parameters for ellipsoid fitting. Can be overridden by kwargs.
    ellP = {
        'vecLen':[], # Vector length from the reference stimulus, indicating the magnitude of comparison stimuli.
        'nThetaEllipsoid':200, # Number of theta points for generating the unit ellipsoid.
        'nPhilEllipsoid':100, # Number of phi points for generating the unit ellipsoid.
        'ellipsoid_scaler':1} # Scaling factor for adjusting the fitted ellipsoid size.
    ellP.update(kwargs)
    
    # Generate a unit ellipsoid for ellipsoid fitting.
    circleIn3D = UnitCircleGenerate_3D(ellP['nThetaEllipsoid'], \
                                       ellP['nPhiEllipsoid'])
    # If comparison stimuli are not provided, compute them based on the reference
    # stimulus, vector length, and the grid of XYZ coordinates.
    if len(rgb_comp) == 0:
        #compute the comparison stimuli if not provided
        rgb_comp_unscaled = np.reshape(rgb_ref,(1,1,3)) +\
            np.tile(ellP['vecLen'][:,:,np.newaxis],(1,1,3)) *grid_xyz
        rgb_comp_scaled = np.reshape(rgb_ref,(1,1,3)) + \
            np.tile(ellP['vecLen'][:,:,np.newaxis],(1,1,3)) *\
                ellP['ellipsoid_scaler'] * grid_xyz
    else:
        # Use provided comparison stimuli.
        rgb_comp_unscaled = rgb_comp
        # Scale provided comparison stimuli.
        rgb_comp_scaled = rgb_ref + (rgb_comp_unscaled - rgb_ref)*\
            ellP['ellipsoid_scaler']
    
    # Compute covariance of the unscaled comparison stimuli for ellipsoid fitting.
    if len(rgb_comp_scaled.shape) == 3:
        rgb_comp_unscaled_reshape = rgb_comp_unscaled.reshape(-1,3)
    else:
        rgb_comp_unscaled_reshape = rgb_comp_unscaled
    rgb_contour_cov = np.cov(rgb_comp_unscaled_reshape,rowvar=False)
    
    # Fit an ellipsoid to the unscaled comparison stimuli.
    ellFits = {}
    ellFits['center'], ellFits['radii'], ellFits['evecs'],\
        ellFits['v'], ellFits['chi2'] = ellipsoid_fit(rgb_comp_unscaled_reshape)
    
    # Generate points on the surface of the fitted ellipsoid using the unit ellipsoid.
    fitEllipsoid_unscaled = PointsOnEllipsoid(ellFits['radii'], ellFits['center'],\
                                              ellFits['evecs'], circleIn3D)
    # Scale the fitted ellipsoid points.
    fitEllipsoid_scaled = (fitEllipsoid_unscaled - np.reshape(rgb_ref,(3,1))) * \
        ellP['ellipsoid_scaler'] + np.reshape(rgb_ref,(3,1))
        
    return fitEllipsoid_scaled, fitEllipsoid_unscaled, rgb_comp_scaled,\
        rgb_contour_cov, ellFits
 
#%%
def plot_3D_isothreshold_ellipsoid(x_grid_ref, y_grid_ref, z_grid_ref,
                                   fitEllipsoid, nTheta, nPhi, **kwargs):
    # Default parameters for ellipsoid fitting. Can be overridden by kwargs.
    pltP = {
        'slc_x_grid_ref':np.array(len(x_grid_ref)), 
        'slc_y_grid_ref':np.array(len(y_grid_ref)), 
        'slc_z_grid_ref':np.array(len(z_grid_ref)), 
        'visualize_ref':True,
        'visualize_ellipsoids':True,
        'visualize_thresholdPoints':False,
        'threshold_points':[],
        'ms_ref':100,
        'lw_ref':2,
        'color_ref_rgb':[],
        'color_surf':[],
        'color_threshold':[],
        'fontsize':15,
        'saveFig':False,
        'figDir':'',
        'figName':'Isothreshold_ellipsoids'} 
    pltP.update(kwargs)
    
    #selected ref points
    x_grid_ref_trunc = x_grid_ref[pltP['slc_x_grid_ref']]
    y_grid_ref_trunc = y_grid_ref[pltP['slc_y_grid_ref']]
    z_grid_ref_trunc = z_grid_ref[pltP['slc_z_grid_ref']]
    
    nGridPts_ref_x = len(x_grid_ref_trunc)
    nGridPts_ref_y = len(y_grid_ref_trunc)
    nGridPts_ref_z = len(z_grid_ref_trunc)
    
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1,1,1])
    for i in range(nGridPts_ref_x):
        ii = pltP['slc_x_grid_ref'][i]
        for j in range(nGridPts_ref_y):
            jj = pltP['slc_y_grid_ref'][j]
            for k in range(nGridPts_ref_z):    
                kk = pltP['slc_z_grid_ref'][k]
                if pltP['visualize_ref']:
                    if len(pltP['color_ref_rgb']) == 0:
                        cmap_ijk = [x_grid_ref[ii], y_grid_ref[jj], z_grid_ref[kk]]
                    else:
                        cmap_ijk = pltP['color_ref_rgb']
                    ax.scatter(cmap_ijk[0], cmap_ijk[1], cmap_ijk[2], s=pltP['ms_ref'],\
                               c=cmap_ijk, marker='+', linewidth=pltP['lw_ref'])
    
                if pltP['visualize_ellipsoids']:
                    if len(pltP['color_surf']) == 0:
                        cmap_ijk = [x_grid_ref[ii], y_grid_ref[jj], z_grid_ref[kk]]
                    else:
                        cmap_ijk = pltP['color_surf']
                    ell_ijk = fitEllipsoid[ii, jj, kk]
                    ell_ijk_x = ell_ijk[0].reshape(nPhi, nTheta)
                    ell_ijk_y = ell_ijk[1].reshape(nPhi, nTheta)
                    ell_ijk_z = ell_ijk[2].reshape(nPhi, nTheta)
                    ax.plot_surface(ell_ijk_x, ell_ijk_y, ell_ijk_z, \
                                    color=cmap_ijk, edgecolor='none', alpha=0.5)
    
                if pltP['visualize_thresholdPoints'] and pltP['threshold_points'] is not None:
                    if len(pltP['color_threshold']) == 0:
                        cmap_ijk = [x_grid_ref[ii], y_grid_ref[jj], z_grid_ref[kk]]
                    else:
                        cmap_ijk = pltP['color_threshold']
                    tp = pltP['threshold_points'][ii, jj, kk, :, :]
                    tp_x, tp_y, tp_z = tp[:,:,0], tp[:,:,1], tp[:,:,2]
                    tp_x_f = tp_x.flatten()
                    tp_y_f = tp_y.flatten()
                    tp_z_f = tp_z.flatten()
                    ax.scatter(tp_x_f, tp_y_f, tp_z_f, s=3, c=cmap_ijk, alpha=1)
    
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1]); ax.set_zlim([0, 1])
    ax.set_xticks(sorted([0, 1] + list(x_grid_ref[pltP['slc_x_grid_ref']])))
    ax.set_yticks(sorted([0, 1] + list(y_grid_ref[pltP['slc_y_grid_ref']])))
    ax.set_zticks(sorted([0, 1] + list(z_grid_ref[pltP['slc_z_grid_ref']])))
    ax.set_xlabel('R'); ax.set_ylabel('G'); ax.set_zlabel('B')
    ax.view_init(elev=35, azim=-120)   # Adjust viewing angle for better visualization
    plt.show()
    if pltP['saveFig'] and pltP['figDir'] != '':
        full_path2 = os.path.join(pltP['figDir'],pltP['figName']+'.png')
        fig.savefig(full_path2)
    

#%%
def main():
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
    plot_3D_isothreshold_ellipsoid(stim['grid_ref'], stim['grid_ref'],
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
                                    
                
if __name__ == "__main__":
    main()    
    
    
    
    
    
    