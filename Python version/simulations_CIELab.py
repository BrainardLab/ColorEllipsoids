#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 10:14:11 2024

@author: fangfang
"""

from scipy.io import loadmat
import os
import colour
import math
import numpy as np
from scipy.optimize import minimize
from skimage.measure import EllipseModel
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from itertools import combinations, product

path_str = "/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/FilesFromPsychtoolbox"
os.chdir(path_str)

#%% LOAD DATA WE NEED
#load data
T_cones_mat = loadmat('T_cones.mat')
T_cones = T_cones_mat['T_cones'] #size: (3, 61)

B_monitor_mat = loadmat('B_monitor.mat')
B_monitor = B_monitor_mat['B_monitor'] #size: (61, 3)

M_LMSToXYZ_mat = loadmat('M_LMSToXYZ.mat')
M_LMSToXYZ = M_LMSToXYZ_mat['M_LMSToXYZ'] #size: (3, 3)

#%% FUNCTIONS
def get_gridPts(X, Y, val_fixed_dim, fixed_dim = list(range(3))):
    """
    GET_GRIDPTS Generates grid points for RGB values with one dimension fixed.

    This function returns a cell array of grid points for RGB combinations
    when one of the R, G, or B dimensions is fixed to a specific value. The
    grid points are generated based on the input ranges for the two varying
    dimensions.

    Parameters:
    - X (array; N x N): specifying the range of values for the first varying dimension.
    - Y (array; N x N): pecifying the range of values for the second varying dimension.
        where N is the number of grid points
    - val_fixed_dim (array; 3,): A list or array of values for the fixed dimension(s). 
    - fixed_dim (list): A list indicating which dimension(s) are to be 
        fixed (0 for R, 1 for G, 2 for B). The default value `[0, 1, 2]` 
        indicates all dimensions are considered for fixing.

    Returns:
    - grid_pts (array): NumPy array of shape `(len(fixed_dim), 3, len(X), len(Y))`, 
        representing grid points in RGB space (2nd dimension). Each slice of the
        first dimension corresponds to a specific combination of fixed and varying 
        dimensions, where one or more dimensions are fixed at specified values.
    """
    #Initialize an array to hold the grid points for each fixed dimension.
    grid_pts = np.full((len(fixed_dim),3, X.shape[0], X.shape[1]), np.nan) 
    #Loop through each fixed dimension specified.
    for i in range(len(fixed_dim)):
        #Determine the dimensions that will vary.
        varying_dim = list(set(range(3)) - set([fixed_dim[i]]))
        #Initialize a cell array to hold the current set of grid points.
        grid_pts_i = np.zeros((3, X.shape[0], X.shape[1]))
        #Set the fixed dimension to its specified value across all grid points.
        grid_pts_i[fixed_dim[i],:,:] = val_fixed_dim[i] * np.ones_like(X)
        #Assign the input ranges to the varying dimensions.
        grid_pts_i[varying_dim[0],:,:] = X
        grid_pts_i[varying_dim[1],:,:] = Y
        #Concatenate the individual dimension arrays into a 3D matrix and
        #store it in the output cell array.
        grid_pts[i,:,:,:] = grid_pts_i
        
    return grid_pts

def UnitCircleGenerate(nTheta):
    """
    Generate a set of points on the unit circle in two dimensions.
    nTheta - number of samples around the azimuthal theta (0 to 2pi)

    Coordinates are returned in an 2 by (nTheta) matrix, with the rows
    being the x, y coordinates of the points.
    """
    
    #generate a unit sphere in 3D
    theta = np.linspace(0,2*math.pi,nTheta)
    rho = 1
    xCoords = rho*np.cos(theta)
    yCoords = rho*np.sin(theta)
    
    #stuff the coordinates into a single nTheta 
    x = np.stack((xCoords, yCoords), axis = 0)
    
    return x

#%% FUNCTIONS
def convert_rgb_lab(monitor_Spd, background_RGB, color_RGB,\
                    T_CONES = T_cones, M_LMS_TO_XYZ = M_LMSToXYZ):
    """
    Convert an RGB color value into the CIELab color space using the monitor's 
    spectral power distribution (SPD), the background RGB values, cone sensitivities 
    (T_CONES), and a matrix that converts from LMS (cone responses) to CIEXYZ 
    color space (M_LMS_TO_XYZ).

    Parameters:
    - monitor_Spd (array; N x 3): Spectral power distribution of the monitor.
    - background_RGB (array; 3 x 1): Background RGB values used for normalization.
    - T_CONES (array; 3 x N): Matrix of cone sensitivities for absorbing photons at different wavelengths.
    - M_LMS_TO_XYZ (array; 3 x 3): Matrix to convert LMS cone responses to CIEXYZ.
    - color_RGB (array; 3,): RGB color value(s) to be converted.
        where N is the number of selected wavelengths
    
    Returns:
    - color_Lab (array; 3,): The converted color(s) in CIELab color space, a 1D array.
    - color_XYZ (array; 3,): The intermediate CIEXYZ color space representation, a 1D array.
    - color_LMS (array; 3,): The LMS cone response representation, a 1D array.

    """

    # Convert background RGB to SPD using the monitor's SPD
    background_Spd = monitor_Spd @ background_RGB
    # Convert background SPD to LMS (cone response)
    background_LMS = T_CONES @ background_Spd
    # Convert background LMS to XYZ (for use in Lab conversion)
    background_XYZ = M_LMS_TO_XYZ @ background_LMS
    
    #RGB -> SPD
    color_Spd = monitor_Spd @ color_RGB

    #SPD -> LMS
    color_LMS = T_CONES @ color_Spd
 
    #LMS -> XYZ
    color_XYZ = M_LMS_TO_XYZ @ color_LMS

    #XYZ -> Lab
    background_XYZ_arr = np.array(background_XYZ)
    background_XYZ_reshape = background_XYZ_arr.reshape(1,3)
    background_xyY = colour.XYZ_to_xyY(background_XYZ_reshape)

    color_Lab = colour.XYZ_to_Lab(color_XYZ, background_xyY[0]) 
    #print(color_Lab)
    
    return color_Lab, color_XYZ, color_LMS

def compute_deltaE(vecLen, background_RGB, ref_RGB, ref_Lab, vecDir,\
                  T_CONES = T_cones, M_LMS_TO_XYZ = M_LMSToXYZ,\
                  B_MONITOR = B_monitor):
    """
    Computes the perceptual difference (deltaE) between a reference stimulus
    and a comparison stimulus in the CIELab color space. The comparison stimulus
    is derived based on a specified chromatic direction and length from the reference.

    Parameters:
    - vecLen (array): The length to move in the specified direction from the 
        reference stimulus.
    - background_RGB (array; 3 x 1): The RGB values of the background, used 
        in the conversion process.
    - ref_RGB (array; 3,): The RGB values of the reference stimulus.
    - ref_Lab (array; 3,): The CIELab values of the reference stimulus.
    - vecDir (array; 1 x 3): The direction vector along which the comparison 
        stimulus varies from the reference.

    Returns:
    - deltaE (float): The computed perceptual difference between the reference 
        and comparison stimuli.
    """

    #pdb.set_trace()
    # Calculate the RGB values for the comparison stimulus by adjusting the reference RGB
    # along the specified chromatic direction by the given vector length (vecLen).
    comp_RGB = ref_RGB + vecDir[0] * vecLen
    
    # Convert the computed RGB values of the comparison stimulus into Lab values
    # using the provided parameters and the background RGB. 
    comp_Lab,_,_ = convert_rgb_lab(B_MONITOR, background_RGB, comp_RGB,\
                                  T_CONES, M_LMS_TO_XYZ)
    
    # Calculate the perceptual difference (deltaE) between the reference and comparison
    # stimuli as the Euclidean distance between their Lab values.
    deltaE = np.linalg.norm(comp_Lab - ref_Lab)
    
    return deltaE

def find_vecLen(background_RGB, ref_RGB_test, ref_Lab_test, vecDir_test, deltaE = 1,                  
                T_CONES = T_cones, M_LMS_TO_XYZ = M_LMSToXYZ, B_MONITOR = B_monitor):
    """
    This function finds the optimal vector length for a chromatic direction
    that achieves a target perceptual difference in the CIELab color space.

    Parameters:
    - background_RGB (array): The RGB values of the background
    - ref_RGB_test (array): The RGB values of the reference stimulus
    - ref_Lab_test (array): The CIELab values of the reference stimulus
    - vecDir_test (array): The chromatic direction vector for comparison stimulus variation
    - deltaE (float): The target deltaE value (e.g., 1 JND)
    
    Returns:
    - opt_vecLen (float): The optimal vector length that achieves the target deltaE value
    """
    #The lambda function computes the absolute difference between the
    #deltaE obtained from compute_deltaE function and the target deltaE.
    deltaE_func = lambda d: abs(compute_deltaE(d, background_RGB, ref_RGB_test,\
                                               ref_Lab_test, vecDir_test, T_CONES,\
                                               M_LMS_TO_XYZ, B_MONITOR) - deltaE)
        
    # Define the lower and upper bounds for the search of the vector length.
    # Define the number of runs for optimization to ensure we don't get stuck 
    # at local minima
    lb, ub, N_runs = 0, 0.1, 3
    # Generate initial points for the optimization algorithm within the bounds.
    init = np.random.rand(1, N_runs) * (ub - lb) + lb
    # Set the options for the optimization algorithm.
    options = {'maxiter': 1e5, 'disp': False}
    # Initialize arrays to store the vector lengths and corresponding deltaE 
    #values for each run.
    vecLen_n = np.empty(N_runs)
    deltaE_n = np.empty(N_runs)
    
    # Loop over the number of runs to perform the optimizations.
    for n in range(N_runs):
        # Use scipy's minimize function to find the vector length that minimizes
        # the difference to the target deltaE. SLSQP method is used for 
        #constrained optimization.
        res = minimize(deltaE_func, init[0][n],method='SLSQP', bounds=[(lb, ub)], \
                       options=options)
        # Store the result of each optimization run.
        vecLen_n[n] = res.x
        deltaE_n[n] = res.fun
        
    # Identify the index of the run that resulted in the minimum deltaE value.
    idx_min = np.argmin(deltaE_n)
    # Choose the optimal vector length from the run with the minimum deltaE value.
    opt_vecLen = vecLen_n[idx_min]
    
    return opt_vecLen

#%% FUNCTIONS
def PointsOnEllipseQ(a, b, theta, xc, yc, nTheta = 200):
    """
    Generates points on an ellipse using parametric equations.
    
    The function scales points from a unit circle to match the given ellipse
    parameters and then rotates the points by the specified angle.

    Parameters:
    - a (float): The semi-major axis of the ellipse.
    - b (float): The semi-minor axis of the ellipse.
    - theta (float): The rotation angle of the ellipse in degrees, measured 
        from the x-axis to the semi-major axis in the counter-clockwise 
        direction.
    - xc (float): The x-coordinate of the center of the ellipse
    - yc (float): The y-coordinate of the center of the ellipse
    - nTheta (int): The number of angular points used to generate the unit 
        circle, which is then scaled to the ellipse. More points will make the
        ellipse appear smoother. Default value is 200.   

    Returns:
    - x_rotated (array): The x-coordinates of the points on the ellipse.
    - y_rotated (array): The y-coordinates of the points on the ellipse.

    """
    #generate points for unit circle
    circle = UnitCircleGenerate(nTheta) #shape: (2,100)
    x_circle, y_circle = circle[0,:], circle[1,:]
    
    #scale the unit circle to the ellipse
    x_ellipse = a * x_circle
    y_ellipse = b * y_circle
    
    #Rotate the ellipse
    angle_rad = np.radians(theta)
    x_rotated = x_ellipse * np.cos(angle_rad) - y_ellipse * np.sin(angle_rad) + xc
    y_rotated = x_ellipse * np.sin(angle_rad) + y_ellipse * np.cos(angle_rad) + yc
    
    return x_rotated, y_rotated

def fit_2d_isothreshold_contour(ref_RGB, comp_RGB, grid_theta_xy, **kwargs):
    """
    Fits an ellipse to 2D isothreshold contours for color stimuli.
    
    This function takes reference and comparison RGB values and fits an ellipse
    to the isothreshold contours based on the provided grid of angle points.
    It allows for scaling and adjusting of the contours with respect to a 
    reference stimulus.
    
    Parameters:
    - ref_RGB (array; 3,): The reference RGB values.
    - comp_RGB (array): The comparison RGB values. If empty, they will be 
        computed within the function.
    - grid_theta_xy (array; 2 x M): A grid of angles (in the xy plane) used to 
        generate the comparison stimuli.
            where M is the number of chromatic directions
    - kwargs (dict)
        Additional keyword arguments:
        - vecLength: Length of the vector (optional).
        - nThetaEllipse: Number of angular points to use for fitting the ellipse (default 200).
        - varyingRGBplane: The RGB plane that varies (optional).
        - ellipse_scaler: The factor by which the ellipse is scaled (default 5).

    Returns:
    - fitEllipse_scaled (array; 2 x nThethaEllipse): The scaled coordinates of 
        the fitted ellipse.
    - fitEllipse_unscaled (array; 2 x nThethaEllipse): The unscaled coordinates 
        of the fitted ellipse.
    - rgb_comp_scaled (array; 2 x M): The scaled comparison stimuli RGB values.
    - rgb_contour_cov (array; 2 x 2): The covariance matrix of the comparison stimuli.
    - ellipse_params (list): The parameters of the fitted ellipse: 
        [xCenter, yCenter, majorAxis, minorAxis, theta].

    """
    
    # Accessing a keyword argument with a default value if it's not provided
    vecLen = kwargs.get('vecLength', [])
    nThetaEllipse = kwargs.get('nThetaEllipse',200)
    varyingRGBplane = kwargs.get('varyingRGBplan',[])
    ellipse_scaler = kwargs.get('ellipse_scaler', 5)
    
    #Truncate the reference RGB to the specified dimensions
    rgb_ref_trunc = ref_RGB[varyingRGBplane]
    
    #Compute or use provided comparison stimuli
    if comp_RGB == []:
        #Compute the comparison stimuli if not provided
        rgb_comp_unscaled = rgb_ref_trunc.reshape(2,1) + vecLen * grid_theta_xy
        rgb_comp_scaled = rgb_ref_trunc.reshape(2,1) + vecLen * \
            ellipse_scaler * grid_theta_xy
        
        #Compute covariance of the unscaled comparison stimuli
        rgb_contour_cov = np.cov(rgb_comp_unscaled)
        
    else:
        #use provided comparison stimuli
        rgb_comp_trunc = comp_RGB[varyingRGBplane,:]
        rgb_comp_scaled = rgb_ref_trunc + rgb_comp_trunc * ellipse_scaler
        rgb_contour_cov = np.cov(rgb_comp_trunc)
        
    #fit an ellipse
    ellipse = EllipseModel()
    #Points need to be in (N,2) array where N is the number of points 
    #Each row is a point [x,y]
    ellipse.estimate(np.transpose(rgb_comp_unscaled))
    
    #pdb.set_trace()
            
    #Parameters of the fitted ellipse
    xCenter, yCenter, majorAxis, minorAxis, theta_rad = ellipse.params
    theta = np.rad2deg(theta_rad)
    
    #Adjust the fitted ellipse based on the reference stimulus
    fitEllipse_unscaled_x, fitEllipse_unscaled_y = \
        PointsOnEllipseQ(majorAxis,minorAxis,theta,xCenter,yCenter, nThetaEllipse)
        
    fitEllipse_unscaled = np.stack((fitEllipse_unscaled_x, fitEllipse_unscaled_y), axis = 0)
        
    #scale the fitted ellipse
    fitEllipse_scaled = (fitEllipse_unscaled - rgb_ref_trunc.reshape(2,1)) *\
        ellipse_scaler + rgb_ref_trunc.reshape(2,1)
        
    return fitEllipse_scaled, fitEllipse_unscaled, rgb_comp_scaled, \
        rgb_contour_cov, [xCenter, yCenter, majorAxis, minorAxis, theta]
        
#%% FUNCTIONS
def plot_2D_randRef_nearContourComp(ax, fig, xref, xcomp, idx_fixedPlane,\
                                    fixedVal, bounds, **kwargs):
    # Default parameters for ellipsoid fitting. Can be overridden by kwargs.
    pltP = {
        'visualize_bounds': True,
        'visualize_lines': True,
        'bounds_alpha':0.1,
        'linealpha':0.5,
        'ref_marker': '+',
        'ref_markersize': 20,
        'ref_markeralpha': 0.8,
        'comp_marker': 'o',
        'comp_markersize': 4,
        'comp_markeralpha': 0.8,     
        'plane_2D':'',
        'flag_rescale_axes_label':True,
        'flag_add_trialNum_title': False,
        'fontsize':8,
        'saveFig':False,
        'figDir':'',
        'figName':'RandomSamples'} 
    pltP.update(kwargs)
    
    plt.rcParams['figure.dpi'] = 250 
    cmap = (xref+1)/2
    cmap = np.insert(cmap, idx_fixedPlane, np.ones((xref.shape[0],))*fixedVal, axis=1)
    # Add grey patch
    if pltP['visualize_bounds']:
        rectangle = Rectangle((bounds[0], bounds[0]), bounds[1] - bounds[0],\
                              bounds[1] - bounds[0], facecolor='grey', alpha= pltP['bounds_alpha'])  # Adjust alpha for transparency
        rectangle.set_label('Bounds for the reference')  # Set the label here
        ax.add_patch(rectangle)
    
    ax.scatter(xref[:,0],xref[:,1], c = cmap, marker = pltP['ref_marker'],\
               s = pltP['ref_markersize'], alpha = pltP['ref_markeralpha'],\
               label = 'Reference stimulus')
    ax.scatter(xcomp[:,0], xcomp[:,1], c = cmap, marker = pltP['comp_marker'],\
               s = pltP['comp_markersize'], alpha = pltP['comp_markeralpha'],\
               label = 'Comparison stimulus') 
    if pltP['visualize_lines']:
        for l in range(xref.shape[0]):
            ax.plot([xref[l,0],xcomp[l,0]], [xref[l,1],xcomp[l,1]], c = cmap[l],\
                    alpha = pltP['linealpha'],lw = 0.5)
    
    plt.grid(alpha = 0.2)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1])
    ticks = np.sort(np.concatenate((np.linspace(-0.5, 0.5, 3), np.array([-0.85, 0.85]))))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    if pltP['flag_rescale_axes_label']:
        ax.set_xticklabels([str((f+1)/2) for f in ticks])
        ax.set_yticklabels([str((f+1)/2) for f in ticks])
    ax.tick_params(axis='both', which='major', labelsize=pltP['fontsize'])
    if pltP['plane_2D'] != '':
        ax.set_xlabel(pltP['plane_2D'][0], fontsize=pltP['fontsize']);
        ax.set_ylabel(pltP['plane_2D'][1], fontsize=pltP['fontsize'])
    ttl = pltP['plane_2D'] + ' (n = ' +str(xref.shape[0])+')' if pltP['flag_add_trialNum_title'] else pltP['plane_2D']
    ax.set_title(ttl, fontsize=pltP['fontsize'])
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.47),fontsize = pltP['fontsize'])
    fig.tight_layout(); plt.show()
    if pltP['saveFig'] and pltP['figDir'] != '':
        full_path = os.path.join(pltP['figDir'],pltP['figName']+'.png') 
        fig.savefig(full_path)   
        
def plot_2D_isothreshold_contour(x_grid_ref, y_grid_ref, fitEllipse,
                                 fixed_RGBvec,**kwargs):
    #default values for optional parameters
    pltParams = {
        'slc_x_grid_ref': np.arange(len(x_grid_ref)),
        'slc_y_grid_ref': np.arange(len(y_grid_ref)),
        'visualizeRawData': False,
        'WishartEllipses': np.array([]),
        'WishartEllipses_contour_CI':[],
        'IndividualEllipses_contour_CI':[],
        'ExtrapEllipses':np.array([]),
        'rgb_contour':np.array([]),
        'rgb_background':True,
        'subTitles':['GB plane', 'RB plane', 'RG plane'],
        'refColor':[0,0,0],
        'EllipsesColor':[0,0,0],
        'WishartEllipsesColor':[76/255, 153/255,0],
        'ExtrapEllipsesColor':[0.5,0.5,0.5],
        'EllipsesLine':'--',
        'WishartEllipsesLine':'-',
        'ExtrapEllipsesLine':':',
        'xlabel':'',
        'ylabel':'',
        'fontsize':10,
        'saveFig':False,
        'figName':'Isothreshold_contour',
        }
    
    #update default parameters with any user-provided values
    pltParams.update(kwargs)
    
    nPlanes = fitEllipse.shape[0]
    nGridPts_ref_x = len(pltParams['slc_x_grid_ref'])
    nGridPts_ref_y = len(pltParams['slc_y_grid_ref'])
    
    fig, ax = plt.subplots(1, nPlanes,figsize=(20, 6))
    
    for p in range(nPlanes):
        if pltParams['rgb_background']:
            #fill in RGB color
            print('later')
        
        #Ground truth
        for i in range(nGridPts_ref_x):
            for j in range(nGridPts_ref_y):
                #reference location 
                ax[p].scatter(x_grid_ref[pltParams['slc_x_grid_ref']],\
                              y_grid_ref[pltParams['slc_y_grid_ref']],\
                                  s = 10,c = pltParams['refColor'],marker ='+',linewidth = 1)
                
                #ellipses
                ax[p].plot(fitEllipse[p,i,j,0,:],\
                           fitEllipse[p,i,j,1,:],\
                          linestyle = pltParams['EllipsesLine'],\
                          color = pltParams['EllipsesColor'],\
                          linewidth = 1)
                    
                #individual ellipse
                if pltParams['visualizeRawData']:
                    ax[p].scatter(pltParams['rgb_contour'][p,i,j,0,:],\
                                  pltParams['rgb_contour'][p,i,j,1,:],\
                                      marker ='o', color = [0.6,0.6,0.6],\
                                          s = 20)
                    
        ax[p].set_xlim([0,1])
        ax[p].set_ylim([0,1])
        ax[p].set_aspect('equal','box')
        ax[p].set_xticks(np.arange(0,1.2,0.2))
        ax[p].set_yticks(np.arange(0,1.2,0.2))
        ax[p].set_title(pltParams['subTitles'][p])
        if pltParams['xlabel'] == '': xlbl = pltParams['subTitles'][p][0]
        if pltParams['ylabel'] == '': ylbl = pltParams['subTitles'][p][1]
        ax[p].set_xlabel(xlbl)
        ax[p].set_ylabel(ylbl)
        ax[p].tick_params(axis='both', which='major', labelsize=pltParams['fontsize'])
        
#%% 3D
def plot_3D_randRef_nearContourComp(ax, fig, xref, xcomp, **kwargs):
    # Default parameters for ellipsoid fitting. Can be overridden by kwargs.
    pltP = {
        'visualize_lines': True,
        'linealpha':0.5,
        'ref_marker': '+',
        'ref_markersize': 20,
        'ref_markeralpha': 0.8,
        'comp_marker': 'o',
        'comp_markersize': 4,
        'comp_markeralpha': 0.8,        
        'fontsize':8,
        'flag_rescale_axes_label':True,
        'saveFig':False,
        'figDir':'',
        'figName':'3D_randRef_nearContourComp'} 
    pltP.update(kwargs)
    
    # Mapping data to RGB color space for visualization
    color_map_ref = (xref + 1) / 2  # Ensure the colors are within [0,1]
    color_map_comp = (xcomp + 1) / 2
    ticks = np.unique(xref)
    
    ax.scatter(xref[:, 0], xref[:, 1], xref[:, 2], c=color_map_ref,\
               marker= pltP['ref_marker'], s= pltP['ref_markersize'], \
                   alpha= pltP['ref_markeralpha'], label = 'Reference stimulus')
    ax.scatter(xcomp[:, 0], xcomp[:, 1], xcomp[:, 2], c=color_map_comp,\
               marker=pltP['comp_marker'], s= pltP['comp_markersize'],\
                   alpha= pltP['comp_markeralpha'], label = 'Comparison stimulus')
    
    for l in range(xref.shape[0]):
        ax.plot([xref[l, 0], xcomp[l, 0]],[xref[l, 1], xcomp[l, 1]],\
                [xref[l, 2], xcomp[l, 2]],
                color= np.array(color_map_ref[l]), alpha= pltP['linealpha'], lw= 0.5)
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)
    if pltP['flag_rescale_axes_label']:
        ax.set_xticklabels([str((f+1)/2) for f in ticks])
        ax.set_yticklabels([str((f+1)/2) for f in ticks])
        ax.set_zticklabels([str((f+1)/2) for f in ticks])
    ax.set_xlabel('R')
    ax.set_ylabel('G')
    ax.set_zlabel('B')
    ttl = '3D RGB space ' + ' (n = ' +str(xref.shape[0])+')' if pltP['flag_add_trialNum_title'] else pltP['plane_2D']
    ax.set_title(ttl, fontsize=pltP['fontsize'])
    ax.grid(True)
    ax.set_aspect('equal')
    ax.tick_params(axis='both', which='major', labelsize= pltP['fontsize'])
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.24),fontsize = pltP['fontsize'])

    if pltP['saveFig'] and pltP['figDir'] != '':
        full_path = os.path.join(pltP['figDir'],pltP['figName']+'.png') 
        fig.savefig(full_path, bbox_inches='tight', pad_inches=0.3)   




