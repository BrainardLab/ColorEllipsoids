# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from scipy.io import loadmat
import os
import colour
import math
import numpy as np
import pdb #pdb.set_trace()
from scipy.optimize import minimize
from skimage.measure import EllipseModel
import matplotlib.pyplot as plt
import matplotlib.patches as patches

path_str = "/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/FilesFromPsychtoolbox"
os.chdir(path_str)

#%%
def get_gridPts(X, Y, val_fixed_dim, fixed_dim = list(range(3))):
    """
GET_GRIDPTS Generates grid points for RGB values with one dimension fixed.

This function returns a cell array of grid points for RGB combinations
when one of the R, G, or B dimensions is fixed to a specific value. The
grid points are generated based on the input ranges for the two varying
dimensions.

Parameters:
X - A vector of values for the first varying dimension.
Y - A vector of values for the second varying dimension.
fixed_dim - An array of indices (1 for R, 2 for G, 3 for B) indicating 
            which dimension(s) to fix. Can handle multiple dimensions.
val_fixed_dim - An array of values corresponding to each fixed dimension
                specified in `fixed_dim`. Each value in `val_fixed_dim`
                is used to fix the value of the corresponding dimension.

Returns:
grid_pts - A cell array where each cell contains a 3D matrix of grid 
           points. Each matrix corresponds to a set of RGB values where 
           one dimension is fixed. The size of each matrix is determined 
           by the lengths of X and Y, with the third dimension representing
           the RGB channels.
    """
    #Initialize a cell array to hold the grid points for each fixed dimension.
    grid_pts = [] 
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
        grid_pts.append(grid_pts_i)
        
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

#%%
#load data
param = {}
T_cones_mat = loadmat('T_cones.mat')
param['T_cones'] = T_cones_mat['T_cones'] #size: (3, 61)

B_monitor_mat = loadmat('B_monitor.mat')
param['B_monitor'] = B_monitor_mat['B_monitor'] #size: (61, 3)

M_LMSToXYZ_mat = loadmat('M_LMSToXYZ.mat')
param['M_LMSToXYZ'] = M_LMSToXYZ_mat['M_LMSToXYZ'] #size: (3, 3)

#First create a cube and select the RG, the RB and the GB planes
param['nGridPts'] = 100
param['grid'] = np.linspace(0,1,param['nGridPts'])
param['x_grid'], param['y_grid'] = np.meshgrid(param['grid'], param['grid'])

#number of selected planes
param['nPlanes'] = 3

#%%
def convert_rgb_lab(monitor_Spd, background_RGB, color_RGB,\
                    T_CONES = param['T_cones'],\
                    M_LMS_TO_XYZ = param['M_LMSToXYZ']):
    """
    Convert an RGB color value into the CIELab color space using the monitor's 
    spectral power distribution (SPD), the background RGB values, cone sensitivities 
    (T_CONES), and a matrix that converts from LMS (cone responses) to CIEXYZ 
    color space (M_LMS_TO_XYZ).

    Parameters:
    - monitor_Spd: Spectral power distribution of the monitor.
    - background_RGB: Background RGB values used for normalization.
    - T_CONES: Matrix of cone sensitivities for absorbing photons at different wavelengths.
    - M_LMS_TO_XYZ: Matrix to convert LMS cone responses to CIEXYZ.
    - color_RGB: RGB color value(s) to be converted.
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

    color_Lab = colour.XYZ_to_Lab(color_XYZ, background_xyY[0]) #xyz2lab(color_XYZ)#
    #print(color_Lab)
    
    return color_Lab, color_XYZ, color_LMS

def compute_deltaE(vecLen, background_RGB, ref_RGB, ref_Lab, vecDir,\
                  T_CONES = param['T_cones'], \
                  M_LMS_TO_XYZ = param['M_LMSToXYZ'],\
                  B_monitor = param['B_monitor']):
    """
    Computes the perceptual difference (deltaE) between a reference stimulus
    and a comparison stimulus in the CIELab color space. The comparison stimulus
    is derived based on a specified chromatic direction and length from the reference.

    Parameters:
    - vecLen (float): The length to move in the specified direction from the reference stimulus.
    - background_RGB (array): The RGB values of the background, used in the conversion process.
    - ref_RGB (array): The RGB values of the reference stimulus.
    - ref_Lab (array): The CIELab values of the reference stimulus.
    - vecDir (array): The direction vector along which the comparison stimulus varies from the reference.

    Returns:
    - deltaE (float): The computed perceptual difference between the reference and comparison stimuli.
    """

    #pdb.set_trace()
    # Calculate the RGB values for the comparison stimulus by adjusting the reference RGB
    # along the specified chromatic direction by the given vector length (vecLen).
    comp_RGB = ref_RGB + vecDir[0] * vecLen
    
    # Convert the computed RGB values of the comparison stimulus into Lab values
    # using the provided parameters and the background RGB. 
    comp_Lab,_,_ = convert_rgb_lab(B_monitor, background_RGB, comp_RGB,\
                                  T_CONES, M_LMS_TO_XYZ)
    
    # Calculate the perceptual difference (deltaE) between the reference and comparison
    # stimuli as the Euclidean distance between their Lab values.
    deltaE = np.linalg.norm(comp_Lab - ref_Lab)
    
    return deltaE

def find_vecLen(background_RGB, ref_RGB_test, ref_Lab_test, vecDir_test, deltaE = 1,                  
                T_CONES = param['T_cones'], M_LMS_TO_XYZ = param['M_LMSToXYZ'],
                B_monitor = param['B_monitor']):
    deltaE_func = lambda d: abs(compute_deltaE(d, background_RGB, ref_RGB_test,\
                                               ref_Lab_test, vecDir_test, T_CONES,\
                                               M_LMS_TO_XYZ, B_monitor) - deltaE)
        
    
    lb, ub, N_runs = 0, 0.1, 3
    init = np.random.rand(1, N_runs) * (ub - lb) + lb
    options = {'maxiter': 1e5, 'disp': False}
    vecLen_n = np.empty(N_runs)
    deltaE_n = np.empty(N_runs)
    
    for n in range(N_runs):
        res = minimize(deltaE_func, init[0][n],method='SLSQP', bounds=[(lb, ub)], options=options)
        vecLen_n[n] = res.x
        deltaE_n[n] = res.fun
        
    # Finding the optimal value
    idx_min = np.argmin(deltaE_n)
    opt_vecLen = vecLen_n[idx_min]
    
    return opt_vecLen

#%%
def PointsOnEllipseQ(a, b, theta, xc, yc, nTheta = 200):
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
        rgb_comp_trunc = comp_RGB[idx_varyingDim,:]
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

#%%
#for RG / RB / GB plane, we fix the B / G / R value to be one of the following
stim = {}
stim['fixed_RGBvec'] = 0.5

#get the grid points for those three planes with one dimension having a specific fixed value
stim['plane_points'] = get_gridPts(param['x_grid'],\
                                   param['y_grid'],\
                                   np.full(3, stim['fixed_RGBvec']))

#set a grid for the reference stimulus
#pick 5 x 5 reference points 
stim['grid_ref'] = np.arange(0.2, 0.8, 0.15)
stim['nGridPts_ref'] = len(stim['grid_ref'])

stim['x_grid_ref'], stim['y_grid_ref'] = np.meshgrid(stim['grid_ref'],stim['grid_ref'])

#get the grid points for the reference stimuli of each plane
stim['ref_points'] = get_gridPts(stim['x_grid_ref'],\
                                 stim['y_grid_ref'],\
                                 np.full(3, stim['fixed_RGBvec']))
    
#compute iso-threshold contour
#set the background RGB
stim['background_RGB'] = stim['fixed_RGBvec'] * np.ones((param['nPlanes'],1))

#sample total of 16 directions (0 to 360 deg) 
stim['numDirPts'] = 16
stim['grid_theta'] = np.linspace(0,2*math.pi,stim['numDirPts'])
stim['grid_theta_xy'] = np.stack((np.cos(stim['grid_theta']),\
                                 np.sin(stim['grid_theta'])),\
                                 axis = 0)

#define threshold as deltaE = 0.5
stim['deltaE_1JND'] = 1

#make a finer grid for the direction (just for the purpose of visualization)
plt_specifics = {}
#the raw isothreshold contou is very tiny, we can amplify it by 5 times for the purpose of visualization
plt_specifics['contour_scaler'] = 5
plt_specifics['nThetaEllipse'] = 200
plt_specifics['colorMatrix'] = stim['ref_points']
plt_specifics['circleIn2D'] = UnitCircleGenerate(plt_specifics['nThetaEllipse'])

#%%
#initialize
results = {}
results['ref_Lab'] = np.full((param['nPlanes'], stim['nGridPts_ref'],\
                            stim['nGridPts_ref'],3), np.nan)
results['opt_vecLen'] = np.full((param['nPlanes'], stim['nGridPts_ref'],\
                            stim['nGridPts_ref'],stim['numDirPts']), np.nan)
results['fitEllipse_scaled'] = np.full((param['nPlanes'],stim['nGridPts_ref'],\
                            stim['nGridPts_ref'],2,plt_specifics['nThetaEllipse']),\
                                np.nan)
results['fitEllipse_unscaled'] = np.full(results['fitEllipse_scaled'].shape,\
                                         np.nan)
results['rgb_comp_contour_scaled'] = np.full((param['nPlanes'], stim['nGridPts_ref'],\
                            stim['nGridPts_ref'], 2, stim['numDirPts']),  np.nan)
results['rgb_comp_contour_cov'] = np.full((param['nPlanes'], stim['nGridPts_ref'],\
                            stim['nGridPts_ref'],2,2),np.nan)
results['ellParams'] = np.full((param['nPlanes'], stim['nGridPts_ref'],\
                            stim['nGridPts_ref'], 5),  np.nan) #5 free parameters for the ellipse

#for each fixed R / G / B value in the BG / RB / RG plane
for p in range(param['nPlanes']):
    #vecDir is a vector that tells us how far we move along a specific direction 
    vecDir = np.zeros((1,param['nPlanes']))
    #indices for the varying chromatic directions 
    #GB plane: [1,2]; RB plane: [0,2]; RG plane: [0,1]
    idx_varyingDim_full = np.arange(0,param['nPlanes'])
    idx_varyingDim = idx_varyingDim_full[idx_varyingDim_full != p]
    
    #for each reference stimulus
    for i in range(stim['nGridPts_ref']):
        for j in range(stim['nGridPts_ref']):
            #grab the reference stimulus' RGB
            rgb_ref_pij = stim['ref_points'][p][:,i,j]
            #convert it to Lab
            Lab_ref_pij,_,_ = convert_rgb_lab(param['B_monitor'],\
                              stim['background_RGB'], rgb_ref_pij)
            results['ref_Lab'][p,i,j,:] = Lab_ref_pij
            
            #for each chromatic direction
            for k in range(stim['numDirPts']):
                #determine the direction we are varying
                vecDir[0][idx_varyingDim] = stim['grid_theta_xy'][:,k]
                
                #fun minimize to search for the magnitude of vector that 
                #leads to a pre-determined deltaE
                results['opt_vecLen'][p,i,j,k] = find_vecLen(stim['background_RGB'],\
                                                            rgb_ref_pij, Lab_ref_pij,\
                                                            vecDir,stim['deltaE_1JND'])
            
            #fit an ellipse
            results['fitEllipse_scaled'][p,i,j,:,:],results['fitEllipse_unscaled'][p,i,j,:,:],\
                results['rgb_comp_contour_scaled'][p,i,j,:,:],\
                results['rgb_comp_contour_cov'][p,i,j,:,:],\
                results['ellParams'][p,i,j,:] = fit_2d_isothreshold_contour(rgb_ref_pij, [], \
                stim['grid_theta_xy'],vecLength = results['opt_vecLen'][p,i,j,:],\
                varyingRGBplan = idx_varyingDim,\
                    nThetaEllipse = plt_specifics['nThetaEllipse'],\
                    ellipse_scaler = plt_specifics['contour_scaler'])
                                                        
#%% PLOTTING 
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

    x = np.array([0,0,1,1])
    y = np.array([1,0,0,1])
    
    fig, ax = plt.subplots(1, nPlanes,figsize=(20, 6))
    
    for p in range(nPlanes):
        #pdb.set_trace()
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

    #plt.show()

#%%
plot_2D_isothreshold_contour(stim['x_grid_ref'], stim['y_grid_ref'],\
                             results['fitEllipse_scaled'], [],\
                             visualizeRawData = True,\
                                 rgb_contour = results['rgb_comp_contour_scaled'],\
                                 EllipsesLine = '-', fontsize =12)    










