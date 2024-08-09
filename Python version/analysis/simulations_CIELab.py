#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 10:14:11 2024

@author: fangfang
"""

from scipy.io import loadmat
import colour
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import sys
import os

class SimThresCIELab:
    def __init__(self, fileDir):
        # Walk through all subdirectories in fileDir
        for root, dirs, files in os.walk(fileDir):
            for directory in dirs:
                subDirPath = os.path.join(root, directory)
                sys.path.append(subDirPath)
                
        #load data
        T_cones_mat     = loadmat('T_cones.mat')
        self.T_cones    = T_cones_mat['T_cones'] #size: (3, 61)

        B_monitor_mat   = loadmat('B_monitor.mat')
        self.B_monitor  = B_monitor_mat['B_monitor'] #size: (61, 3)

        M_LMSToXYZ_mat  = loadmat('M_LMSToXYZ.mat')
        self.M_LMSToXYZ = M_LMSToXYZ_mat['M_LMSToXYZ'] #size: (3, 3)


    #%% methods
    def get_gridPts(self, X, Y, val_fixed_dim, fixed_dim = list(range(3))):
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
            grid_pts_i[fixed_dim[i]] = val_fixed_dim[i] * np.ones_like(X)
            #Assign the input ranges to the varying dimensions.
            grid_pts_i[varying_dim[0]] = X
            grid_pts_i[varying_dim[1]] = Y
            #Concatenate the individual dimension arrays into a 3D matrix and
            #store it in the output cell array.
            grid_pts[i] = grid_pts_i
            
        return grid_pts
    
    def convert_rgb_lab(self, monitor_Spd, background_RGB, color_RGB):
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
        background_LMS = self.T_CONES @ background_Spd
        # Convert background LMS to XYZ (for use in Lab conversion)
        background_XYZ = self.M_LMS_TO_XYZ @ background_LMS
        
        #RGB -> SPD
        color_Spd = monitor_Spd @ color_RGB
    
        #SPD -> LMS
        color_LMS = self.T_CONES @ color_Spd
     
        #LMS -> XYZ
        color_XYZ = self.M_LMS_TO_XYZ @ color_LMS
    
        #XYZ -> Lab
        background_XYZ_arr = np.array(background_XYZ)
        background_XYZ_reshape = background_XYZ_arr.reshape(1,3)
        background_xyY = colour.XYZ_to_xyY(background_XYZ_reshape)
    
        color_Lab = colour.XYZ_to_Lab(color_XYZ, background_xyY[0]) 
        #print(color_Lab)
        
        return color_Lab, color_XYZ, color_LMS
    
    def compute_deltaE(self, vecLen, background_RGB, ref_RGB, ref_Lab, vecDir):
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
        comp_Lab,_,_ = self.convert_rgb_lab(background_RGB, comp_RGB)
        
        # Calculate the perceptual difference (deltaE) between the reference and comparison
        # stimuli as the Euclidean distance between their Lab values.
        deltaE = np.linalg.norm(comp_Lab - ref_Lab)
        
        return deltaE
    
    def find_vecLen(self, background_RGB, ref_RGB_test, ref_Lab_test, vecDir_test, 
                    deltaE = 1):
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
        deltaE_func = lambda d: abs(self.compute_deltaE(d, background_RGB, ref_RGB_test,\
                                                   ref_Lab_test, vecDir_test, self.T_CONES,\
                                                   self.M_LMS_TO_XYZ, self.B_monitor) - deltaE)
            
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
                        ax[p].scatter(pltParams['rgb_contour'][p,i,j,0],\
                                      pltParams['rgb_contour'][p,i,j,1],\
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
        



