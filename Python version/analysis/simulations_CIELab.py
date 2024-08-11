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
import sys
import os

class SimThresCIELab:
    def __init__(self, fileDir, background_rgb):
        """
        Parameters:
        - background_RGB (array; 3 x 1): Background RGB values used for normalization.
        - T_CONES (array; 3 x N): Matrix of cone sensitivities for absorbing photons at different wavelengths.
        - monitor_Spd (array; N x 3): Spectral power distribution of the monitor.
        - M_LMS_TO_XYZ (array; 3 x 3): Matrix to convert LMS cone responses to CIEXYZ.
        """
        self.background_rgb = background_rgb

        sys.path.append(fileDir)
        os.chdir(fileDir)
                
        #load data
        T_cones_mat     = loadmat('T_cones.mat')
        self.T_CONES    = T_cones_mat['T_cones'] #size: (3, 61)

        B_monitor_mat   = loadmat('B_monitor.mat')
        self.B_MONITOR = B_monitor_mat['B_monitor'] #size: (61, 3)

        M_LMSToXYZ_mat  = loadmat('M_LMSToXYZ.mat')
        self.M_LMS_TO_XYZ = M_LMSToXYZ_mat['M_LMSToXYZ'] #size: (3, 3)
        
        #number of selected planes
        self.plane_2D_list  = ['GB plane', 'RB plane', 'RG plane']   
        self.nPlanes        = len(self.plane_2D_list)
        self.plane_2D_dict  = dict(zip(self.plane_2D_list, list(range(self.nPlanes))))
        self.varying_dims   = [[1,2],[0,2],[0,1]]

    #%% methods
    def _get_plane_1slice(self, grid_lb, grid_ub, num_grid_pts,fixed_val, plane_2D):
        """
        Generates a 2D slice of a 3D plane with one fixed dimension.
        
        Parameters:
        grid_lb (float): Lower bound of the grid.
        grid_ub (float): Upper bound of the grid.
        num_grid_pts (int): Number of grid points in each dimension.
        fixed_val (float): Fixed value for the plane's constant dimension.
        plane_2D (str): Identifier for the 2D plane being generated (e.g., 'XY', 'XZ').
        
        Returns:
        tuple: A tuple containing:
            - plane_1slice (numpy.ndarray, shape: (3 x num_grid_pts x num_grid_pts)): 
                3D array representing the slice with varying values in two dimensions and 
                a fixed value in the third.
            - grid_1d (numpy.ndarray, shape: (num_grid_pts,)): 1D array of grid points 
                used for the X and Y dimensions.
            - X (numpy.ndarray, shape: (num_grid_pts x num_grid_pts)): 2D array of X coordinates.
            - Y (numpy.ndarray, shape: (num_grid_pts x num_grid_pts)): 2D array of Y coordinates.
        """
        
        # Generate a 1D grid and mesh grids for the X and Y dimensions
        grid_1d = np.linspace(grid_lb, grid_ub, num_grid_pts)
        X, Y = np.meshgrid(grid_1d, grid_1d)
        
        # Initialize a 3D array to store the slice with NaN values
        plane_1slice = np.full((self.nPlanes, num_grid_pts, num_grid_pts), np.nan)
        
        # Identify which dimension remains constant and which two vary
        plane_2D_idx = self.plane_2D_dict[plane_2D]
        varying_dim = self.varying_dims[plane_2D_idx]
        
        # Assign the meshgrid values to the varying dimensions in the 3D slice
        plane_1slice[varying_dim[0]] = X
        plane_1slice[varying_dim[1]] = Y
        
        # Assign the fixed value to the fixed dimension
        plane_1slice[plane_2D_idx] = np.ones((num_grid_pts, num_grid_pts))*fixed_val
            
        return plane_1slice, grid_1d, X, Y
    
    def get_planes(self, grid_lb, grid_ub, num_grid_pts = 5, fixed_val = 0.5):
        """
        Generates multiple 2D slices of 3D planes for different 2D planes within a 3D space.
        
        Parameters:
        grid_lb (float): Lower bound of the grid.
        grid_ub (float): Upper bound of the grid.
        num_grid_pts (int, optional): Number of grid points in each dimension. Default is 5.
        fixed_val (float, optional): Fixed value for the plane's constant dimension. Default is 0.5.
        
        Returns:
        tuple: A tuple containing:
            - plane_3slices (numpy.ndarray, shape: (3 x 3 x num_grid_pts x num_grid_pts)): 
                    4D array where each slice corresponds to a 2D plane within the 3D space.
            - grid_1d (numpy.ndarray, shape: (num_grid_pts,)): 1D array of grid points used 
                for the X and Y dimensions.
            - X (numpy.ndarray, shape: num_grid_pts x num_grid_pts): 2D array of X coordinates 
                (from the last processed plane).
            - Y (numpy.ndarray, shape: num_grid_pts x num_grid_pts): 2D array of Y coordinates 
                (from the last processed plane).
        """
        
        # Initialize a 4D array to store slices for each 2D plane
        plane_3slices = np.full((self.nPlanes, self.nPlanes, num_grid_pts, 
                                 num_grid_pts), np.nan)
        # Iterate over each 2D plane identifier and generate its corresponding slice
        for i, plane_str in enumerate(self.plane_2D_list):
            plane_3slices[i], grid_1d, X, Y = self._get_plane_1slice(grid_lb,
                                                     grid_ub, 
                                                     num_grid_pts, 
                                                     fixed_val, 
                                                     plane_2D = plane_str)
        return plane_3slices, grid_1d, X, Y
    
    def convert_rgb_lab(self, color_RGB):
        """
        Convert an RGB color value into the CIELab color space using the monitor's 
        spectral power distribution (SPD), the background RGB values, cone sensitivities 
        (T_CONES), and a matrix that converts from LMS (cone responses) to CIEXYZ 
        color space (M_LMS_TO_XYZ).
    
        Parameters:
        - color_RGB (array; 3,): RGB color value(s) to be converted.
            where N is the number of selected wavelengths
        
        Returns:
        - color_Lab (array; 3,): The converted color(s) in CIELab color space, a 1D array.
        - color_XYZ (array; 3,): The intermediate CIEXYZ color space representation, a 1D array.
        - color_LMS (array; 3,): The LMS cone response representation, a 1D array.
    
        """
    
        # Convert background RGB to SPD using the monitor's SPD
        background_Spd = self.B_MONITOR @ self.background_rgb
        # Convert background SPD to LMS (cone response)
        background_LMS = self.T_CONES @ background_Spd
        # Convert background LMS to XYZ (for use in Lab conversion)
        background_XYZ = self.M_LMS_TO_XYZ @ background_LMS
        
        #RGB -> SPD
        color_Spd = self.B_MONITOR @ color_RGB
        #SPD -> LMS
        color_LMS = self.T_CONES @ color_Spd
        #LMS -> XYZ
        color_XYZ = self.M_LMS_TO_XYZ @ color_LMS
    
        #XYZ -> Lab
        background_XYZ_arr = np.array(background_XYZ)
        background_xyY = colour.XYZ_to_xyY(background_XYZ_arr)
    
        color_Lab = colour.XYZ_to_Lab(color_XYZ, background_xyY) 
        #print(color_Lab)
        
        return color_Lab, color_XYZ, color_LMS
    
    def compute_deltaE(self, ref_RGB, vecDir, vecLen):
        """
        Computes the perceptual difference (deltaE) between a reference stimulus
        and a comparison stimulus in the CIELab color space. The comparison stimulus
        is derived based on a specified chromatic direction and length from the reference.
    
        Parameters:
        - ref_RGB (array; 3,): The RGB values of the reference stimulus.
        - vecDir (array; 1 x 3): The direction vector along which the comparison 
            stimulus varies from the reference.
        - vecLen (array): The length to move in the specified direction from the 
            reference stimulus.
    
        Returns:
        - deltaE (float): The computed perceptual difference between the reference 
            and comparison stimuli.
        """
    
        #pdb.set_trace()
        # Calculate the RGB values for the comparison stimulus by adjusting the reference RGB
        # along the specified chromatic direction by the given vector length (vecLen).
        ref_Lab,_,_ = self.convert_rgb_lab(ref_RGB)
        comp_RGB = ref_RGB + vecDir * vecLen
        
        # Convert the computed RGB values of the comparison stimulus into Lab values
        # using the provided parameters and the background RGB. 
        comp_Lab,_,_ = self.convert_rgb_lab(comp_RGB)
        
        # Calculate the perceptual difference (deltaE) between the reference and comparison
        # stimuli as the Euclidean distance between their Lab values.
        deltaE = np.linalg.norm(comp_Lab - ref_Lab)
        
        return deltaE
    
    def find_vecLen(self, ref_RGB_test, vecDir_test, deltaE = 1, lb_opt = 0,
                    ub_opt = 0.1, N_opt = 3):
        """
        This function finds the optimal vector length for a chromatic direction
        that achieves a target perceptual difference in the CIELab color space.
    
        Parameters:
        - ref_RGB_test (array): The RGB values of the reference stimulus
        - vecDir_test (array): The chromatic direction vector for comparison stimulus variation
        - deltaE (float): The target deltaE value (e.g., 1 JND)
        - lb_gridsearch (float): the lower bounds for the search of the vector length
        - ub_gridsearch (float): the upper bounds for the search of the vector length
        - N_gridsearch (int): the number of runs for optimization to ensure we don't get stuck 
            at local minima
        
        Returns:
        - opt_vecLen (float): The optimal vector length that achieves the target deltaE value
        """
        #The lambda function computes the absolute difference between the
        #deltaE obtained from compute_deltaE function and the target deltaE.
        deltaE_func = lambda d: abs(self.compute_deltaE(ref_RGB_test, vecDir_test, d) - deltaE)
            
        # Generate initial points for the optimization algorithm within the bounds.
        init = np.random.rand(N_opt) * (ub_opt - lb_opt) + lb_opt
        # Set the options for the optimization algorithm.
        options = {'maxiter': 1e5, 'disp': False}
        # Initialize arrays to store the vector lengths and corresponding deltaE 
        #values for each run.
        vecLen_n = np.empty(N_opt)
        deltaE_n = np.empty(N_opt)
        
        # Loop over the number of runs to perform the optimizations.
        for n in range(N_opt):
            # Use scipy's minimize function to find the vector length that minimizes
            # the difference to the target deltaE. SLSQP method is used for 
            #constrained optimization.
            res = minimize(deltaE_func, init[n],method='SLSQP',
                           bounds=[(lb_opt, ub_opt)], options=options)
            # Store the result of each optimization run.
            vecLen_n[n] = res.x
            deltaE_n[n] = res.fun
            
        # Identify the index of the run that resulted in the minimum deltaE value.
        idx_min = np.argmin(deltaE_n)
        # Choose the optimal vector length from the run with the minimum deltaE value.
        opt_vecLen = vecLen_n[idx_min]
        
        return opt_vecLen
    
    #%%
    @staticmethod
    def set_chromatic_directions(num_dir_pts = 16):
        """
        Generates a set of chromatic directions in the 2D space (xy-plane).
        
        Parameters:
        num_dir_pts (int): The number of points (directions) to generate. Default is 16.
        
        Returns:
        numpy.ndarray: A 2xN array where each column represents a direction vector 
                       in the xy-plane. The first row contains the x-components, 
                       and the second row contains the y-components.
        """
        
        # Generate linearly spaced angles (in radians) between 0 and 2π. 
        grid_theta_temp = np.linspace(0, 2*np.pi, num_dir_pts + 1)
        # The extra point (2π) is included to close the loop, so we remove it.
        grid_theta      = grid_theta_temp[:-1]
        # compute x, y given the angles
        grid_theta_xy   = np.stack((np.cos(grid_theta),np.sin(grid_theta)),axis = 0)
        return grid_theta_xy
            

