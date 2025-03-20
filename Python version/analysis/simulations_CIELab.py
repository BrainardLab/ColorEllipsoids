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
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000, delta_e_cie1994, delta_e_cie1976

#in order for delta_e_cie2000 to work, we need to do the following adjustment
def patch_asscalar(a):
    return a.item()
setattr(np, "asscalar", patch_asscalar)

#%%
class SimThresCIELab:
    def __init__(self, fileDir, background_rgb, plane_2D_list = ['GB plane', 'RB plane', 'RG plane']):
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
        # Validate plane_2D_list
        self._validate_plane_list(plane_2D_list)
        self.plane_2D_list  = plane_2D_list
        self.nPlanes        = len(self.plane_2D_list)
        
        #Note that if plane_2D_list is 'Isoluminant plane', some of the following methods are not applicable
        #come back to this class in the future and refine the code to be more generalizable
        self.plane_2D_dict  = dict(zip(self.plane_2D_list, list(range(self.nPlanes))))
        if self.nPlanes == 3:
            self.varying_dims = [[1,2],[0,2],[0,1]]
        elif self.nPlanes == 1:
            self.varying_dims = [[0,1]] #treat it as RG plane with the third dimension fixed at 1
            
    def _validate_plane_list(self, plane_2D_list):
        """Internal method to validate plane_2D_list."""
        valid_plane_options = [['GB plane', 'RB plane', 'RG plane'], ['Isoluminant plane']]
        if plane_2D_list not in valid_plane_options:
            raise ValueError(f"Invalid plane_2D_list: {plane_2D_list}. Must be one of {valid_plane_options}.")
            
    #%% methods
    def get_plane_1slice(self, grid_lb, grid_ub, num_grid_pts,fixed_val, plane_2D):
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
            plane_3slices[i], grid_1d, X, Y = self.get_plane_1slice(grid_lb,
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
        background_xyY = colour.XYZ_to_xyY(background_XYZ)
    
        color_Lab = colour.XYZ_to_Lab(color_XYZ, background_xyY) 
        #print(color_Lab)
        
        return color_Lab, color_XYZ, color_LMS
    
    def compute_deltaE(self, ref_RGB, vecDir, vecLen, comp_RGB=None, method='CIE1976'):
        """
        Computes the perceptual difference (deltaE) between a reference stimulus
        and a comparison stimulus in the CIELab color space. The comparison stimulus
        can either be specified directly or calculated based on a chromatic direction
        and distance from the reference stimulus.
    
        Parameters:
        - ref_RGB (array; 3,): RGB values of the reference stimulus (source color).
        - vecDir (array; 1 x 3): Chromatic direction vector defining how the comparison 
          stimulus varies from the reference stimulus in RGB space.
        - vecLen (float): Magnitude of the variation along the chromatic direction `vecDir`.
        - comp_RGB (array; 3, optional): RGB values of the comparison stimulus. If not provided,
          it will be calculated by applying the `vecDir` and `vecLen` to the `ref_RGB`.
        - method (str): The method for calculating deltaE. Options are:
            - 'CIE1976': DeltaE using the CIE1976 method (Euclidean distance in CIELab).
            - 'CIE1994': DeltaE using the CIE1994 method (accounts for perceptual non-uniformity).
            - 'CIE2000': DeltaE using the CIE2000 method (more advanced perceptual uniformity).
    
        Returns:
        - deltaE (float): The computed perceptual difference between the reference and comparison stimuli.
    
        Notes:
        - If an invalid method is specified, the function will issue a warning and default to
          the Euclidean distance in CIELab ('Euclidean').
    
        """
    
        # Convert reference RGB to CIELab values.
        ref_Lab, _, _ = self.convert_rgb_lab(ref_RGB)
    
        # If comparison RGB is not provided, calculate it by moving ref_RGB along vecDir by vecLen.
        if comp_RGB is None:
            comp_RGB = ref_RGB + vecDir * vecLen
    
        # Convert comparison RGB to CIELab values.
        comp_Lab, _, _ = self.convert_rgb_lab(comp_RGB)
        
        # Define reference and comparison colors in LabColor format.
        color1 = LabColor(lab_l=ref_Lab[0], lab_a=ref_Lab[1], lab_b=ref_Lab[2])
        color2 = LabColor(lab_l=comp_Lab[0], lab_a=comp_Lab[1], lab_b=comp_Lab[2])
    
        # Compute deltaE using the specified method.
        if method == 'CIE2000':
            # CIE2000 method (more accurate perceptual uniformity).
            deltaE = delta_e_cie2000(color1, color2)
        elif method == 'CIE1994':
            # CIE1994 method (intermediate between CIE1976 and CIE2000).
            deltaE = delta_e_cie1994(color1, color2)
        else:
            # Simple Euclidean distance in CIELab.
            deltaE = np.linalg.norm(comp_Lab - ref_Lab)
            # THIS IS EQUIVALENT AS CIE1976
            
            # CIE1976 method (Euclidean distance in CIELab).
            #deltaE = delta_e_cie1976(color1, color2)
    
        return deltaE    
    
    def find_vecLen(self, ref_RGB_test, vecDir_test, deltaE = 1, lb_opt = 0,
                    ub_opt = 0.1, N_opt = 3, coloralg = 'CIE1976'):
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
        
        if coloralg not in ['CIE1976', 'CIE1994', 'CIE2000']:
            raise ValueError("The method can only be 'CIE1976' or 'CIE1994' or 'CIE2000'.")
                
        #The lambda function computes the absolute difference between the
        #deltaE obtained from compute_deltaE function and the target deltaE.
        deltaE_func = lambda d: abs(self.compute_deltaE(ref_RGB_test,
                                                        vecDir_test, 
                                                        d, method=coloralg) - deltaE)
            
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
    
    def find_threshold_point_on_isoluminant_plane(self, W_ref, chrom_dir, M_RGBTo2DW, 
                                                  M_2DWToRGB, deltaE = 1, 
                                                  color_diff_algorithm = 'CIE2000'):
        # Step 1: Define a chromatic direction in W-space and convert to RGB
        chrom_dir_W = chrom_dir + W_ref[:2]  # Shifted chromatic direction
        chrom_dir_rgb = M_2DWToRGB @ np.append(chrom_dir_W, 1)  # Convert back to RGB

        # Step 2: Compute normalized direction vector in RGB space
        rgb_ref = M_2DWToRGB @ W_ref
        rgb_vecDir_temp = chrom_dir_rgb - rgb_ref  # Vector difference
        rgb_vecDir = rgb_vecDir_temp / np.linalg.norm(rgb_vecDir_temp)  # Normalize

        # Step 3: Find vector length that produces ΔE = 1 in CIELab space
        opt_vecLen = self.find_vecLen(
            rgb_ref, rgb_vecDir, deltaE = deltaE, coloralg=color_diff_algorithm
        )

        # Step 4: Compute threshold point in RGB space
        rgb_comp_threshold = opt_vecLen * rgb_vecDir + rgb_ref
        
        # Step 5: Transform Threshold Points from RGB → W 
        W_comp_temp = M_RGBTo2DW @ rgb_comp_threshold  # Convert to W-space
        W_comp = W_comp_temp / W_comp_temp[-1]  # Normalize last row to 1
        W_comp_threshold = W_comp[:2]
        
        return rgb_vecDir, opt_vecLen, rgb_comp_threshold, W_comp_threshold
    
    
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
            
    