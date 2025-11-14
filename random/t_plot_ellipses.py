#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This tutorial shows you how to plot threshold ellipses on a grid of 
reference stimuli

"""

import dill as pickled
import numpy as np
import math
import matplotlib.pyplot as plt
import os

#%%
#---------------------------------------------------------------------------
# SECTION 1: load the model fits 
# --------------------------------------------------------------------------
base_dir = "/Volumes/T9/Aguirre-Brainard Lab Dropbox/Fangfang Hong/"+\
    "ELPS_analysis/Experiment_DataFiles/pilot2/sub1/fits"
file_name = "Fitted_ColorDiscrimination_4dExpt_Isoluminant plane_sub1"+\
    "_decayRate0.5_nBasisDeg5.pkl"

# Construct the full path to the selected file
full_path = os.path.join(base_dir, file_name)

# Load the necessary variables from the file
with open(full_path, 'rb') as f:
    vars_dict = pickled.load(f)

# - Transformation matrices for converting between RGB, and W spaces
color_thres_data = vars_dict['color_thres_data']
print('Transformation matrix from model space to RGB:')
print(np.around(color_thres_data.M_2DWToRGB,3))

#load model predictions
model_pred = vars_dict['model_pred_Wishart']
grid = vars_dict['grid']

#ellipse parameters
fitEll = model_pred.fitEll_unscaled
paramsEll = model_pred.params_ell

#%% 
#---------------------------------------------------------------------------
# SECTION 2: plot using x, y coordinates
# --------------------------------------------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(4,4), dpi=1024)
for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        #we need the rgb to color code the ellipse
        cmap_ij = color_thres_data.M_2DWToRGB @ np.append(grid[i,j], 1)
        #ax.scatter(*grid[i,j], color = cmap_ij)
        ax.plot(*fitEll[i,j], color = cmap_ij)
ax.set_xlabel('Model space dimension 1')
ax.set_ylabel('Model space dimension 2')
ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_xticks(np.linspace(-0.7, 0.7, 5))
ax.set_yticks(np.linspace(-0.7, 0.7, 5))
ax.set_title('Isoluminant plane')
ax.set_aspect('equal')
ax.grid(True, alpha = 0.1)
plt.show()

#%% 
#---------------------------------------------------------------------------
# SECTION 3: Alternatively, plot using parameters that characterize ellipses
# --------------------------------------------------------------------------
def UnitCircleGenerate(nTheta):
    """
    Generate a set of points on the unit circle in two dimensions.
    nTheta - number of samples around the azimuthal theta (0 to 2pi)

    Coordinates are returned in an 2 by (nTheta) matrix, with the rows
    being the x, y coordinates of the points.
    """
    
    #generate a unit sphere in 3D
    theta = np.linspace(0,2*math.pi,nTheta)
    xCoords = np.cos(theta)
    yCoords = np.sin(theta)
    
    #stuff the coordinates into a single nTheta 
    x = np.stack((xCoords, yCoords), axis = 0)
    
    return x

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

#%% 
fig1, ax1 = plt.subplots(1, 1, figsize=(4,4), dpi=1024)
for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        #we need the rgb to color code the ellipse
        cmap_ij = color_thres_data.M_2DWToRGB @ np.append(grid[i,j], 1)
        
        #construct ellipses based on the 5 parameters 
        #(centroid_x, centroid_y, major, minor axis and rotation angle)
        paramsEll_ij = paramsEll[i][j]
        
        #extract those parameters
        x0_ij, y0_ij, a_ij, b_ij, rotAng_ij = paramsEll_ij
        
        #convert them to x, y coordinates
        fitEll_x_ij, fitELL_y_ij = PointsOnEllipseQ(a_ij, b_ij, rotAng_ij, x0_ij, y0_ij)
        
        #plot it
        ax1.plot(fitEll_x_ij, fitELL_y_ij, color = cmap_ij)
ax1.set_xlabel('Model space dimension 1')
ax1.set_ylabel('Model space dimension 2')
ax1.set_xlim([-1,1])
ax1.set_ylim([-1,1])
ax1.set_xticks(np.linspace(-0.7, 0.7, 5))
ax1.set_yticks(np.linspace(-0.7, 0.7, 5))
ax1.set_title('Isoluminant plane')
ax1.set_aspect('equal')
ax1.grid(True, alpha = 0.1)
plt.show()