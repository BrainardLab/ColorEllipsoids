#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 20:48:33 2024

@author: fangfang
"""

import jax
jax.config.update("jax_enable_x64", True)
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids')
from analysis.ellipses_tools import ellParamsQ_to_covMat
from core.probability_surface import pC_Weibull_one_ref_stim
from matplotlib import cm
# Define base directory and figure output directory
base_dir = '/Volumes/T9/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'
figDir_fits = base_dir+'ELPS_analysis/ModelComparison_FigFiles/2D_oddity_task/'

#%% Simulate a surface of probability
# Specify which ellipse condition to use
ell_slc_str = 'stretch_rotate'
ellP_slc = {'unit': [1,1,90], #unit circle
            'stretch': [np.sqrt(1.5),np.sqrt(0.5),90], #streth it by semi-axis length
            'stretch_rotate': [np.sqrt(1.5),np.sqrt(0.5),45]} #rotate it by a rotation matrix
# Flag for saving the figure
saveFig = False
# Parameters for the shape of the Weibull PM function
weibull_params = [1.17, 2.33]
# Reference stimulus location (x-coordinate)
xref = 0.5

# Convert ellipse parameters to covariance matrix
covMat_slc = ellParamsQ_to_covMat(*ellP_slc[ell_slc_str])
# Define the bounds for the probability surface grid
xbds = np.array([-3, 3]) + xref  # Boundary of grid along both x and y axes
xrange = np.array([-6, 6]) + xref  # Extended range for x and y coordinates
xticks = np.array([-1.5, 0.5, 2.5])  # Major tick marks for axes

# Number of grid points for surface
nx = 1000
# Set up a 2D grid (x, y) for plotting
x = np.linspace(*xrange, nx)
y = np.linspace(*xrange, nx)
X, Y = np.meshgrid(x, y)

# Stack (x, y) points into a 2D array where each row is a point (x_i, y_i)
xy = np.stack((X.flatten(), Y.flatten()), axis=-1)  # Shape: (nx * nx, 2)

# Apply the covariance matrix: matrix multiplication (each row [x_i, y_i] transformed)
xy_transform = xy @ covMat_slc.T + xref  # Shape: (nx * nx, 2)

# Reshape the transformed x and y values back to meshgrid format
X = xy_transform[:, 0].reshape(nx, nx)
Y = xy_transform[:, 1].reshape(nx, nx)

# Flatten the transformed grid and pass it to the Weibull function
Z_trans_flat = pC_Weibull_one_ref_stim(weibull_params, xy)

# Reshape the result into the grid form
Z = np.reshape(Z_trans_flat, (nx, nx))

# Create a mask for out-of-bound values
mask = (X >= xbds[0]) & (X <= xbds[1]) & (Y >= xbds[0]) & (Y <= xbds[1]) 

# Apply the mask: set values outside the bounds to NaN
X = np.where(mask, X, np.nan)
Y = np.where(mask, Y, np.nan)
Z = np.where(mask, Z, np.nan)

# Adjust x-axis tick labels if the ellipse is not a unit circle
if ell_slc_str != 'unit':
    # Scale tick labels to match simulated CIE data
    xticklabels = (xticks - xref)*0.03 + xref
else:
    xticklabels = xticks


#%%
# Plot parameters
vmin = 1/3  # Minimum value for the probability surface
vmax = 1.02  # Maximum value for the probability surface
n_levels = 100  # Number of color levels in the colormap
# Create discrete colormap with specific number of levels
cmap = cm.get_cmap('gist_earth', n_levels)  # Get the colormap

# Create a 3D plot
fig = plt.figure(figsize = (4.5,4.5), dpi = 256)

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams.update({'font.size': 10})
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(X, Y, Z, cmap=cmap, vmin = vmin, vmax = 1.02)
# Add projection (contour) on the bottom (z=minimum of z values)
surface = ax.contourf(X, Y, Z, zdir='z', offset=0, cmap=cmap, 
            vmin = vmin, vmax = 0.97)
ax.set_zlim([0,1])
ax.set_xlim(xbds)
ax.set_ylim(xbds)
ax.set_xticks(xticks)
ax.set_yticks(xticks)
ax.set_xticklabels(np.around(xticklabels,2))
ax.set_yticklabels(np.around(xticklabels,2))
ax.set_zticks([0.333, 0.667, 1])
# Customize plot appearance
ax.set_title(ell_slc_str)
ax.set_xlabel('Color dim1')
ax.set_ylabel('Color dim2')
ax.set_zlabel('Probability of correct')
# Add horizontal color bar at the bottom
cbar = fig.colorbar(surface, ax=ax, orientation='horizontal', 
                    pad=0.1, shrink=0.4, aspect=30)

# Set equal scaling for all axes
ax.set_box_aspect([1, 1, 1])  # Makes x, y, z axes the same length
# Set the viewing angle (elevation and azimuth)
ax.view_init(elev=30, azim=-70)  # Change this to set the desired viewing angle
plt.subplots_adjust(left=-0.1, right=1, top=0.8, bottom=0.05)
#plt.tight_layout()
figName1 = f"IndvModel_probability_surface_{ell_slc_str}.pdf"
full_path1 = os.path.join(figDir_fits, figName1)
if saveFig: fig.savefig(full_path1)   
# Show the plot
plt.show()


