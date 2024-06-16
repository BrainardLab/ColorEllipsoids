#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 23:09:16 2024

@author: fangfang
"""

#%%
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import warnings
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import Axes3D
import imageio.v2 as imageio
import os
import pickle
import numpy as np

import sys
sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids')
from core import viz, utils, oddity_task, model_predictions, optim
from core.wishart_process import WishartProcessModel
sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/Python version')
from Simulate_probCorrectResp_3D import plot_3D_sampledComp
from Isothreshold_ellipsoids_CIELab import fit_3d_isothreshold_ellipsoid

nSims = 640
jitter = 0.1
file_name = 'Sims_isothreshold_ellipsoids_sim'+str(nSims)+\
            'perCond_samplingNearContour_jitter'+str(jitter)+'.pkl'
outputDir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                        'ELPS_analysis/ModelFitting_DataFiles/'
output_file = 'Fitted'+file_name[4:-4]+'_bandwidth' + str(5e-3) + '.pkl'
full_path4 = f"{outputDir}{output_file}"

# Write the list of dictionaries to a file using pickle
with open(full_path4, 'rb') as f:
    # Load the object from the file
    data_load = pickle.load(f)
for key, value in data_load.items():
    locals()[key] = value
ref_size_dim1, ref_size_dim2, ref_size_dim3 = x1_raw.shape[0:3]
y_jnp, xref_jnp, x0_jnp, x1_jnp = data 

#%%
#file 2
path_str = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
        'ELPS_analysis/Simulation_DataFiles/'
file_name3 = 'Isothreshold_ellipsoid_CIELABderived.pkl'
full_path3 = f"{path_str}{file_name3}"
os.chdir(path_str)

#Here is what we do if we want to load the data
with open(full_path3, 'rb') as f:
    # Load the object from the file
    data_load1 = pickle.load(f)
param3D, stim3D, results3D, plt_specifics = data_load1[0], data_load1[1], data_load1[2], data_load1[3]

#%%
model_predictions.plot_3D_modelPredictions_byWishart(xref_raw, x1_raw,\
        xref_jnp, x1_jnp, np.transpose(xgrid,(1,0,2,3)), gt_covMat, Sigmas_est_grid,\
        recover_fitEllipsoid_scaled, gt_slice_2d_ellipse, pred_slice_2d_ellipse,\
        samples_alpha = 0.3)




