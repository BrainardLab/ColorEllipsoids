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

nSims = 240
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

grid_theta            = stim3D['grid_theta'] #from 0 to 2*pi
n_theta               = len(grid_theta)
n_theta_finergrid     = plt_specifics['nThetaEllipsoid']
grid_phi              = stim3D['grid_phi'] #from 0 to pi
n_phi                 = len(grid_phi)
n_phi_finergrid       = plt_specifics['nPhiEllipsoid']
nSteps_bruteforce     = 100 #number of grids
bds_scaler_gridsearch = [0.5, 3]
pC_threshold          = 0.78            

recover_fitEllipsoid_scaled, recover_fitEllipsoid_unscaled,\
    recover_rgb_comp_scaled, recover_rgb_contour_cov,\
    params_ellipsoids = model_predictions.convert_Sig_3DisothresholdContour_oddity_batch(\
        xref_raw, stim3D['grid_xyz'], pC_threshold, W_est, model,\
        results3D['opt_vecLen'], scaler_x1 = scaler_x1,\
        ngrid_bruteforce=nSteps_bruteforce,\
        scaler_bds_bruteforce = bds_scaler_gridsearch,\
        bandwidth = opt_params['bandwidth'], opt_key = OPT_KEY,search_method='minimize')
        
        




