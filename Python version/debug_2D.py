#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 13:48:55 2024

@author: fangfang
"""

#%% import modules
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
import pickle
import sys
import numpy as np

sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids")
from core import optim, model_predictions, oddity_task
from core.wishart_process import WishartProcessModel

#three variables we need to define for loading the data
plane_2D      = 'RB plane'
sim_jitter    = '0.1'
BANDWIDTH     = 5e-3
nSims         = 240 #number of simulations: 240 trials for each ref stimulus
file_sim      = 'Sims_isothreshold_'+plane_2D+'_sim'+str(nSims)+'perCond_'+\
                'samplingNearContour_jitter'+sim_jitter+'.pkl'
outputDir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                        'ELPS_analysis/ModelFitting_DataFiles/'
output_file = 'Fitted'+file_sim[4:-4]+'_bandwidth' + str(BANDWIDTH) + '.pkl'
full_path4 = f"{outputDir}{output_file}"

# Write the list of dictionaries to a file using pickle
with open(full_path4, 'rb') as f:
    # Load the object from the file
    data_load = pickle.load(f)
for key, value in data_load.items():
    locals()[key] = value

#%%
# -----------------------------------------------------------
# Load data simulated using CIELab and organize data
# -----------------------------------------------------------
#file 1
path_str      = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                'ELPS_analysis/Simulation_DataFiles/'
os.chdir(path_str)
file_CIE      = 'Isothreshold_contour_CIELABderived_fixedVal0.5.pkl'
full_path     = f"{path_str}{file_CIE}"
with open(full_path, 'rb') as f: data_load = pickle.load(f)
stim          = data_load[1]
results       = data_load[2]

#file 2
plane_2D_dict = {'GB plane': 0, 'RB plane': 1, 'RG plane': 2}
plane_2D_idx  = plane_2D_dict[plane_2D]
file_sim      = 'Sims_isothreshold_'+plane_2D+'_sim'+str(nSims)+'perCond_'+\
                'samplingNearContour_jitter'+sim_jitter+'.pkl'
full_path     = f"{path_str}{file_sim}"   
with open(full_path, 'rb') as f:  data_load = pickle.load(f)
sim = data_load[0]
data, x1_raw, xref_raw = model_predictions.organize_data(sim,\
        5, visualize_samples = False, plane_2D = plane_2D)
ref_size_dim1, ref_size_dim2 = x1_raw.shape[0:2]
y_jnp, xref_jnp, x0_jnp, x1_jnp = data 

#%%
# -----------------------------
# Compute model predictions
# -----------------------------
ngrid_search            = 250
bds_scaler_gridsearch   = [0.5, 3]
nTheta                  = 200
recover_fitEllipse_scaled, recover_fitEllipse_unscaled,\
    recover_rgb_comp_scaled, recover_rgb_contour_cov, params_ellipses =\
    model_predictions.convert_Sig_2DisothresholdContour_oddity_batch(xref_raw,\
    sim['varying_RGBplane'], stim['grid_theta_xy'], sim['pC_given_alpha_beta'],\
    W_est, model, results['opt_vecLen'], ngrid_bruteforce = ngrid_search,\
    scaler_bds_bruteforce = bds_scaler_gridsearch, scaler_x1 = 5,\
    nThetaEllipse = nTheta, mc_samples = MC_SAMPLES,bandwidth = BANDWIDTH,\
    opt_key = OPT_KEY)



