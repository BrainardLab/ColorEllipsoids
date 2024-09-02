#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 15:01:11 2024

@author: fangfang
"""

import os
import numpy as np
import pickle

#import functions from the other script
func_path = '/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/Python version/'
os.chdir(func_path)
from analysis.trial_placement import TrialPlacementWithoutAdaptiveSampling
from analysis.simulations_CIELab import SimThresCIELab
from plotting.trial_placement_nonadaptive_plotting import TrialPlacementVisualization

base_dir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/ELPS_analysis/'
#specify figure name and path
output_figDir = base_dir+'Simulation_FigFiles/'
output_fileDir = base_dir + 'Simulation_DataFiles/'

#%%
file_name = 'Isothreshold_contour_CIELABderived_fixedVal0.5.pkl'
full_path = f"{output_fileDir}{file_name}"
os.chdir(output_fileDir)

#Here is what we do if we want to load the data
with open(full_path, 'rb') as f:
    # Load the object from the file
    gt_CIE = pickle.load(f)
    
#%% Define dictionary sim
# Define the path to the directory containing the necessary files for the simulation
path_str2 = "/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/"+\
                "FilesFromPsychtoolbox/"
background_RGB = np.reshape(gt_CIE[1]['background_RGB'],(-1))
sim_CIELab = SimThresCIELab(path_str2, background_RGB)

sim_trial = TrialPlacementWithoutAdaptiveSampling(gt_CIE)

#%% visualize individual steps
# Use ellipsoidal parameters to generate comparison stimuli
row_eg = 4
col_eg = 0
ellPara_eg = sim_trial.gt_CIE_results['ellParams'][sim_trial.sim['slc_RGBplane']][row_eg, col_eg]
rgb_ref_eg = sim_trial.sim['ref_points'][sim_trial.sim['varying_RGBplane'],row_eg, col_eg]
rgb_comp_eg, rgb_comp_eg_1stepback, rgb_comp_eg_2stepback, rgb_comp_eg_3stepback =\
    sim_trial.sample_rgb_comp_2DNearContour(rgb_ref_eg, ellPara_eg, random_seed = 0)  

sim_trial_2D_vis = TrialPlacementVisualization(sim_trial,
                                               fig_dir= output_figDir, 
                                               save_fig=False)  
sim_trial_2D_vis.plot_transformation(rgb_comp_eg_3stepback,
                                     rgb_comp_eg_2stepback, 
                                     rgb_comp_eg_1stepback,
                                     rgb_comp_eg[sim_trial.sim['varying_RGBplane']])

#%%
# Define the Weibull psychometric function.
# Calculate the probability of correct response given alpha and beta.
sim_trial.setup_WeibullFunc(alpha = 1.17, beta = 2.33, guessing_rate = 1/3)
print(f"target probability: {sim_trial.sim['pC_given_alpha_beta']}")

rnd_seed = 1
sim_trial.run_sim(sim_CIELab, random_seed = rnd_seed)

# visualize the samples and save the data
sim_trial.plot_2D_sampledComp(sim_trial.sim['rgb_comp'], 
                              resp = sim_trial.sim['resp_binary'])  

#%%save to pkl
file_name_firsthalf = f"Sims_isothreshold_{sim_trial.sim['plane_2D']}_sim"+\
    f"{sim_trial.sim['nSims']}_perCond_sampling_{sim_trial.sim['method_sampling']}"
if sim_trial.sim['method_sampling'] == 'NearContour':
    file_name_secondhalf = f"_jitter{sim_trial.sim['random_jitter']}_seed{rnd_seed}.pkl"
elif sim_trial.sim['method_sampling'] == 'Random':
    file_name_secondhalf = f"_range{sim_trial.sim['range_randomSampling']}_seed{rnd_seed}.pkl"        
file_name = file_name_firsthalf + file_name_secondhalf
full_path = f"{output_fileDir}{file_name}"
    
sim = sim_trial.sim
# Write the list of dictionaries to a file using pickle
with open(full_path, 'wb') as f:
    pickle.dump([sim], f)


        
        
    
    
    
