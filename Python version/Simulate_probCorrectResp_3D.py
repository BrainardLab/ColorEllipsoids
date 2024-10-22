#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 14:30:18 2024

@author: fangfang
"""

import numpy as np
import pickle
import sys

#import functions from the other script
func_path = '/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/Python version/'
sys.path.append(func_path)
from analysis.trial_placement import TrialPlacementWithoutAdaptiveSampling
from analysis.simulations_CIELab import SimThresCIELab
from plotting.trial_placement_nonadaptive_plotting import TrialPlacementVisualization

# Define base directories for saving figure and data files
base_dir = '/Volumes/T9/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'
output_figDir = base_dir+'ELPS_analysis/Simulation_FigFiles/Python_version/transformation'
output_fileDir = base_dir + 'ELPS_analysis/Simulation_DataFiles/'

#%%
# specify the seed
rnd_seed = 1
file_name = 'Isothreshold_ellipsoid_CIELABderived.pkl'
path_str =  base_dir + 'ELPS_analysis/Simulation_DataFiles/'
full_path = f"{path_str}{file_name}"

#Here is what we do if we want to load the data
with open(full_path, 'rb') as f:
    # Load the object from the file
    gt_CIE = pickle.load(f)

# Initialize the SimThresCIELab object with a path to necessary files and background RGB value
path_str2 = "/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/"+\
                "FilesFromPsychtoolbox/"
background_RGB = np.reshape(gt_CIE[1]['background_RGB'],(-1))
sim_CIELab = SimThresCIELab(path_str2, background_RGB)

# Set up the simulation object
# QUESTION 1: Ask the user which RGB plane to fix during the simulation
# QUESTION 2: Ask the user to choose the sampling method
# QUESTION 3: For 'NearContour', ask for the jitter variability
# QUESTION 4: Ask how many simulation trials
sim_trial = TrialPlacementWithoutAdaptiveSampling(gt_CIE)

#% Define the Weibull psychometric function with specified parameters
# Calculate the probability of correct response given alpha and beta.
sim_trial.setup_WeibullFunc(alpha = 1.17, beta = 2.33, guessing_rate = 1/3)
# Print the target probability based on the Weibull function for the given delta E
print(f"target probability: {sim_trial.sim['pC_given_alpha_beta']}")

# Initialize the visualization object for plotting simulation results
sim_trial_vis = TrialPlacementVisualization(sim_trial,
                                               fig_dir= output_figDir, 
                                               save_fig=False) 

# Visualize the Weibull psychometric function
x_PMF = np.linspace(0,3,100)
sim_trial_vis.plot_WeibullPMF(x_PMF)

#% specify the seed
# Run the simulation with the specified random seed
sim_trial.run_sim(sim_CIELab, random_seed = rnd_seed)

#% plotting and saving data
sim = sim_trial.sim
for test in 'RGB':
    ttl = 'RGB plane'
    ttl_new = ttl.replace(test,'')
    sim_trial_vis.plot_3D_sampledComp(sim_trial.gt_CIE_stim['grid_ref'], 
                                        sim_trial.gt_CIE_results['fitEllipsoid_unscaled'],\
                                        sim['rgb_comp'],
                                        test, 0.5,
                                        slc_grid_ref_dim1 = [0,2,4],
                                        slc_grid_ref_dim2 = [0,2,4],
                                        title = ttl_new)
    
#save to pkl
file_name = f"Sims_isothreshold_ellipsoids_sim{sim['nSims']}" +\
    f"perCond_sampling{sim['method_sampling']}_jitter{sim['random_jitter']}_seed{rnd_seed}.pkl"
full_path = f"{output_fileDir}{file_name}"
        
# Write the list of dictionaries to a file using pickle
with open(full_path, 'wb') as f:
    pickle.dump([sim], f)
                    
                    
                    
                    
                    
                    
                    
                
    