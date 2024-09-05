#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 15:01:11 2024

@author: fangfang
"""

import sys
import numpy as np
import pickle

#import functions from the other script
func_path = '/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/Python version/'
sys.path.append(func_path)
from analysis.trial_placement import TrialPlacementWithoutAdaptiveSampling
from analysis.simulations_CIELab import SimThresCIELab
from plotting.trial_placement_nonadaptive_plotting import TrialPlacementVisualization
sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids")
from analysis.ellipses_tools import PointsOnEllipseQ, UnitCircleGenerate

# Define base directories for saving figure and data files
base_dir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/ELPS_analysis/'
output_figDir = base_dir+'Simulation_FigFiles/Python_version/transformation'
output_fileDir = base_dir + 'Simulation_DataFiles/'

#%% Load precomputed ground truth data from a pickle file
file_name = 'Isothreshold_contour_CIELABderived_fixedVal0.5.pkl'
full_path = f"{output_fileDir}{file_name}"
#os.chdir(output_fileDir)

#Here is what we do if we want to load the data
with open(full_path, 'rb') as f:
    # Load the object from the file
    gt_CIE = pickle.load(f)
    
#%% 
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

#%% Define the Weibull psychometric function with specified parameters
# Calculate the probability of correct response given alpha and beta.
sim_trial.setup_WeibullFunc(alpha = 1.17, beta = 2.33, guessing_rate = 1/3)
# Print the target probability based on the Weibull function for the given delta E
print(f"target probability: {sim_trial.sim['pC_given_alpha_beta']}")

# Initialize the visualization object for plotting simulation results
sim_trial_2D_vis = TrialPlacementVisualization(sim_trial,
                                               fig_dir= output_figDir, 
                                               save_fig=False) 
# Visualize the Weibull psychometric function
x_PMF = np.linspace(0,3,100)
sim_trial_2D_vis.plot_WeibullPMF(x_PMF)

#%% specify the seed
rnd_seed = 9
# Run the simulation with the specified random seed
sim_trial.run_sim(sim_CIELab, random_seed = rnd_seed)

#%% Visualize the sampled data from the simulation
sim_trial_2D_vis.plot_2D_sampledComp()  

#%% If the sampling method is 'NearContour', visualize the entire transformation process
if sim_trial.sim['method_sampling'] == 'NearContour':
    # Select a reference stimulus location (e.g., row 4, column 4)
    row_eg = 4
    col_eg = 4
    # Retrieve parameters for the ellipsoid at the selected location
    ellPara_eg = sim_trial.gt_CIE_results['ellParams'][sim_trial.sim['slc_RGBplane']][row_eg, col_eg]
    # Extract the reference RGB values at the selected location
    rgb_ref_eg = sim_trial.sim['ref_points'][sim_trial.sim['varying_RGBplane'],row_eg, col_eg]
    # Retrieve the ground truth ellipses during the transformation process
    rgb_comp_eg, rgb_comp_eg_1stepback, rgb_comp_eg_2stepback, rgb_comp_eg_3stepback =\
        sim_trial.sample_rgb_comp_2DNearContour(rgb_ref_eg, ellPara_eg, random_seed = rnd_seed) 
        
    # Compute the ground truth for each step of the transformation process
    nTheta = 200
    # initial form: unit circle
    gt_comp_eg_3stepback = UnitCircleGenerate(nTheta)
    # ground truth of the 2nd form: still a unit circle
    gt_comp_eg_2stepback = gt_comp_eg_3stepback
    # ground truth of the 3rd form: stretched
    gt_comp_eg_1stepback = PointsOnEllipseQ(ellPara_eg[2], ellPara_eg[3],  0, 0, 0)
    # ground truth of the 4th form: rotated and relocated
    gt_comp_eg = PointsOnEllipseQ(ellPara_eg[2], ellPara_eg[3], ellPara_eg[4], ellPara_eg[0], ellPara_eg[1])
    
    #define figure name
    fig_name_firsthalf = f"Sims_transformation_{sim_trial.sim['plane_2D']}"+\
        f"_dim1_{rgb_ref_eg[0]:.2f}_dim2_{rgb_ref_eg[1]:.2f}" +\
        f"_sim{sim_trial.sim['nSims']}_perCond_sampling_{sim_trial.sim['method_sampling']}"
    fig_name_secondhalf = f"_jitter{sim_trial.sim['random_jitter']}_seed{rnd_seed}"
        
    #plot the transformation
    sim_trial_2D_vis.plot_transformation(rgb_comp_eg_3stepback,
                                         rgb_comp_eg_2stepback, 
                                         rgb_comp_eg_1stepback,
                                         rgb_comp_eg[sim_trial.sim['varying_RGBplane']],
                                         resp = sim_trial.sim['resp_binary'][row_eg, col_eg],
                                         gt = [gt_comp_eg_3stepback,
                                               gt_comp_eg_2stepback,
                                               gt_comp_eg_1stepback,
                                               gt_comp_eg],
                                         colorcode_resp = True,
                                         fig_name = fig_name_firsthalf + fig_name_secondhalf +'.pdf')

                                         # xlim = [[-1.24473268,  1.25473268], [-1.50095096,  1.51095096],\
                                         #         [-0.03224849,  0.03247935], [ 0.77258853,  0.82782443]],
                                         # ylim = [[-1.24473268,  1.25473268], [-1.50095096,  1.51095096],\
                                         #         [-0.03224849,  0.03247935], [ 0.77258853,  0.82782443]],


#%%save to pkl
file_name_firsthalf = f"Sims_isothreshold_{sim_trial.sim['plane_2D']}_sim"+\
    f"{sim_trial.sim['nSims']}perCond_sampling{sim_trial.sim['method_sampling']}"
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


        
        
    
    
    
