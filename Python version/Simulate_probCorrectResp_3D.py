#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 14:30:18 2024

@author: fangfang
"""

import numpy as np
import pickle
import sys
import os
from dataclasses import replace
import dill as pickled
sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/Python version/')
from analysis.trial_placement import StimConfig, TrialPlacement_RGB_gridRef_gtCIE
from analysis.simulations_CIELab import SimThresCIELab
from plotting.trial_placement_nonadaptive_plotting import TrialPlacementVisualization,\
        PlotWeibullPMFSettings, Plot3DSampledCompSettings
from plotting.wishart_plotting import PlotSettingsBase
sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids")
from analysis.color_thres import color_thresholds

# Define base directories for saving figure and data files
base_dir = '/Volumes/T9/Aguirre-Brainard Lab Dropbox/Fangfang Hong'
output_figDir = os.path.join(base_dir, 'ELPS_analysis','Simulation_FigFiles',
                             'Python_version','3D') #transformation
output_fileDir = os.path.join(base_dir, 'ELPS_analysis','Simulation_DataFiles')

#%% ---------------------- Set up simulation parameters ----------------------
# Set the random seed for reproducibility
rnd_seed = 0    

# Choose the color difference algorithm used for computing ΔE
# Options: 'CIE1976', 'CIE1994', 'CIE2000'
colordiff_alg = 'CIE1994' 

# Initialize color_thresholds class to load ground-truth CIELab threshold data
color_thres_data = color_thresholds(3, base_dir)
color_thres_data.load_CIE_data(CIE_version=colordiff_alg)

#% ---------------------- Configure simulation settings ----------------------

# Define the stimulus configuration
# We're working in the full 3D RGB cube, with fixed reference locations
stim_config = StimConfig(
    fixed_plane='[]',  # No fixed color plane in 3D
    gt=colordiff_alg,  # Ground truth based on the selected ΔE method
    fixed_ref=True,    # Use a fixed reference for all trials
    M_RGBTo2DW=None,   # Not used in this simulation
    M_2DWToRGB=None,   # Not used in this simulation
    file_name=f'Isothreshold_ellipsoid_CIELABderived_{colordiff_alg}.pkl'  # File with GT thresholds
)

#%% ---------------------- Initialize simulation trials ----------------------

# Load ground-truth threshold data from file
with open(color_thres_data.file_path_CIE_data, 'rb') as f: 
    gt_CIE = pickle.load(f)

# Set up the simulation with the given configuration and loaded GT data
sim_trial = TrialPlacement_RGB_gridRef_gtCIE(
    gt_CIE, config=stim_config, random_seed=rnd_seed
)

# Define the psychometric function (Weibull) parameters:
# - alpha and beta determine the shape of the function
# - guessing_rate for 3AFC = 1/3
# - deltaE_1JND defines how ΔE values map to perceptual units
sim_trial.setup_WeibullFunc(
    alpha=3.189, 
    beta=1.505, 
    guessing_rate=1/3, 
    deltaE_1JND=2.5  # perceptual threshold unit
)

# Print out the target accuracy (probability correct) at ΔE = 1
print(f"target probability: {sim_trial.sim['pC_given_alpha_beta']}")

#% ---------------------- Visualize psychometric function ----------------------

# Create a visualization object to plot simulation results
output_figDir_CIE = os.path.join(output_figDir, colordiff_alg)
os.makedirs(output_figDir_CIE, exist_ok= True)
pltSettings_base = PlotSettingsBase(fig_dir= output_figDir_CIE, fontsize = 12)
sim_vis = TrialPlacementVisualization(
    sim_trial, settings=pltSettings_base, save_fig=True
)

# Customize plot settings for the psychometric (Weibull) function
pltSettings_PMF = replace(PlotWeibullPMFSettings(), **pltSettings_base.__dict__)

# Generate x-axis values and plot the PMF
x_PMF = np.linspace(0, 6, 100)
sim_vis.plot_WeibullPMF(x_PMF, settings=pltSettings_PMF)

#% ---------------------- Run simulation ----------------------

# Set up CIELab simulation environment with background color
background_RGB = np.array([0.5, 0.5, 0.5])
sim_CIELab = SimThresCIELab(background_RGB)

# Run the main simulation: generates comparison stimuli and responses
sim_trial.run_sim(sim_CIELab)

#%% visualization
plt3DSettings = replace(Plot3DSampledCompSettings(), **pltSettings_base.__dict__)
#% plotting and saving data
sim = sim_trial.sim
fixed_val = 0.8 #0.2, 0.35, 0.5, 0.65, 0.8
for fixed_dim in 'RGB':
    ttl = 'RGB plane'
    ttl_new = ttl.replace(fixed_dim,'')
    figName_i = f"Sims_isothreshold_ellipsoids_sim{sim['nSims']}perCond_"+\
        f"sampling{sim['method_sampling']}_jitter{sim['random_jitter']}_"+\
            f"{colordiff_alg}{ttl_new[:2]}_fixedVal{fixed_val}.pdf"
            
    #plot the transformation
    plt3DSettings = replace(plt3DSettings, 
                            bds= 0.12,
                            slc_grid_ref_dim1 = list(range(0,5,2)), #list(range(0,5,1))
                            slc_grid_ref_dim2 = list(range(0,5,2)),
                            title = ttl_new, 
                            fig_name = figName_i)

    sim_vis.plot_3D_sampledComp(sim_trial.gt_CIE_stim['grid_ref'], 
                                sim_trial.gt_CIE_results['fitEllipsoid_unscaled'],\
                                sim['comp'],
                                fixed_dim, fixedPlaneVal= fixed_val,
                                settings = plt3DSettings,
                                save_fig= True)
    
#%% save the data
str_colordiff_alg = '' if colordiff_alg == 'CIE1976' else f'_{colordiff_alg}'
file_name = f"Sims_isothreshold_ellipsoids_sim{sim['nSims']}" +\
    f"perCond_sampling{sim['method_sampling']}_jitter{sim['random_jitter']}_"+\
    f"seed{rnd_seed}{str_colordiff_alg}.pkl"
    
output_fileDir_CIE = os.path.join(output_fileDir, 'ellipsoids',colordiff_alg)
os.makedirs(output_fileDir_CIE, exist_ok=True)
full_path = os.path.join(output_fileDir_CIE, file_name)
                            
variable_names = ['sim_trial','color_thres_data','background_RGB', 'sim_CIELab']
vars_dict = {}
for var_name in variable_names:
    try:
        # Check if the variable exists in the global scope
        vars_dict[var_name] = eval(var_name)
    except NameError:
        # If the variable does not exist, assign None and print a message
        vars_dict[var_name] = None
        print(f"Variable '{var_name}' does not exist. Assigned as None.")

# Write the list of dictionaries to a file using pickle
with open(full_path, 'wb') as f:
    pickled.dump(vars_dict, f)                    
                    
                    
                    
                    
                
    