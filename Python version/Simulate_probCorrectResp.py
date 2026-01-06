#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 15:01:11 2024

@author: fangfang
"""

import sys
import os
import numpy as np
import pickle
import dill as pickled
from dataclasses import replace
sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids")
from analysis.ellipses_tools import PointsOnEllipseQ, UnitCircleGenerate
from analysis.simulations_CIELab import SimThresCIELab
from analysis.trial_placement import StimConfig_RGBslices, StimConfig_isoluminant,\
    TrialPlacement_RGB_gridRef_gtCIE
from analysis.color_thres import color_thresholds
from plotting.wishart_plotting import PlotSettingsBase
from plotting.trial_placement_nonadaptive_plotting import TrialPlacementVisualization,\
        Plot2DSampledCompSettings, PlotWeibullPMFSettings, PlotTransformationSettings

# Define base directories for saving figure and data files
base_dir = '/Volumes/T9/Aguirre-Brainard Lab Dropbox/Fangfang Hong/ELPS_analysis/'
output_figDir = os.path.join(base_dir, 'Simulation_FigFiles','Python_version',
                             'Isoluminant plane')
output_fileDir = os.path.join(base_dir, 'Simulation_DataFiles', 'Isoluminant plane')
pltSettings_base = PlotSettingsBase(fig_dir=output_figDir, fontsize = 8)

#%% 
# -----------------------------------------------------------
# Set up color-threshold object + stimulus configuration
# -----------------------------------------------------------
stim_dims = 2
rnd_seed = 0
colordiff_alg = "CIE1994"
plane_2D = "GB plane" #"GB plane", "RB plane", "RG plane"
num_grid_pts = 5
fixed_val = 0.5
jitter = 0.1

# Color-threshold helper (handles transforms + loading GT CIE contours)
color_thres_data = color_thresholds(stim_dims,
                                    base_dir,
                                    plane_2D=plane_2D,
                                    fixed_value= fixed_val
                                    )

# Load ground-truth CIE thresholds (precomputed) for the chosen ΔE variant + grid resolution
color_thres_data.load_CIE_data(CIE_version=colordiff_alg,
                               num_grid_pts=num_grid_pts,
                               )

# Build the appropriate stimulus config for this plane
if plane_2D == "Isoluminant plane":    
    # Isoluminant: work in 2D W-space (bounded [-1, 1]) but keep transforms for RGB <-> W
    color_thres_data.load_transformation_matrix(file_date="02242025")

    stim_config = StimConfig_isoluminant(
        gt=colordiff_alg,
        fixed_ref=True,
        num_grid_pts=num_grid_pts,
        random_seed=rnd_seed,
        random_jitter= jitter,
        M_RGBTo2DW=color_thres_data.M_RGBTo2DW,
        M_2DWToRGB=color_thres_data.M_2DWToRGB,
    )
else:    
    # RGB slice: choose which RGB channel is held fixed based on the selected 2D plane
    plane_to_fixed = {"GB plane": "R", "RB plane": "G", "RG plane": "B"}

    # RGB slice: pick which channel is held fixed (R/G/B) and what value it is fixed to
    stim_config = StimConfig_RGBslices(
        fixed_plane=plane_to_fixed[plane_2D],
        fixed_val=0.5,
        gt=colordiff_alg,
        fixed_ref=True,
        num_grid_pts=num_grid_pts,
        random_seed=rnd_seed,
        random_jitter= jitter
    )


#%%
#Here is what we do if we want to load the data
with open(color_thres_data.file_path_CIE_data, 'rb') as f: 
    gt_CIE = pickle.load(f)
sim_trial = TrialPlacement_RGB_gridRef_gtCIE(gt_CIE,
                                             config = stim_config
                                             )

#% Define the Weibull psychometric function with specified parameters
# Calculate the probability of correct response given alpha and beta.
deltaE_1JND = 2.5
sim_trial.setup_WeibullFunc(alpha = 3.189, 
                            beta = 1.505, 
                            guessing_rate = 1/3, 
                            deltaE_1JND= deltaE_1JND
                            ) 
# Print the target probability based on the Weibull function for the given delta E
print(f"target probability: {sim_trial.sim['pC_given_alpha_beta']}")

# Initialize the SimThresCIELab object with a path to necessary files and background RGB value
background_RGB = np.array([0.5, 0.5, 0.5])
sim_CIELab = SimThresCIELab(background_RGB)

# Run the simulation with the specified random seed
sim_trial.run_sim(sim_CIELab)    

#%%
#first visualize the Weibull psychometric functions
sim_vis = TrialPlacementVisualization(sim_trial, 
                                      settings = pltSettings_base,
                                      save_fig = False)
pltSettings_PMF = replace(PlotWeibullPMFSettings(), **pltSettings_base.__dict__)
pltSettings_PMF = replace(pltSettings_PMF, xticks = np.linspace(0, 9, 4))
x_PMF = np.linspace(0,9,100)
sim_vis.plot_WeibullPMF(x_PMF, settings = pltSettings_PMF)


pltSettings_2D = replace(Plot2DSampledCompSettings(), **pltSettings_base.__dict__)
#define figure name
str_ext = f'_{colordiff_alg}' if sim_trial.config.plane_2D == 'Isoluminant plane' else ''
fig_name_firsthalf = f"_{sim_trial.config.plane_2D}{str_ext}"
fig_name_secondhalf = f"_sim{sim_trial.sim['nSims']}_perCond_sampling_NearContour"
fig_name_end = f"_jitter{sim_trial.config.random_jitter}_seed{rnd_seed}"

pltSettings_2D = replace(pltSettings_2D, 
                         xbds = [-0.15, 0.15],
                         ybds = [-0.15, 0.15],
                         fig_name = f"Sim{fig_name_firsthalf}{fig_name_secondhalf}{fig_name_end}")

# Visualize the sampled data from the simulation
sim_vis.plot_2D_sampledComp(settings = pltSettings_2D)  

#% If the sampling method is 'NearContour', visualize the entire transformation process
# Select a reference stimulus location (e.g., row 4, column 4)
row_eg = 4
col_eg = 0
# Retrieve parameters for the ellipsoid at the selected location
ellPara_eg = sim_trial.gt_CIE_results['ellParams'][sim_trial.config.fixed_color_dim,row_eg, col_eg]
# Extract the reference RGB values at the selected location
rgb_ref_eg = sim_trial.sim['ref_points'][row_eg, col_eg] #sim_trial.config.varying_RGBplane
# Retrieve the ground truth ellipses during the transformation process
rgb_comp_eg, rgb_comp_eg_1stepback, rgb_comp_eg_2stepback, rgb_comp_eg_3stepback =\
    sim_trial.sample_comp_2DNearContour(rgb_ref_eg, ellPara_eg) 
    
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
fig_name_insert = f"_dim1_{rgb_ref_eg[0]:.2f}_dim2_{rgb_ref_eg[1]:.2f}" 
    
#plot the transformation
pltSettings_ts = replace(PlotTransformationSettings(), **pltSettings_base.__dict__)
pltSettings_ts = replace(pltSettings_ts, 
                         colorcode_resp = True,
                         fig_name = f"Transformation{fig_name_firsthalf}"+\
                             f"{fig_name_secondhalf}{fig_name_end}{fig_name_insert}")
sim_vis.plot_transformation(rgb_comp_eg_3stepback, rgb_comp_eg_2stepback, 
                            rgb_comp_eg_1stepback, rgb_comp_eg[sim_trial.config.varying_color_dim],
                            resp = sim_trial.sim['resp_binary'][row_eg, col_eg],
                            gt = [gt_comp_eg_3stepback, gt_comp_eg_2stepback,
                                  gt_comp_eg_1stepback, gt_comp_eg],
                            settings = pltSettings_ts)

#%% save to pkl
file_name = f"Sims_isothreshold_{plane_2D}_{colordiff_alg}_sim"+\
    f"{sim_trial.sim['nSims']}perCond_samplingNearContour"+\
    f"_jitter{jitter}_seed{rnd_seed}.pkl"     
full_path = os.path.join(output_fileDir, file_name)

variable_names = ['sim_trial','color_thres_data','background_RGB', 'sim_CIELab',
                  'sobol_lb', 'sobol_ub']
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


