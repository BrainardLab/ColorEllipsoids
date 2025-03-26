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
import matplotlib.pyplot as plt
from dataclasses import replace
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

#import functions from the other script
func_path = '/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/Python version/'
sys.path.append(func_path)
from analysis.trial_placement import TrialPlacement_Isoluminant_sobolRef_gtCIE, TrialPlacement_RGB_gridRef_gtCIE
from analysis.simulations_CIELab import SimThresCIELab
from plotting.trial_placement_nonadaptive_plotting import TrialPlacementVisualization,\
        Plot2DSampledCompSettings
sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids")
from analysis.ellipses_tools import PointsOnEllipseQ, UnitCircleGenerate
from analysis.color_thres import color_thresholds
from plotting.adaptive_sampling_plotting import SamplingRefCompPairVisualization
from plotting.wishart_plotting import PlotSettingsBase

# Define base directories for saving figure and data files
base_dir = '/Volumes/T9/Aguirre-Brainard Lab Dropbox/Fangfang Hong/ELPS_analysis/'
CIE_dir = '/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/FilesFromPsychtoolbox/'
output_figDir = os.path.join(base_dir, 'Simulation_FigFiles','Python_version','transformation')
output_fileDir = os.path.join(base_dir, 'Simulation_DataFiles', 'Isoluminant plane')

#%% 
flag_grid = True
#specify the seed
rnd_seed = 1
colordiff_alg = 'CIE2000'
color_thres_data = color_thresholds(2, base_dir,
                                    plane_2D = 'Isoluminant plane')
# Load Wishart model fits
color_thres_data.load_CIE_data(CIE_version = colordiff_alg)
color_thres_data.load_transformation_matrix()

test_case1 = {'stimulus_space': 'GB RG RB slices',
              'gt': colordiff_alg,
              'fixed_ref': True,
              'M_RGBTo2DW': None,
              'M_2DWToRGB': None,
              'file_name': f'Isothreshold_contour_CIELABderived_fixedVal0.5_{colordiff_alg}.pkl'}
test_case2 = {'stimulus_space': 'Isoluminant plane',
              'gt': colordiff_alg,
              'fixed_ref': True,
              'M_RGBTo2DW': color_thres_data.M_RGBTo2DW,
              'M_2DWToRGB': color_thres_data.M_2DWToRGB,
              'file_name': f'Isothreshold_ellipses_isoluminant_{colordiff_alg}.pkl'}
# test_case3 = {'stimulus_space': 'RGB',
#               'gt': colordiff_alg,
#               'fixed_ref': True,
#               'M_RGBTo2DW': None,
#               'M_2DWToRGB': None,
#               'file_name': f'Isothreshold_ellipsoid_CIELABderived_{colordiff_alg}.pkl'}
test_case3 = {'stimulus space': 'Isoluminant plane',
              'gt': colordiff_alg,
              'fixed_ref': False,
              'M_RGBTo2DW': color_thres_data.M_RGBTo2DW,
              'M_2DWToRGB': color_thres_data.M_2DWToRGB,
              'file_name': None}


all_test_cases = {1: test_case1, 2: test_case2, 3: test_case3}

#%%
selected_case = 1
# Set up the simulation object
# QUESTION 1: Ask the user which RGB plane to fix during the simulation
# QUESTION 2: For 'NearContour', ask for the jitter variability
# QUESTION 3: Ask how many simulation trials
if selected_case in [1,2]:
    slc_case = all_test_cases[selected_case]
    #load this file if we want to sample trials on the isoluminant plane
    full_path = f"{base_dir}/Simulation_DataFiles/{slc_case['file_name']}"

    #Here is what we do if we want to load the data
    with open(full_path, 'rb') as f:
        gt_CIE = pickle.load(f)
        
    sim_trial = TrialPlacement_RGB_gridRef_gtCIE(gt_CIE,
                                                 colordiff_alg = colordiff_alg,
                                                 random_seed = rnd_seed,
                                                 M_RGBTo2DW = slc_case['M_RGBTo2DW'],
                                                 M_2DWToRGB = slc_case['M_2DWToRGB'])
else:
    # Set up the simulation object
    sim_trial = TrialPlacement_Isoluminant_sobolRef_gtCIE(M_RGBTo2DW = slc_case['M_RGBTo2DW'],
                                                          M_2DWToRGB = slc_case['M_2DWToRGB'],
                                                          colordiff_alg = colordiff_alg,
                                                          random_seed= rnd_seed)


#% Define the Weibull psychometric function with specified parameters
# Calculate the probability of correct response given alpha and beta.
deltaE_1JND = 2.5
sim_trial.setup_WeibullFunc(alpha = 3.189, 
                            beta = 1.505, 
                            guessing_rate = 1/3, 
                            deltaE_1JND= deltaE_1JND) #1.17, 2.33
# Print the target probability based on the Weibull function for the given delta E
print(f"target probability: {sim_trial.sim['pC_given_alpha_beta']}")

# Initialize the SimThresCIELab object with a path to necessary files and background RGB value
background_RGB = np.array([0.5, 0.5, 0.5])
sim_CIELab = SimThresCIELab(CIE_dir, background_RGB)


if selected_case in [1,2]:
    # Run the simulation with the specified random seed
    sim_trial.run_sim(sim_CIELab)
else:
    sim_trial.run_sim(sim_CIELab, 
                     [-0.75, -0.75, 0],
                     [0.75, 0.75, 360])    
    

#%%
# Create settings instance with custom fig_dir
pltSettings_base = PlotSettingsBase(fig_dir=os.path.join(output_figDir))
# Initialize the visualization object for plotting simulation results
sim_trial_2D_vis = TrialPlacementVisualization(sim_trial,
                                               settings = pltSettings_base,
                                               save_fig=False) 
# Visualize the Weibull psychometric function
x_PMF = np.linspace(0,6,100)
pltSettings_PMF = replace(Plot2DSampledCompSettings(), **pltSettings_base.__dict__)
sim_trial_2D_vis.plot_WeibullPMF(x_PMF)

if not flag_grid:
    sampling_vis = SamplingRefCompPairVisualization(2,
                                                    color_thres_data,
                                                    save_fig = False)
    
    # This array defines the opacity of markers in the plots, decreasing with more trials.
    slc_datapoints_to_show_lb = np.arange(0, sim_trial.sim['nSims'], 200)
    slc_datapoints_to_show_ub = np.arange(200, sim_trial.sim['nSims'], 200)
    xref_jnp = jnp.array(sim_trial.sim['ref_points'])[:2,:].T
    x1_jnp = jnp.array(sim_trial.sim['rgb_comp'])[:2,:].T
    
    # Loop over the selected data points to generate and visualize each corresponding figure.
    for i, (lb_i, ub_i) in enumerate(zip(slc_datapoints_to_show_lb, slc_datapoints_to_show_ub)):
        fig, ax = plt.subplots(1, 1, figsize = (3,3.5), dpi= 1024)
        # Visualize the trials up to the nth data point with specified marker transparency.
        sampling_vis.plot_sampling(xref_jnp[lb_i:ub_i],  # Reference points up to the nth data point
                                   x1_jnp[lb_i:ub_i],    # Comparison points up to the nth data point
                                   ax = ax,
                                   linealpha = 0.3,        # Line transparency for this subset of data
                                   bounds = 0.75 *np.array([-1,1]),
                                   flag_rescale_axes_label = False,
                                   comp_markeralpha = 0.3)              # Filename under which the figure will be saved
        ax.set_xlabel('Wishart space dimension 1')
        ax.set_ylabel('Wishart space dimension 2')
        ax.set_title('Isoluminant plane', fontsize = 10)
        # Save the figure as a PDF
        #fig.savefig(os.path.join(output_figDir_fits, fig_name +'.pdf'), bbox_inches='tight')    
        plt.show()
else:
    pltSettings_2D = replace(Plot2DSampledCompSettings(), **pltSettings_base.__dict__)
    pltSettings_2D = replace(pltSettings_2D, 
                             xbds = [-0.07, 0.07],
                             ybds = [-0.07, 0.07])

    # Visualize the sampled data from the simulation
    sim_trial_2D_vis.plot_2D_sampledComp(settings = pltSettings_2D)  

#%% If the sampling method is 'NearContour', visualize the entire transformation process
# Select a reference stimulus location (e.g., row 4, column 4)
row_eg = 4
col_eg = 0
# Retrieve parameters for the ellipsoid at the selected location
ellPara_eg = sim_trial.gt_CIE_results['ellParams'][sim_trial.sim['slc_RGBplane'],row_eg, col_eg]
# Extract the reference RGB values at the selected location
rgb_ref_eg = sim_trial.sim['ref_points'][:,row_eg, col_eg] #sim_trial.sim['varying_RGBplane']
# Retrieve the ground truth ellipses during the transformation process
rgb_comp_eg, rgb_comp_eg_1stepback, rgb_comp_eg_2stepback, rgb_comp_eg_3stepback =\
    sim_trial.sample_rgb_comp_2DNearContour(rgb_ref_eg, ellPara_eg) 
    
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
str_ext = f'_{colordiff_alg}' if sim_trial.sim['plane_2D'] == 'Isoluminant plane' else ''
fig_name_firsthalf = f"Sims_transformation_{sim_trial.sim['plane_2D']}{str_ext}"+\
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
                                     fig_name = f"{fig_name_firsthalf}{fig_name_secondhalf}.pdf")

#%% save to pkl
file_name = f"Sims_isothreshold_{sim_trial.sim['plane_2D']}_{colordiff_alg}_sim"+\
    f"{sim_trial.sim['nSims']}perCond_samplingNearContour"+\
    f"_jitter{sim_trial.sim['random_jitter']}_seed{rnd_seed}.pkl"     
full_path = f"{output_fileDir}/{file_name}"
    
sim = sim_trial.sim
# Write the list of dictionaries to a file using pickle
with open(full_path, 'wb') as f:
    pickle.dump([sim], f)


        
        
    
    
    
