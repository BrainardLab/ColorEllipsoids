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
import matplotlib.pyplot as plt
from dataclasses import replace
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

#import functions from the other script
func_path = '/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/Python version/'
sys.path.append(func_path)
from analysis.trial_placement import StimConfig, TrialPlacement_Isoluminant_sobolRef_gtCIE,\
    TrialPlacement_RGB_gridRef_gtCIE
from analysis.simulations_CIELab import SimThresCIELab
from plotting.trial_placement_nonadaptive_plotting import TrialPlacementVisualization,\
        Plot2DSampledCompSettings, PlotWeibullPMFSettings, PlotTransformationSettings
sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids")
from analysis.ellipses_tools import PointsOnEllipseQ, UnitCircleGenerate
from analysis.color_thres import color_thresholds
from plotting.adaptive_sampling_plotting import SamplingRefCompPairVisualization, \
    Plot2DSamplingSettings
from plotting.wishart_plotting import PlotSettingsBase

# Define base directories for saving figure and data files
base_dir = '/Volumes/T9/Aguirre-Brainard Lab Dropbox/Fangfang Hong/ELPS_analysis/'
output_figDir = os.path.join(base_dir, 'Simulation_FigFiles','Python_version',
                             'Isoluminant plane')
output_fileDir = os.path.join(base_dir, 'Simulation_DataFiles', 'Isoluminant plane')
pltSettings_base = PlotSettingsBase(fig_dir=output_figDir, fontsize = 8)

#%% 
#specify the seed
rnd_seed = 0
colordiff_alg = 'CIE1994'
color_thres_data = color_thresholds(2, base_dir, plane_2D = 'Isoluminant plane')
# Load Wishart model fits
color_thres_data.load_CIE_data(CIE_version = colordiff_alg)
color_thres_data.load_transformation_matrix()

"""
EXAMPLE 1: isoluminant plane, no fixed ref
    
stim_config = StimConfig(
    fixed_plane ='lum',
    gt = colordiff_alg,
    fixed_ref = False,
    M_RGBTo2DW=color_thres_data.M_RGBTo2DW,
    M_2DWToRGB=color_thres_data.M_2DWToRGB,
    file_name=None 
)

EXAMPLE 2: GB/RG/RB plane, fixed ref
    
stim_config = StimConfig(
    fixed_plane ='R', #could be R, G, or B
    gt = colordiff_alg,
    fixed_ref = True,
    M_RGBTo2DW= None,
    M_2DWToRGB= None,
    file_name=f'Isothreshold_contour_CIELABderived_fixedVal0.5_{colordiff_alg}.pkl'
)

EXAMPLE 3: isoluminant plane, fixed ref

stim_config = StimConfig(
    fixed_plane = 'lum',
    gt = colordiff_alg,
    fixed_ref = True,
    M_RGBTo2DW=color_thres_data.M_RGBTo2DW,
    M_2DWToRGB=color_thres_data.M_2DWToRGB,
    file_name = f'Isothreshold_ellipses_isoluminant_{colordiff_alg}.pkl'
)

"""
stim_config = StimConfig(
    fixed_plane = 'lum',
    gt = colordiff_alg,
    fixed_ref = True,
    M_RGBTo2DW=color_thres_data.M_RGBTo2DW,
    M_2DWToRGB=color_thres_data.M_2DWToRGB,
    file_name = f'Isothreshold_ellipses_isoluminant_{colordiff_alg}.pkl'
)

#%%
if stim_config.fixed_ref:
    #Here is what we do if we want to load the data
    with open(color_thres_data.file_path_CIE_data, 'rb') as f: 
        gt_CIE = pickle.load(f)
    sim_trial = TrialPlacement_RGB_gridRef_gtCIE(gt_CIE,
                                                 config = stim_config,
                                                 random_seed = rnd_seed)
else:
    # Set up the simulation object
    sim_trial = TrialPlacement_Isoluminant_sobolRef_gtCIE(config = stim_config,
                                                          random_seed= rnd_seed)

#% Define the Weibull psychometric function with specified parameters
# Calculate the probability of correct response given alpha and beta.
deltaE_1JND = 2.5
sim_trial.setup_WeibullFunc(alpha = 3.189, 
                            beta = 1.505, 
                            guessing_rate = 1/3, 
                            deltaE_1JND= deltaE_1JND) #all these parameters will be saved in sim_trial.sim
# Print the target probability based on the Weibull function for the given delta E
print(f"target probability: {sim_trial.sim['pC_given_alpha_beta']}")

# Initialize the SimThresCIELab object with a path to necessary files and background RGB value
background_RGB = np.array([0.5, 0.5, 0.5])
sim_CIELab = SimThresCIELab(background_RGB)

if stim_config.fixed_ref:
    # Run the simulation with the specified random seed
    sim_trial.run_sim(sim_CIELab)
else:
    sobol_lb = [-0.8, -0.8, 0]
    sobol_ub = [0.8, 0.8, 360]
    sim_trial.run_sim(sim_CIELab, sobol_bounds_lb = sobol_lb, sobol_bounds_ub = sobol_ub)    
    

#%%
#first visualize the Weibull psychometric functions
sim_vis = TrialPlacementVisualization(sim_trial, 
                                      settings = pltSettings_base,
                                      save_fig = True)
pltSettings_PMF = replace(PlotWeibullPMFSettings(), **pltSettings_base.__dict__)
pltSettings_PMF = replace(pltSettings_PMF, xticks = np.linspace(0, 9, 4))
x_PMF = np.linspace(0,9,100)
sim_vis.plot_WeibullPMF(x_PMF, settings = pltSettings_PMF)

#if the ref is sobol-generated, then we visualize the data using the following way
if not stim_config.fixed_ref: 
    # Create settings instance with custom fig_dir
    pltSettings_tp = replace(Plot2DSamplingSettings(), **pltSettings_base.__dict__)
    pltSettings_tp = replace(pltSettings_tp,
                             linealpha = 0.3,        # Line transparency for this subset of data
                             bounds = 0.75 *np.array([-1,1]),
                             flag_rescale_axes_label = False,
                             comp_markeralpha = 0.3)
    sampling_vis = SamplingRefCompPairVisualization(2,
                                                    color_thres_data,
                                                    settings = pltSettings_tp,
                                                    save_fig = False)
    
    # These two sets of data are selected for no particular reason
    # if we plot too many data, visibility will go down
    slc_datapoints_to_show_lb = [0, 300]
    slc_datapoints_to_show_ub = [300, 600]
    xref_jnp = jnp.array(sim_trial.sim['ref_points'])[:2,:].T #the last row is a filler row (all 1's)
    x1_jnp = jnp.array(sim_trial.sim['comp'])[:2,:].T #so we can just get rid of that row
    
    # Loop over the selected data points to generate and visualize each corresponding figure.
    for i, (lb_i, ub_i) in enumerate(zip(slc_datapoints_to_show_lb, slc_datapoints_to_show_ub)):
        fig, ax = plt.subplots(1, 1, figsize = pltSettings_tp.fig_size, dpi= pltSettings_tp.dpi)
        sampling_vis.plot_sampling(xref_jnp[lb_i:ub_i], x1_jnp[lb_i:ub_i], ax = ax,
                                   settings = pltSettings_tp)     
        ax.set_title(f'Isoluminant plane ({colordiff_alg})')
        fig_name = f"Sims_isothreshold_isoluminant_from{lb_i}to{ub_i}_randomRef_"+\
            f"{colordiff_alg}_jitter{sim_trial.sim['random_jitter']}.pdf"
        # Save the figure as a PDF
        fig.savefig(os.path.join(output_figDir, fig_name), bbox_inches='tight')    
        plt.show()
else:
    pltSettings_2D = replace(Plot2DSampledCompSettings(), **pltSettings_base.__dict__)
    #define figure name
    str_ext = f'_{colordiff_alg}' if sim_trial.sim['plane_2D'] == 'Isoluminant plane' else ''
    fig_name_firsthalf = f"_{sim_trial.sim['plane_2D']}{str_ext}"
    fig_name_secondhalf = f"_sim{sim_trial.sim['nSims']}_perCond_sampling_"+\
        f"{sim_trial.sim['method_sampling']}"
    fig_name_end = f"_jitter{sim_trial.sim['random_jitter']}_seed{rnd_seed}"

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
    ellPara_eg = sim_trial.gt_CIE_results['ellParams'][sim_trial.sim['slc_RGBplane'],row_eg, col_eg]
    # Extract the reference RGB values at the selected location
    rgb_ref_eg = sim_trial.sim['ref_points'][row_eg, col_eg] #sim_trial.sim['varying_RGBplane']
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
                                rgb_comp_eg_1stepback, rgb_comp_eg[sim_trial.sim['varying_RGBplane']],
                                resp = sim_trial.sim['resp_binary'][row_eg, col_eg],
                                gt = [gt_comp_eg_3stepback, gt_comp_eg_2stepback,
                                      gt_comp_eg_1stepback, gt_comp_eg],
                                settings = pltSettings_ts)

#%% save to pkl
str_trial = 'perCond' if stim_config.fixed_ref else 'total'
file_name = f"Sims_isothreshold_{sim_trial.sim['plane_2D']}_{colordiff_alg}_sim"+\
    f"{sim_trial.sim['nSims']}{str_trial}_samplingNearContour"+\
    f"_jitter{sim_trial.sim['random_jitter']}_seed{rnd_seed}.pkl"     
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


