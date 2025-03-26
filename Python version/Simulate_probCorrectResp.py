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
CIE_dir = '/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/FilesFromPsychtoolbox/'
output_figDir = os.path.join(base_dir, 'Simulation_FigFiles','Python_version','Isoluminant plane')
output_fileDir = os.path.join(base_dir, 'Simulation_DataFiles', 'Isoluminant plane')
pltSettings_base = PlotSettingsBase(fig_dir=os.path.join(output_figDir), fontsize = 8)

#%% 
flag_grid = True
#specify the seed
rnd_seed = 1
colordiff_alg = 'CIE2000'
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
    file_name=f'Isothreshold_ellipses_isoluminant_{colordiff_alg}.pkl'
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
    #load this file if we want to sample trials on the isoluminant plane
    full_path = f"{base_dir}/Simulation_DataFiles/{stim_config.file_name}"

    #Here is what we do if we want to load the data
    with open(full_path, 'rb') as f: gt_CIE = pickle.load(f)
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
                            deltaE_1JND= deltaE_1JND) #1.17, 2.33
# Print the target probability based on the Weibull function for the given delta E
print(f"target probability: {sim_trial.sim['pC_given_alpha_beta']}")

# Initialize the SimThresCIELab object with a path to necessary files and background RGB value
background_RGB = np.array([0.5, 0.5, 0.5])
sim_CIELab = SimThresCIELab(CIE_dir, background_RGB)


if stim_config.fixed_ref:
    # Run the simulation with the specified random seed
    sim_trial.run_sim(sim_CIELab)
else:
    sim_trial.run_sim(sim_CIELab, 
                     sobol_bounds_lb = [-0.75, -0.75, 0],
                     sobol_bounds_ub = [0.75, 0.75, 360])    
    

#%%
#first visualize the Weibull psychometric functions
sim_vis = TrialPlacementVisualization(sim_trial, settings = pltSettings_base,
                                      save_fig= False)
pltSettings_PMF = replace(PlotWeibullPMFSettings(), **pltSettings_base.__dict__)
x_PMF = np.linspace(0,6,100)
sim_vis.plot_WeibullPMF(x_PMF, settings = pltSettings_PMF)

#if 
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
    
    # This array defines the opacity of markers in the plots, decreasing with more trials.
    slc_datapoints_to_show_lb = [0, 200]
    slc_datapoints_to_show_ub = [200, 400]
    xref_jnp = jnp.array(sim_trial.sim['ref_points'])[:2,:].T #the last row is a filler row (all 1's)
    x1_jnp = jnp.array(sim_trial.sim['rgb_comp'])[:2,:].T #so we can just get rid of that row
    
    # Loop over the selected data points to generate and visualize each corresponding figure.
    for i, (lb_i, ub_i) in enumerate(zip(slc_datapoints_to_show_lb, slc_datapoints_to_show_ub)):
        fig, ax = plt.subplots(1, 1, figsize = pltSettings_tp.fig_size, dpi= pltSettings_tp.dpi)
        sampling_vis.plot_sampling(xref_jnp[lb_i:ub_i], x1_jnp[lb_i:ub_i], ax = ax,
                                   settings = pltSettings_tp)     
        ax.set_title('Isoluminant plane')
        # Save the figure as a PDF
        #fig.savefig(os.path.join(output_figDir_fits, fig_name +'.pdf'), bbox_inches='tight')    
        plt.show()
else:
    pltSettings_2D = replace(Plot2DSampledCompSettings(), **pltSettings_base.__dict__)
    pltSettings_2D = replace(pltSettings_2D, 
                             xbds = [-0.07, 0.07],
                             ybds = [-0.07, 0.07])

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
    pltSettings_ts = replace(PlotTransformationSettings(), **pltSettings_base.__dict__)
    pltSettings_ts = replace(pltSettings_ts, 
                             colorcode_resp = True,
                             fig_name = f"{fig_name_firsthalf}{fig_name_secondhalf}")
    sim_vis.plot_transformation(rgb_comp_eg_3stepback, rgb_comp_eg_2stepback, 
                                rgb_comp_eg_1stepback, rgb_comp_eg[sim_trial.sim['varying_RGBplane']],
                                resp = sim_trial.sim['resp_binary'][row_eg, col_eg],
                                gt = [gt_comp_eg_3stepback, gt_comp_eg_2stepback,
                                      gt_comp_eg_1stepback, gt_comp_eg],
                                settings = pltSettings_ts)

#%% save to pkl
file_name = f"Sims_isothreshold_{sim_trial.sim['plane_2D']}_{colordiff_alg}_sim"+\
    f"{sim_trial.sim['nSims']}perCond_samplingNearContour"+\
    f"_jitter{sim_trial.sim['random_jitter']}_seed{rnd_seed}.pkl"     
full_path = f"{output_fileDir}/{file_name}"
    
sim = sim_trial.sim
# Write the list of dictionaries to a file using pickle
with open(full_path, 'wb') as f:
    pickle.dump([sim], f)


