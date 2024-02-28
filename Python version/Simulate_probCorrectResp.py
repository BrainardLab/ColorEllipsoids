#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 15:01:11 2024

@author: fangfang
"""

import os
import pickle
import numpy as np
from math import pi
import matplotlib.pyplot as plt

#import functions from other files
func_path = '/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/Python version/'
os.chdir(func_path)
from IsothresholdContour_RGBcube import convert_rgb_lab

file_name = 'Isothreshold_contour_CIELABderived.pkl'
path_str = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/ELPS_analysis/Simulation_DataFiles/'
full_path = f"{path_str}{file_name}"
os.chdir(path_str)

#Here is what we do if we want to load the data
with open(full_path, 'rb') as f:
    # Load the object from the file
    data_load = pickle.load(f)
param, stim, results, plt_specifics = data_load[0], data_load[1], data_load[2], data_load[3]
    
#%%
sim = {}
#First select a plane
default_value = 'R'
slc_RGBplane = input('Which plane would you like to fix (R/G/B; default: R):')

RGB_test = 'RGB'
if RGB_test.find(slc_RGBplane) != -1:
    sim['slc_RGBplane'] = RGB_test.find(slc_RGBplane)
else:
    sim['slc_RGBplane'] = default_value

sim['varying_RGBplane'] = list(range(3))
sim['varying_RGBplane'].remove(sim['slc_RGBplane'])

sim['plane_points'] = stim['plane_points'][sim['slc_RGBplane']] #size: 3 x 3 x 100 x 100
sim['ref_points'] = stim['ref_points'][sim['slc_RGBplane']]
sim['background_RGB'] = stim['background_RGB']
sim['fixed_RGBvec'] = stim['fixed_RGBvec']

#%%
#information of the psychometric function
sim['alpha'] = 1.1729
sim['beta']  = 1.2286
WeibullFunc = lambda x: (1 - 0.5*np.exp(- (x/sim['alpha'])** sim['beta']))
sim['pC_given_alpha_beta'] = WeibullFunc(stim['deltaE_1JND'])
sim['nSims'] = 480
default_str = 'NearContour'
sim['method_sampling'] = input('Which sampling method (NearContour/Random):')

if sim['method_sampling'] == 'NearContour':
    sim['random_jitter'] = 0.5
elif sim['method_sampling'] == 'Random':
    sim['range_randomSampling'] = [-0.025, 0.025]
else: 
    sim['method_sampling'] = default_str
    sim['random_jitter'] = 0.5

#%% FUNCTIONS
def sample_rgb_comp_2DNearContour(rgb_ref, varying_RGBplane, slc_fixedVal,
                                  nSims, paramEllipse, jitter):
    #Identify the fixed RGB dimension by excluding the varying dimensions.
    allPlans = set(range(3))
    fixed_RGBplane = list(allPlans.difference(set(varying_RGBplane)))
    
    #Initialize the output matrix with nans
    rgb_comp_sim = np.full((3, nSims), np.nan)
    
    #Generate random angles to simulate points around the ellipse.
    randTheta = np.random.rand(1,nSims) * 2 * pi
    
    #calculate x and y coordinates with added jitter
    randx = np.cos(randTheta) + np.random.randn(1,nSims) * jitter
    randy = np.sin(randTheta) + np.random.randn(1,nSims) * jitter
    
    #adjust coordinates based on the ellipsoid's semi-axis lengths
    randx_stretched = randx * paramEllipse[2]
    randy_stretched = randy * paramEllipse[3]
    
    #calculate the varying RGB dimensions, applying rotation and translation 
    #based on the reference RGB values and ellipsoid parameters
    rgb_comp_sim[varying_RGBplane[0],:] = \
        randx_stretched * np.cos(np.deg2rad(paramEllipse[-1])) - \
        randy_stretched * np.sin(np.deg2rad(paramEllipse[-1])) + rgb_ref[0]
    rgb_comp_sim[varying_RGBplane[1],:] = \
        randx_stretched * np.sin(np.deg2rad(paramEllipse[-1])) + \
        randy_stretched * np.cos(np.deg2rad(paramEllipse[-1])) + rgb_ref[1]
        
    #set the fixed RGB dimension to the specificed fixed value for all simulations
    rgb_comp_sim[fixed_RGBplane,:] = slc_fixedVal;
    
    return rgb_comp_sim

def sample_rgb_comp_2Drandom(rgb_ref, varying_RGBplane, slc_fixedVal,
                             box_range, nSims):
    #Identify the fixed RGB dimension by excluding the varying dimensions.
    allPlans = set(range(3))
    fixed_RGBplane = list(allPlans.difference(set(varying_RGBplane)))
    
    rgb_comp_sim = np.random.rand(3, nSims) * (box_range[1] - box_range[0]) + \
        box_range[0]
        
    rgb_comp_sim[fixed_RGBplane,:] = slc_fixedVal
    
    rgb_comp_sim[varying_RGBplane,:] = rgb_comp_sim[varying_RGBplane,:] + \
        rgb_ref.reshape((2,1))
        
    return rgb_comp_sim

#%%
def plot_2D_sampledComp(grid_ref_x, grid_ref_y, rgb_comp, varying_RGBplane,\
                        method_sampling, **kwargs):
    pltParams = {
        'slc_x_grid_ref': np.arange(len(grid_ref_x)),
        'slc_y_grid_ref': np.arange(len(grid_ref_y)),
        'groundTruth': None,
        'modelPredictions': None,
        'responses':None,
        'xbds':[-0.025, 0.025],
        'ybds':[-0.025, 0.025],
        'nFinerGrid': 50,
        'EllipsesColor': np.array([178, 34, 34]) / 255,
        'WishartEllipsesColor': np.array([76, 153, 0]) / 255,
        'marker1':'.',
        'marker0':'*',
        'lineWidth': 1,
        'markerColor1': np.array([173, 216, 230]) / 255,
        'markerColor0': np.array([255, 179, 138]) / 255,
        'saveFig': False,
        'figName': 'Sampled comparison stimuli'
        }
    
    #update default options with any keyword arguments provided
    pltParams.update(kwargs)
    
    nGrid_x = len(grid_ref_x)
    nGrid_y = len(grid_ref_y)
    
    plt.figure(figsize = (8,8))
    for i in range(nGrid_x):
        for j in range(nGrid_y):
            x_axis = np.linspace(pltParams['xbds'][0], pltParams['xbds'][1],\
                                 pltParams['nFinerGrid']) + grid_ref_x[j]
            y_axis = np.linspace(pltParams['ybds'][0], pltParams['ybds'][1],\
                                 pltParams['nFinerGrid']) + grid_ref_y[i]    
            
            #subplot
            plt.subplot(nGrid_x, nGrid_y, (nGrid_x-i-1)*nGrid_y + j + 1)
            
            #plot the ground truth
            if pltParams['groundTruth'] is not None:
                plt.plot(pltParams['groundTruth'][i,j,0,:],\
                         pltParams['groundTruth'][i,j,1,:],\
                         color=pltParams['EllipsesColor'],\
                         linestyle = '--', linewidth = pltParams['lineWidth'])
            
            idx_1 = np.where(pltParams['responses'][i,j,:] == 1)
            idx_0 = np.where(pltParams['responses'][i,j,:] == 0)
            plt.scatter(rgb_comp[i, j, varying_RGBplane[0], idx_1],\
                        rgb_comp[i, j, varying_RGBplane[1], idx_1],\
                        s = 2, marker=pltParams['marker1'],\
                        c=pltParams['markerColor1'],alpha=0.5)
                
            plt.scatter(rgb_comp[i, j, varying_RGBplane[0], idx_0], 
                        rgb_comp[i, j, varying_RGBplane[1], idx_0], \
                        s = 2, marker=pltParams['marker0'], \
                        c=pltParams['markerColor0'],alpha=0.5)
            
            plt.xlim([x_axis[0], x_axis[-1]])
            plt.ylim([y_axis[0], y_axis[-1]])
            
            if j == 0: plt.yticks(np.round([grid_ref_y[i]],2))
            else: plt.yticks([])
            
            if i == 0: plt.xticks(np.round([grid_ref_x[j]],2))
            else: plt.xticks([])
        
    plt.subplots_adjust(wspace = 0, hspace = 0)
    plt.tight_layout()
    plt.show()
    
    
#%%
sim['rgb_comp'] = np.full((stim['nGridPts_ref'], stim['nGridPts_ref'],3,sim['nSims']),np.nan)
sim['lab_comp'] = np.full(sim['rgb_comp'].shape, np.nan)
sim['deltaE'] = np.full((stim['nGridPts_ref'], stim['nGridPts_ref'],\
                         sim['nSims']), np.nan)
sim['probC'] = np.full(sim['deltaE'].shape, np.nan)
sim['resp_binary'] = np.full(sim['deltaE'].shape, np.nan)
for i in range(stim['nGridPts_ref']):
    for j in range(stim['nGridPts_ref']):
        #grab the reference stimulus' RGB
        rgb_ref_ij = sim['ref_points'][:,i,j]
        #convert it to Lab 
        ref_Lab_ij, _, _ = convert_rgb_lab(param['B_monitor'],\
                                            sim['background_RGB'],\
                                            rgb_ref_ij)
        
        #simulate comparison stimulus
        if sim['method_sampling'] == 'NearContour':
            opt_vecLen_ij = results['opt_vecLen'][sim['slc_RGBplane']][i,j,:]
            ellPara = results['ellParams'][sim['slc_RGBplane']][i,j,:]
            sim['rgb_comp'][i,j,:,:] = sample_rgb_comp_2DNearContour(\
                rgb_ref_ij[sim['varying_RGBplane']], sim['varying_RGBplane'],
                sim['fixed_RGBvec'], sim['nSims'], ellPara, sim['random_jitter'])
        elif sim['method_sampling'] == 'Random':
            sim['rgb_comp'][i,j,:,:] = sample_rgb_comp_2Drandom(\
                rgb_ref_ij[sim['varying_RGBplane']],sim['varying_RGBplane'],\
                sim['fixed_RGBvec'], sim['range_randomSampling'], sim['nSims'])
            
        
        #simulate binary responses
        for n in range(sim['nSims']):
            sim['lab_comp'][i,j,:,n], _, _ = convert_rgb_lab(param['B_monitor'],\
                sim['background_RGB'], sim['rgb_comp'][i,j,:,n])
            sim['deltaE'][i,j,n] = np.linalg.norm(sim['lab_comp'][i,j,:,n] - \
                                                  ref_Lab_ij)
            sim['probC'][i,j,n] = WeibullFunc(sim['deltaE'][i,j,n])
        sim['resp_binary'][i,j,:] = np.random.binomial(1, sim['probC'][i,j,:],\
                                                       (sim['nSims'],))
            
            
#%%
plot_2D_sampledComp(stim['grid_ref'], stim['grid_ref'], sim['rgb_comp'],\
                    sim['varying_RGBplane'], sim['method_sampling'], \
                    responses = sim['resp_binary'], \
                    groundTruth = results['fitEllipse_unscaled'][sim['slc_RGBplane']])        
        
        
        
        
    
    
    
