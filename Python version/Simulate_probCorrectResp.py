#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 15:01:11 2024

@author: fangfang
"""

import os
import numpy as np
from math import pi
import matplotlib.pyplot as plt
import pickle

#import functions from the other script
func_path = '/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/Python version/'
os.chdir(func_path)
from IsothresholdContour_RGBcube import convert_rgb_lab

#%% FUNCTIONS
def sample_rgb_comp_2DNearContour(rgb_ref, varying_RGBplane, slc_fixedVal,
                                  nSims, paramEllipse, jitter):
    """
    Samples RGB compositions near an isothreshold ellipsoidal contour.
    This function generates simulated RGB compositions based on a reference
    RGB value and parameters defining an ellipsoidal contour. The function
    is designed to vary two of the RGB dimensions while keeping the third fixed,
    simulating points near the contour of an ellipsoid in RGB color space.

    Parameters
    ----------
    - rgb_ref (array): The reference RGB value around which to simulate new 
        points. This RGB value defines the center of the ellipsoidal contour 
        in RGB space. This array only includes RGB of varying dimensions
    - varying_RGBplane (list of int): The indices (0 for R, 1 for G, and 2 for B) 
        of the RGB dimensions to vary during the simulation, with the remaining 
        dimension being fixed.
    - slc_fixedVal (float): The fixed value for the RGB dimension not included 
        in `varying_RGBplane`. This value remains constant across all simulations.
    - nSims (float): The number of simulated RGB compositions to generate.
    - paramEllipse (array): Parameters defining the ellipsoid contour. 
        Includes the center coordinates in the varying dimensions, the lengths 
        of the semi-axes of the ellipsoid in the plane of variation, and the 
        rotation angle of the ellipsoid in degrees.
        Expected format: [xc, yc, semi_axis1_length, semi_axis2_length, rotation_angle].
    - jitter (float): The standard deviation of the Gaussian noise added to 
        the points on the ellipsoid contour, simulating a "jitter" to introduce 
        variability around the contour. 

    Returns
    -------
    - rgb_comp_sim (array): A 3xN array of simulated RGB compositions, 
        where N is the number of simulations (`nSims`). Each column represents 
        an RGB composition near the specified ellipsoidal contour in RGB color 
        space. The row order corresponds to R, G, and B dimensions, respectively.


    """
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
    """
    Generates random RGB compositions within a specified square range in the
    RGB color space. Two of the RGB dimensions are allowed to vary within the 
    specified range, while the third dimension is fixed at a specified value.

    Parameters
    ----------
    - rgb_ref (array): The reference RGB value which serves as the starting 
        point for the simulation. The varying components will be adjusted 
        relative to this reference.
    - varying_RGBplane (list of int): The indices (0 for R, 1 for G, and 2 for B) 
        of the RGB dimensions to vary during the simulation, with the remaining 
        dimension being fixed.
    - slc_fixedVal (float): The fixed value for the RGB dimension not included 
        in `varying_RGBplane`. This value remains constant for all generated 
        samples.
    - box_range (list of float): The range (min, max) within which to generate 
        random values for the varying RGB dimensions. The generated values 
        within this box are then adjusted based on the reference RGB value.
    - nSims (int): The number of random RGB compositions to generate.
        
    Returns
    ----------
    - rgb_comp_sim (array): A 3xN array of simulated RGB compositions, 
         where N is the number of simulations (`nSims`). Each column represents 
         an RGB composition near the specified ellipsoidal contour in RGB color 
         space. The row order corresponds to R, G, and B dimensions, respectively.

    """
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
        'x_label':'',
        'y_label':'',
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
            
            #find indices that correspond to a response of 1 / 0
            idx_1 = np.where(pltParams['responses'][i,j,:] == 1)
            idx_0 = np.where(pltParams['responses'][i,j,:] == 0)
            plt.scatter(rgb_comp[i, j, varying_RGBplane[0], idx_1],\
                        rgb_comp[i, j, varying_RGBplane[1], idx_1],\
                        s = 10, marker=pltParams['marker1'],\
                        c=pltParams['markerColor1'],alpha=0.8)
                
            plt.scatter(rgb_comp[i, j, varying_RGBplane[0], idx_0], 
                        rgb_comp[i, j, varying_RGBplane[1], idx_0], \
                        s = 10, marker=pltParams['marker0'], \
                        c=pltParams['markerColor0'],alpha=0.8)
            
            plt.xlim([x_axis[0], x_axis[-1]])
            plt.ylim([y_axis[0], y_axis[-1]])
            if i == 0 and j == nGrid_y//2: plt.xlabel(pltParams['x_label'])
            if i == nGrid_x//2 and j == 0: plt.ylabel(pltParams['y_label'])
            
            if j == 0: plt.yticks(np.round([grid_ref_y[i]],2))
            else: plt.yticks([])
            
            if i == 0: plt.xticks(np.round([grid_ref_x[j]],2))
            else: plt.xticks([])
        
    plt.subplots_adjust(wspace = 0, hspace = 0)
    plt.tight_layout()
    plt.show()
    
#%%
def main():
    file_name = 'Isothreshold_contour_CIELABderived.pkl'
    path_str = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
            'ELPS_analysis/Simulation_DataFiles/'
    full_path = f"{path_str}{file_name}"
    os.chdir(path_str)
    
    #Here is what we do if we want to load the data
    with open(full_path, 'rb') as f:
        # Load the object from the file
        data_load = pickle.load(f)
    param, stim, results, plt_specifics = data_load[0], data_load[1], data_load[2], data_load[3]
        
    #%% Define dictionary sim
    sim = {}
    #Set default values for the simulation parameters.
    default_value = 'R'
    default_str = 'NearContour'
    default_jitter = 0.1
    default_ub = 0.025
    default_trialNum = 80
    
    # QUESTION 1: Ask the user which RGB plane to fix during the simulation.
    # The default plane is 'R'.
    slc_RGBplane = input('Which plane would you like to fix (R/G/B; default: R):')
    
    # Validate the user input for the RGB plane. If valid, store the index; 
    #otherwise, use the default value.
    RGB_test = 'RGB'
    if RGB_test.find(slc_RGBplane) != -1:
        sim['slc_RGBplane'] = RGB_test.find(slc_RGBplane)
    else: sim['slc_RGBplane'] = default_value
    
    # QUESTION 2: Ask the user to choose the sampling method.
    # The default method is 'NearContour'.
    sim['method_sampling'] = input('Which sampling method (NearContour/Random; default: NearContour):')
    
    # Depending on the chosen sampling method, ask for relevant parameters.
    if sim['method_sampling'] == 'NearContour':
        # QUESTION 3: For 'NearContour', ask for the jitter variability.
        input_jitter = input('Enter variability of random jitter (default: 0.1):')
        if input_jitter != '':
            sim['random_jitter'] = float(input_jitter)
        else: sim['random_jitter'] = default_jitter
    elif sim['method_sampling'] == 'Random':
        # QUESTION 3: For 'Random', ask for the upper bound of the square range.
        square_ub = input('Enter the upper bound of the square (default: 0.025):')
        if square_ub != '':
            sim['range_randomSampling'] = [-float(square_ub), float(square_ub)]
        else: 
            sim['range_randomSampling'] = [-default_ub, default_ub]
    else:
        # If an invalid sampling method is entered, revert to default and ask 
        #for jitter variability.
        sim['method_sampling'] = default_str
        #QUESTION 3
        input_jitter = input('Enter variability of random jitter (default: 0.1):')
        if input_jitter != '':
            sim['random_jitter'] = float(input_jitter)
        else: sim['random_jitter'] = default_jitter
        
    #QUESTION 4: how many simulation trials 
    input_simTrials = input('How many simulation trials per cond (default: 80):')
    if input_simTrials != '':
        sim['nSims'] = int(input_simTrials)
    else: sim['nSims'] = default_trialNum
    
    # Configure the varying RGB planes based on the fixed plane selected by the user.
    sim['varying_RGBplane'] = list(range(3))
    sim['varying_RGBplane'].remove(sim['slc_RGBplane'])
    
    # Load specific simulation data based on the selected RGB plane.
    sim['plane_points'] = stim['plane_points'][sim['slc_RGBplane']] #size: 3 x 3 x 100 x 100
    sim['ref_points'] = stim['ref_points'][sim['slc_RGBplane']]
    sim['background_RGB'] = stim['background_RGB']
    sim['fixed_RGBvec'] = stim['fixed_RGBvec']
    
    # Define parameters for the psychometric function used in the simulation.
    sim['alpha'] = 1.1729
    sim['beta']  = 1.2286
    # Define the Weibull psychometric function.
    WeibullFunc = lambda x: (1 - 0.5*np.exp(- (x/sim['alpha'])** sim['beta']))
    # Calculate the probability of correct response given alpha and beta.
    sim['pC_given_alpha_beta'] = WeibullFunc(stim['deltaE_1JND'])
    
    #%% SIMULATION STARTS HERE
    # Initialize arrays to store the RGB and Lab compositions of comparison stimuli,
    # their color differences (deltaE) from a reference, probabilities of correct
    # identification (probC), and binary responses (resp_binary) for each simulation.
    sim['rgb_comp'] = np.full((stim['nGridPts_ref'],\
                             stim['nGridPts_ref'],3, sim['nSims']),np.nan)
    sim['lab_comp'] = np.full(sim['rgb_comp'].shape, np.nan)
    sim['deltaE'] = np.full((stim['nGridPts_ref'], stim['nGridPts_ref'],\
                             sim['nSims']), np.nan)
    sim['probC'] =  np.full(sim['deltaE'].shape, np.nan)
    sim['resp_binary'] = np.full(sim['deltaE'].shape, np.nan)
    
    # Iterate over the grid points of the reference stimulus.
    for i in range(stim['nGridPts_ref']):
        for j in range(stim['nGridPts_ref']):
            # Extract the reference stimulus' RGB values for the current grid point.
            rgb_ref_ij = sim['ref_points'][:,i,j]
            # Convert the reference RGB values to Lab color space.
            ref_Lab_ij, _, _ = convert_rgb_lab(param['B_monitor'],\
                                                sim['background_RGB'],\
                                                rgb_ref_ij)
            
            # Generate the comparison stimulus based on the sampling method.
            if sim['method_sampling'] == 'NearContour':
                # If 'NearContour', use ellipsoidal parameters to generate 
                #comparison stimuli.
                ellPara = results['ellParams'][sim['slc_RGBplane']][i,j,:]
                sim['rgb_comp'][i,j,:,:] = sample_rgb_comp_2DNearContour(\
                    rgb_ref_ij[sim['varying_RGBplane']], sim['varying_RGBplane'],
                    sim['fixed_RGBvec'], sim['nSims'], ellPara, sim['random_jitter'])
            elif sim['method_sampling'] == 'Random':
                # If 'Random', generate comparison stimuli within a specified 
                #square range.
                sim['rgb_comp'][i,j,:,:] = sample_rgb_comp_2Drandom(\
                    rgb_ref_ij[sim['varying_RGBplane']],sim['varying_RGBplane'],\
                    sim['fixed_RGBvec'], sim['range_randomSampling'], sim['nSims'])
                
            # For each simulation, calculate color difference, probability of 
            #correct identification, and simulate binary responses based on the 
            #probability.
            for n in range(sim['nSims']):
                # Convert the comparison RGB values to Lab color space.
                sim['lab_comp'][i,j,:,n], _, _ = convert_rgb_lab(param['B_monitor'],\
                    sim['background_RGB'], sim['rgb_comp'][i,j,:,n])
                # Calculate the color difference (deltaE) between the 
                #comparison and reference stimuli.
                sim['deltaE'][i,j,n] = np.linalg.norm(sim['lab_comp'][i,j,:,n] - \
                                                      ref_Lab_ij)
                # Calculate the probability of correct identification using the 
                #Weibull function.
                sim['probC'][i,j,n] = WeibullFunc(sim['deltaE'][i,j,n])
            # Simulate binary responses (0 or 1) based on the calculated probabilities.
            sim['resp_binary'][i,j,:] = np.random.binomial(1, sim['probC'][i,j,:],\
                                                           (sim['nSims'],))
                
                
    #%% visualize the samples and save the data
    plot_2D_sampledComp(stim['grid_ref'], stim['grid_ref'], sim['rgb_comp'],\
                        sim['varying_RGBplane'], sim['method_sampling'], \
                        responses = sim['resp_binary'], \
                        groundTruth = results['fitEllipse_unscaled'][sim['slc_RGBplane']],\
                        x_label = plt_specifics['subTitles'][sim['slc_RGBplane']][0],\
                        y_label=plt_specifics['subTitles'][sim['slc_RGBplane']][1])  
    
    #save to pkl
    file_name_firsthalf = 'Sims_isothreshold_'+plt_specifics['subTitles'][sim['slc_RGBplane']] +\
        '_sim' + str(sim['nSims']) + 'perCond_sampling'+sim['method_sampling']
    if sim['method_sampling'] == 'NearContour':
        file_name = file_name_firsthalf+'_jitter'+str(sim['random_jitter'])+'.pkl'
    elif sim['method_sampling'] == 'Random':
        file_name = file_name_firsthalf+'_range'+str(sim['range_randomSampling'])+'.pkl'        
    path_output = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/ELPS_analysis/Simulation_DataFiles/'
    full_path = f"{path_output}{file_name}"
        
    # Write the list of dictionaries to a file using pickle
    with open(full_path, 'wb') as f:
        pickle.dump([sim], f)

if __name__ == "__main__":
    main()
        
        
        
    
    
    
