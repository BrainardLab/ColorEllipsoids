#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 15:09:10 2024

@author: fangfang
"""
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

class TrialPlacementWithoutAdaptiveSampling:
    def __init__(self):
        self.sim = self._query_simCondition()
        #Identify the fixed RGB dimension by excluding the varying dimensions.
        self.varying_RGBplane = set(range(3))
        self.varying_RGBplane.remove(self.sim['slc_RGBplane'])
        
    def _query_simCondition(self, default_value = 'R', default_str = 'NearContour',
                           default_jitter = 0.1, default_ub = 0.025, default_trialNum = 80):
        sim = {}
        # QUESTION 1: Ask the user which RGB plane to fix during the simulation.
        # The default plane is 'R'.
        slc_RGBplane = input('Which plane would you like to fix (R/G/B/[]); default: R. If you are simulating ellipsoids, please enter []:')
        
        # Validate the user input for the RGB plane. If valid, store the index; 
        #otherwise, use the default value.
        RGBplane = 'RGB'
        if len(slc_RGBplane) == 1:
            sim['slc_RGBplane'] = RGBplane.find(slc_RGBplane)
        elif len(slc_RGBplane) == 2:
            sim['slc_RGBplane'] = list(range(3))
        else:
            sim['slc_RGBplane'] = RGBplane.find(default_value)
        
        # QUESTION 2: Ask the user to choose the sampling method.
        # The default method is 'NearContour'.
        sim['method_sampling'] = input('Which sampling method (NearContour/Random/'+\
                                       'Gaussian; default: NearContour):')
        
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
        
        return sim
    
    def sample_rgb_comp_2DNearContour(self, rgb_ref, slc_fixedVal,
                                      paramEllipse):
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
        - slc_fixedVal (float): The fixed value for the RGB dimension not included 
            in `varying_RGBplane`. This value remains constant across all simulations.
        - paramEllipse (array): Parameters defining the ellipsoid contour. 
            Includes the center coordinates in the varying dimensions, the lengths 
            of the semi-axes of the ellipsoid in the plane of variation, and the 
            rotation angle of the ellipsoid in degrees.
            Expected format: [xc, yc, semi_axis1_length, semi_axis2_length, rotation_angle].
    
        Returns
        -------
        - rgb_comp_sim (array): A 3xN array of simulated RGB compositions, 
            where N is the number of simulations (`nSims`). Each column represents 
            an RGB composition near the specified ellipsoidal contour in RGB color 
            space. The row order corresponds to R, G, and B dimensions, respectively.
    
    
        """
        
        #Initialize the output matrix with nans
        rgb_comp_sim = np.full((3, self.sim['nSims']), np.nan)
        
        #Generate random angles to simulate points around the ellipse.
        randTheta = np.random.rand(1, self.sim['nSims']) * 2 * np.pi
        
        #calculate x and y coordinates with added jitter
        randx = np.cos(randTheta) + np.random.randn(1, self.sim['nSims']) * self.sim['random_jitter']
        randy = np.sin(randTheta) + np.random.randn(1, self.sim['nSims']) * self.sim['random_jitter']
        
        #adjust coordinates based on the ellipsoid's semi-axis lengths
        randx_stretched = randx * paramEllipse[2]
        randy_stretched = randy * paramEllipse[3]
        
        #calculate the varying RGB dimensions, applying rotation and translation 
        #based on the reference RGB values and ellipsoid parameters
        rgb_comp_sim[self.varying_RGBplane[0],:] = \
            randx_stretched * np.cos(np.deg2rad(paramEllipse[-1])) - \
            randy_stretched * np.sin(np.deg2rad(paramEllipse[-1])) + rgb_ref[0]
        rgb_comp_sim[self.varying_RGBplane[1],:] = \
            randx_stretched * np.sin(np.deg2rad(paramEllipse[-1])) + \
            randy_stretched * np.cos(np.deg2rad(paramEllipse[-1])) + rgb_ref[1]
            
        #set the fixed RGB dimension to the specificed fixed value for all simulations
        rgb_comp_sim[self.sim['slc_RGBplane'],:] = slc_fixedVal;
        
        return rgb_comp_sim
    
    def sample_rgb_comp_random(self, rgb_ref, slc_fixedVal,
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
        
        rgb_comp_sim = np.random.rand(3, nSims) * (box_range[1] - box_range[0]) + \
            box_range[0]
            
        if len(self.sim['slc_RGBplane']) != 0:
            rgb_comp_sim[self.sim['slc_RGBplane'],:] = slc_fixedVal
        
        rgb_comp_sim[self.varying_RGBplane,:] = rgb_comp_sim[self.varying_RGBplane,:] + \
            rgb_ref.reshape((len(self.varying_RGBplane),1))
            
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
            'markerSize':5,
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
                            s = pltParams['markerSize'], marker=pltParams['marker1'],\
                            c=pltParams['markerColor1'],alpha=0.8)
                    
                plt.scatter(rgb_comp[i, j, varying_RGBplane[0], idx_0], 
                            rgb_comp[i, j, varying_RGBplane[1], idx_0], \
                            s = pltParams['markerSize'], marker=pltParams['marker0'], \
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
        