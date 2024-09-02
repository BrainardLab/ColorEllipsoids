#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 21:12:36 2024

@author: fangfang
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids")
from plotting.wishart_plotting import WishartModelBasicsVisualization 

#%%
class TrialPlacementVisualization(WishartModelBasicsVisualization):
    def __init__(self, sim_trial, fig_dir='', save_fig=False, save_gif=False):

        super().__init__(fig_dir, save_fig, save_gif)
        self.sim_trial = sim_trial
        
    def plot_transformation(self, ell0, ell1, ell2, ell_final, ax = None, **kwargs):
        #default values for optional parameters
        method_specific_settings = {
            'figsize':(8,2),
            'visualize_primaries': True,
            'lim_scaler': 1.25,
            'facecolor':np.array([0.5, 0.5, 0.5]),
            'edgecolor':np.array([1,1,1]),
            'lw':2,
            'fontsize':10,
            'fig_name':'Monitor_primaries',
            }
        
        # Update plot parameters with method-specific settings and external configurations.
        self.pltP.update(method_specific_settings)
        self.pltP.update(kwargs) 
        plt.rcParams['font.sans-serif'] = ['Arial']
        # Set default font size for all elements
        ell = [ell0, ell1, ell2, ell_final]
        
        # Create a new figure and axes if not provided.
        if ax is None:
            fig, ax =  plt.subplots(1, 4, dpi = self.pltP['dpi'], figsize = self.pltP['figsize'])
        else:
            fig = ax.figure
            
        for i in range(4):
            ax[i].scatter(ell[i][0], ell[i][1], facecolor = self.pltP['facecolor'],
                          edgecolor = self.pltP['edgecolor'])
            lim_val = np.max(np.abs(ell[i] - np.mean(ell[i]))) *self.pltP['lim_scaler']
            self._update_axes_limits(ax[i], np.array([-lim_val, lim_val]) + np.mean(ell[i]))
            ax[i].grid(True, alpha=0.5)
            ax[i].set_aspect('equal')
        plt.tight_layout()
        plt.show()
        
        
    def plot_2D_sampledComp(self, **kwargs):
        grid_ref_x = self.sim_trial.sim['grid_ref']
        grid_ref_y = self.sim_trial.sim['grid_ref']
        
        method_specific_settings = {
            'slc_x_grid_ref': np.arange(len(grid_ref_x)),
            'slc_y_grid_ref': np.arange(len(grid_ref_y)),
            'ground_truth': None,
            'resp':None,
            'xbds':[-0.025, 0.025],
            'ybds':[-0.025, 0.025],
            'x_label':'',
            'y_label':'',
            'nFinerGrid': 50,
            'lc': np.array([178, 34, 34]) / 255,
            'WishartEllipsesColor': np.array([76, 153, 0]) / 255,
            'm1':'.',
            'm0':'*',
            'ms':5,
            'lw': 1,
            'alpha':0.8,
            'mc1': np.array([173, 216, 230]) / 255,
            'mc0': np.array([255, 179, 138]) / 255,
            'fontsize': 10,
            'fig_name': 'Sampled comparison stimuli'
            }
        
        # Update plot parameters with method-specific settings and external configurations.
        self.pltP.update(method_specific_settings)
        self.pltP.update(kwargs) 
        plt.rcParams['font.sans-serif'] = ['Arial']
        # Set default font size for all elements
        plt.rcParams.update({'font.size': self.pltP['fontsize']})
        
        nGrid_x = len(grid_ref_x)
        nGrid_y = len(grid_ref_y)
        
        idx = self.sim['varying_RGBplane']
        
        plt.figure(figsize = (8,8))
        for i in range(nGrid_x):
            for j in range(nGrid_y):
                x_axis = np.linspace(*self.pltP['xbds'], self.pltP['nFinerGrid']) + grid_ref_x[j]
                y_axis = np.linspace(*self.pltP['ybds'], self.pltP['nFinerGrid']) + grid_ref_y[i]    
                
                #subplot
                plt.subplot(nGrid_x, nGrid_y, (nGrid_x-i-1)*nGrid_y + j + 1)
                
                #plot the ground truth
                if self.pltP['ground_truth'] is not None:
                    plt.plot(self.pltP['ground_truth'][i,j,0],
                             self.pltP['ground_truth'][i,j,1],
                             color=self.pltP['lc'],
                             linestyle = '--', 
                             linewidth = self.pltP['lw'])
                
                #find indices that correspond to a response of 1 / 0
                idx_1 = np.where(self.pltP['resp'][i,j] == 1)
                idx_0 = np.where(self.pltP['resp'][i,j] == 0)
                plt.scatter(self.sim_trial.sim['rgb_comp'][i, j, idx[0], idx_1],
                            self.sim_trial.sim['rgb_comp'][i, j, idx[1], idx_1],
                            s = self.pltP['ms'], 
                            marker=self.pltP['m1'],
                            c=self.pltP['mc1'],
                            alpha= self.pltP['alpha'])
                    
                plt.scatter(self.sim_trial['rgb_comp'][i, j, idx[0], idx_0], 
                            self.sim_trial['rgb_comp'][i, j, idx[1], idx_0], 
                            s = self.pltP['ms'], 
                            marker=self.pltP['m0'], 
                            c=self.pltP['mc0'],
                            alpha= self.pltP['alpha'])
                
                plt.xlim([x_axis[0], x_axis[-1]])
                plt.ylim([y_axis[0], y_axis[-1]])
                if i == 0 and j == nGrid_y//2: plt.xlabel(self.pltP['x_label'])
                if i == nGrid_x//2 and j == 0: plt.ylabel(self.pltP['y_label'])
                
                if j == 0: plt.yticks(np.round([grid_ref_y[i]],2))
                else: plt.yticks([])
                
                if i == 0: plt.xticks(np.round([grid_ref_x[j]],2))
                else: plt.xticks([])
            
        plt.subplots_adjust(wspace = 0, hspace = 0)
        plt.tight_layout()
        plt.show()
        