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
        
    def plot_WeibullPMF(self, x, ax = None, **kwargs):
        #default values for optional parameters
        method_specific_settings = {
            'figsize':(2.2,2.2),
            'xlabel': r'Perceptual difference ($\Delta E$)',
            'ylabel': 'Percent correct',
            'fontsize':10,
            'fig_name':'Weibull_PMF.pdf',
            }
        
        # Update plot parameters with method-specific settings and external configurations.
        self.pltP.update(method_specific_settings)
        self.pltP.update(kwargs) 
        plt.rcParams['font.sans-serif'] = ['Arial']
        
        a, b = self.sim_trial.sim['alpha'], self.sim_trial.sim['beta']
        g = self.sim_trial.sim['guessing_rate']
        y = self.sim_trial.WeibullFunc(x, a, b, g)
        # Create a new figure and axes if not provided.
        if ax is None:
            fig, ax =  plt.subplots(1, 1, dpi = self.pltP['dpi'], figsize = self.pltP['figsize'])
        else:
            fig = ax.figure
        
        ax.plot(x, y, color = 'k')
        ax.set_xticks(list(range(4)))
        ax.set_yticks(np.round(np.array([1/3, 2/3, 1]),2))
        ax.set_xlabel(self.pltP['xlabel'])
        ax.set_ylabel(self.pltP['ylabel'])
        ax.grid(True, alpha=0.5)
        plt.tight_layout()
        # Save the plot with bbox_inches='tight' to ensure labels are not cropped
        if self.fig_dir and self.save_fig:
            plt.savefig(self.fig_dir + self.pltP['fig_name'])
        plt.show()
        return fig, ax  

    def plot_transformation(self, ell0, ell1, ell2, ell_final, resp, gt,
                            ax = None, **kwargs):
        #default values for optional parameters
        method_specific_settings = {
            'figsize':(9,3),
            'colorcode_resp': False,
            'visualize_gt': True,
            'lim_scaler': 1.25,
            'facecolor':np.array([0.5, 0.5, 0.5]),
            'edgecolor':np.array([1,1,1]),
            'facecolor_yes':np.array([107,142,35])/255,
            'facecolor_no':np.array([178,34,34])/255,
            'alpha':1,
            'xlim':[[]]*4,
            'ylim':[[]]*4,
            'ms':25,
            'xlabel': 'dim 1',
            'ylabel': 'dim 2',
            'fontsize':10,
            'fig_name':'samples_transformation',
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
            mean_val = np.mean(gt[i], axis = 1)
            if self.pltP['visualize_gt']:
                ax[i].plot(gt[i][0], gt[i][1], color = 'k', alpha = 0.35, lw = 2)
                #plot the center
                ax[i].scatter(mean_val[0], mean_val[1],
                              marker = '+', color = 'k', s = 30, lw = 1)
            if self.pltP['colorcode_resp']:
                print(i)
                idx_yes = np.where(resp == 1)[0]
                idx_no  = np.where(resp == 0)[0]
                ell_yes = ell[i][:,idx_yes]
                ell_no  = ell[i][:,idx_no]
                ax[i].scatter(ell_yes[0], ell_yes[1], marker = 'o',
                              facecolor = self.pltP['facecolor_yes'],
                              edgecolor = self.pltP['edgecolor'],
                              s = self.pltP['ms'],
                              alpha = self.pltP['alpha'])
                ax[i].scatter(ell_no[0], ell_no[1], marker = 'o',
                              facecolor = self.pltP['facecolor_no'],
                              edgecolor = self.pltP['edgecolor'],
                              s = self.pltP['ms'],
                              alpha = self.pltP['alpha'])
            else:
                ax[i].scatter(ell[i][0], ell[i][1], 
                              facecolor = self.pltP['facecolor'],
                              edgecolor = self.pltP['edgecolor'],
                              s = self.pltP['ms'],
                              alpha = self.pltP['alpha'])
            lim_val = np.max(np.abs(ell[i] -  mean_val[:,None])) *self.pltP['lim_scaler']
            if len(self.pltP['xlim'][i]) == 0 or len(self.pltP['ylim'][i]) == 0:
                ax[i].set_xlim(np.array([-lim_val, lim_val] + mean_val[0]))
                ax[i].set_ylim(np.array([-lim_val, lim_val] + mean_val[1]))
                print(np.array([-lim_val, lim_val] + mean_val[0]))
                print(np.array([-lim_val, lim_val] + mean_val[1]))
            else:
                ax[i].set_xlim(self.pltP['xlim'][i])
                ax[i].set_ylim(self.pltP['ylim'][i])
            ax[i].set_xlabel(self.pltP['xlabel'])
            ax[i].set_ylabel(self.pltP['ylabel'])
            ax[i].grid(True, alpha=0.5)
            ax[i].set_aspect('equal')
        plt.tight_layout()
        # Save the plot with bbox_inches='tight' to ensure labels are not cropped
        if self.fig_dir and self.save_fig:
            plt.savefig(self.fig_dir + self.pltP['fig_name'])
        plt.show()
        return fig, ax       
        
    def plot_2D_sampledComp(self, **kwargs):
        sim = self.sim_trial.sim
        grid_ref_x = sim['grid_ref']
        grid_ref_y = sim['grid_ref']
        
        method_specific_settings = {
            'slc_x_grid_ref': np.arange(len(grid_ref_x)),
            'slc_y_grid_ref': np.arange(len(grid_ref_y)),
            'ground_truth': None,
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
                idx_1 = np.where(sim['resp_binary'][i,j] == 1)
                idx_0 = np.where(sim['resp_binary'][i,j] == 0)
                plt.scatter(sim['rgb_comp'][i, j, sim['varying_RGBplane'][0], idx_1],
                            sim['rgb_comp'][i, j, sim['varying_RGBplane'][1], idx_1],
                            s = self.pltP['ms'], 
                            marker=self.pltP['m1'],
                            c=self.pltP['mc1'],
                            alpha= self.pltP['alpha'])
                    
                plt.scatter(sim['rgb_comp'][i, j, sim['varying_RGBplane'][0], idx_0], 
                            sim['rgb_comp'][i, j, sim['varying_RGBplane'][1], idx_0], 
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
        