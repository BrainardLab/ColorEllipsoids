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
        
        pltP = {
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
            'ls':'--',
            'alpha':0.8,
            'mc1': np.array([173, 216, 230]) / 255,
            'mc0': np.array([255, 179, 138]) / 255,
            'fontsize': 10,
            'fig_name': 'Sampled comparison stimuli'
            }
        
        # Update plot parameters with method-specific settings and external configurations.
        pltP.update(kwargs) 
        plt.rcParams['font.sans-serif'] = ['Arial']
        # Set default font size for all elements
        plt.rcParams.update({'font.size': pltP['fontsize']})
        
        nGrid_x = len(grid_ref_x)
        nGrid_y = len(grid_ref_y)
        
        #subplot
        fig, ax = plt.subplots(nGrid_x, nGrid_y, figsize = (8,8), dpi = 1024)
        #fig, ax = plt.subplots(1,1, figsize = (8,8), dpi = 1024)
        for i in range(nGrid_x):
            for j in range(nGrid_y):
                # Access the subplot from bottom to top
                ax_idx = ax[nGrid_x - 1 - i, j]  # Reverse row order

                x_axis = np.linspace(*pltP['xbds'], pltP['nFinerGrid']) + grid_ref_x[j]
                y_axis = np.linspace(*pltP['ybds'], pltP['nFinerGrid']) + grid_ref_y[i]    
                
                #plot the ground truth
                if pltP['ground_truth'] is not None:
                    ax_idx.plot(pltP['ground_truth'][i,j,0],
                             pltP['ground_truth'][i,j,1],
                             color= pltP['lc'],
                             linestyle = pltP['ls'], 
                             linewidth = pltP['lw'])
                
                #find indices that correspond to a response of 1 / 0
                idx_1 = np.where(sim['resp_binary'][i,j] == 1)
                idx_0 = np.where(sim['resp_binary'][i,j] == 0)
                ax_idx.scatter(sim['rgb_comp'][i, j, sim['varying_RGBplane'][0], idx_1],
                            sim['rgb_comp'][i, j, sim['varying_RGBplane'][1], idx_1],
                            s = pltP['ms'], 
                            marker= pltP['m1'],
                            c= pltP['mc1'],
                            alpha= pltP['alpha'])
                    
                ax_idx.scatter(sim['rgb_comp'][i, j, sim['varying_RGBplane'][0], idx_0], 
                            sim['rgb_comp'][i, j, sim['varying_RGBplane'][1], idx_0], 
                            s = pltP['ms'], 
                            marker= pltP['m0'], 
                            c= pltP['mc0'],
                            alpha= pltP['alpha'])
                
                ax_idx.set_xlim([x_axis[0], x_axis[-1]])
                ax_idx.set_ylim([y_axis[0], y_axis[-1]])
                if i == 0 and j == nGrid_y//2: ax_idx.set_xlabel(pltP['x_label'])
                if i == nGrid_x//2 and j == 0: ax_idx.set_ylabel(pltP['y_label'])
                
                if j == 0: ax_idx.set_yticks([grid_ref_y[i]])
                else: ax_idx.set_yticks([])
                
                if i == 0: ax_idx.set_xticks([grid_ref_x[j]])
                else: ax_idx.set_xticks([])
            
        plt.subplots_adjust(wspace = 0, hspace = 0)
        plt.tight_layout()
        plt.show()
        if self.save_fig and self.fig_dir != '':
            full_path2 = f"{self.fig_dir}/{pltP['fig_name']}.pdf"
            fig.savefig(full_path2) 
        return fig, ax
        
    @staticmethod
    def plot_3D_sampledComp(ref_points, fitEllipsoid_unscaled, sampledComp,
                            fixedPlane, fixedPlaneVal, nPhi = 100, nTheta = 200, 
                            **kwargs):
        # Default parameters for ellipsoid fitting. Can be overridden by kwargs.
        pltP = {
            'visualize_ellipsoid':True, 
            'visualize_samples':True, 
            'scaled_neg12pos1':False,
            'slc_grid_ref_dim1': list(range(5)),
            'slc_grid_ref_dim2': list(range(5)),
            'surf_alpha': 0.3,
            'samples_alpha': 0.2,
            'markerSize_samples':2,
            'default_viewing_angle':False,
            'bds': 0.025,
            'fontsize':15,
            'figsize':(8,8),
            'title':'',
            'saveFig':False,
            'figDir':'',
            'figName':'Sampled_comparison_stimuli_3D'
            }
        pltP.update(kwargs)
        
        #Determine the indices of the reference points based on the fixed 
        # plane specified ('R', 'G', or 'B' for different color channels)
        if fixedPlane =='R':
            idx_x = np.where(np.abs(ref_points - fixedPlaneVal)< 1e-6)[0][0]
        elif fixedPlane == 'G':
            idx_y = np.where(np.abs(ref_points - fixedPlaneVal)< 1e-6)[0][0]
        elif fixedPlane == 'B':
            idx_z = np.where(np.abs(ref_points - fixedPlaneVal)< 1e-6)[0][0]
        else:
            return
        
        nGridPts_dim1 = len(pltP['slc_grid_ref_dim1'])
        nGridPts_dim2 = len(pltP['slc_grid_ref_dim2'])
        ref_points_idx = np.array(list(range(len(ref_points))))
        
        fig, axs = plt.subplots(nGridPts_dim2, nGridPts_dim1, subplot_kw={'projection': '3d'}, \
                                figsize=pltP['figsize'])
        for j in range(nGridPts_dim2-1,-1,-1):
            jj = ref_points_idx[pltP['slc_grid_ref_dim2'][nGridPts_dim2-j-1]]
            for i in range(nGridPts_dim1):
                ii = ref_points_idx[pltP['slc_grid_ref_dim1'][i]]
                
                ax = axs[j, i]
                base_shape = (nPhi, nTheta)
                if fixedPlane == 'R':
                    slc_ref = np.array([fixedPlaneVal, ref_points[ii], ref_points[jj]])
                    slc_gt = fitEllipsoid_unscaled[idx_x, ii,jj,:,:]
                    slc_rgb_comp = sampledComp[idx_x, ii,jj,:,:]
                elif fixedPlane == 'G':
                    slc_ref = np.array([ref_points[ii], fixedPlaneVal, ref_points[jj]])
                    slc_gt = fitEllipsoid_unscaled[ii,idx_y,jj,:,:]
                    slc_rgb_comp = sampledComp[ii,idx_y,jj,:,:]
                elif fixedPlane == 'B':
                    slc_ref = np.array([ref_points[ii], ref_points[jj], fixedPlaneVal])
                    slc_gt = fitEllipsoid_unscaled[ii,jj,idx_z,:,:]
                    slc_rgb_comp = sampledComp[ii,jj,idx_z,:,:]
                slc_gt_x = np.reshape(slc_gt[0,:], base_shape)
                slc_gt_y = np.reshape(slc_gt[1,:], base_shape)
                slc_gt_z = np.reshape(slc_gt[2,:], base_shape)
                       
                #subplot
                if pltP['visualize_ellipsoid']:
                    if pltP['scaled_neg12pos1']: color_v = (slc_ref+1)/2
                    else: color_v = slc_ref
                    ax.plot_surface(slc_gt_x, slc_gt_y, slc_gt_z, \
                        color=color_v, edgecolor='none', alpha=0.5)
                    
                if pltP['visualize_samples']:
                    ax.scatter(slc_rgb_comp[0,:], slc_rgb_comp[1,:], slc_rgb_comp[2,:],\
                               s=pltP['markerSize_samples'], c= [0,0,0],
                               alpha=pltP['samples_alpha'])
                        
                ax.set_xlim(slc_ref[0]+np.array(pltP['bds']*np.array([-1,1]))); 
                ax.set_ylim(slc_ref[1]+np.array(pltP['bds']*np.array([-1,1])));  
                ax.set_zlim(slc_ref[2]+np.array(pltP['bds']*np.array([-1,1])));  
                ax.set_xlabel('');ax.set_ylabel('');ax.set_zlabel('');
                #set tick marks
                if fixedPlane == 'R':
                    ax.set_xticks([]); 
                else:
                    ax.set_xticks(np.round(slc_ref[0]+\
                        np.array(np.ceil(pltP['bds']*100)/100*\
                        np.array([-1,0,1])),2))
                    
                if fixedPlane == 'G':
                    ax.set_yticks([]); 
                else:
                    ax.set_yticks(np.round(slc_ref[1]+\
                        np.array(np.ceil(pltP['bds']*100)/100*\
                        np.array([-1,0,1])),2))
                    
                if fixedPlane == 'B':
                    ax.set_zticks([]);
                else:
                    ax.set_zticks(np.round(slc_ref[2]+\
                        np.array(np.ceil(pltP['bds']*100)/100*\
                        np.array([-1,0,1])),2))
                # Adjust viewing angle for better visualization
                if not pltP['default_viewing_angle']:
                    if fixedPlane == 'R': ax.view_init(0,0)
                    elif fixedPlane == 'G': ax.view_init(0,-90)
                    elif fixedPlane == 'B': ax.view_init(90,-90)
                else:
                    ax.view_init(30,-37.5)
                ax.grid(True)
                ax.set_aspect('equal')
        fig.suptitle(pltP['title'])
        plt.tight_layout()
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05,\
                            wspace=-0.05, hspace=-0.05)
        plt.show()
        if pltP['saveFig'] and pltP['figDir'] != '':
            full_path2 = f"{pltP['figDir']}/{pltP['figName']}"
            fig.savefig(full_path2)            