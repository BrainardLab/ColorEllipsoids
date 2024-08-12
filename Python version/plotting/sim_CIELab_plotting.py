#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 11:15:34 2024

@author: fangfang
"""

import numpy as np
import matplotlib.pyplot as plt
from plotting.wishart_plotting import WishartModelBasicsVisualization 

#%%
class CIELabVisualization(WishartModelBasicsVisualization):
    def __init__(self, sim_CIE, fig_dir='', save_fig=False, save_gif=False):

        super().__init__(fig_dir, save_fig, save_gif)
        self.sim_CIE = sim_CIE
    
    def plot_primaries(self, rgb = None, ax = None, **kwargs):
        #default values for optional parameters
        method_specific_settings = {
            'figsize':(2,2),
            'visualize_primaries': True,
            'cmap': np.array([[178,34,34], [0, 100,0],[0,0,128]])/255,
            'ls':[':','-','--'],
            'ylim': [],
            'lw':2,
            'fontsize':10,
            'fig_name':'Monitor_primaries',
            }
        
        # Update plot parameters with method-specific settings and external configurations.
        self.pltP.update(method_specific_settings)
        self.pltP.update(kwargs) 
        
        # Create a new figure and axes if not provided.
        if ax is None:
            fig, ax =  plt.subplots(1, 1, dpi = self.pltP['dpi'], figsize = self.pltP['figsize'])
        else:
            fig = ax.figure
            
        if self.pltP['visualize_primaries']:
            for i in range(self.sim_CIE.nPlanes):
                ax.plot(self.sim_CIE.B_MONITOR[:,i],
                        c = self.pltP['cmap'][i],
                        lw = self.pltP['lw'])
        if rgb is not None:
            for j in range(rgb.shape[1]):
                spd_j = self.sim_CIE.B_MONITOR @ rgb[:,j]
                ax.plot(spd_j, c= 'k', 
                        linestyle = self.pltP['ls'][j],
                        lw = self.pltP['lw'])
            if len(self.pltP['ylim']) != 0:
                ax.set_ylim(self.pltP['ylim'])
            ax.set_xticks([])
            ax.set_yticks([])
        # Save the plot with bbox_inches='tight' to ensure labels are not cropped
        if len(self.fig_dir) !=0 and self.save_fig:
            plt.savefig(self.fig_dir + self.pltP['fig_name'])
        return fig, ax
    
        
    def plot_Tcones(self, ax = None, **kwargs):
        #default values for optional parameters
        method_specific_settings = {
            'cmap': np.array([[178,34,34], [0, 100,0],[0,0,128]])/255,
            'ylim': [],
            'fontsize':10,
            'fig_name':'T_cones',
            }
        
        # Update plot parameters with method-specific settings and external configurations.
        self.pltP.update(method_specific_settings)
        self.pltP.update(kwargs) 
        
        # Create a new figure and axes if not provided.
        if ax is None:
            fig, ax =  plt.subplots(1, 1, dpi = self.pltP['dpi'], figsize = (2,2))
        else:
            fig = ax.figure

        for i in range(self.sim_CIE.nPlanes):
            ax.plot(self.sim_CIE.T_CONES[i], c = self.pltP['cmap'][i])
            if len(self.pltP['ylim']) != 0:
                ax.set_ylim(self.pltP['ylim'])
            ax.set_xticks([])
            ax.set_yticks([])
        # Save the plot with bbox_inches='tight' to ensure labels are not cropped
        if len(self.fig_dir) !=0 and self.save_fig:
            plt.savefig(self.fig_dir + self.pltP['fig_name'])
        return fig, ax
    
    def plot_RGB_to_LAB(self, ref_rgb, ref_lab, ax = None, **kwargs):
        #default values for optional parameters
        method_specific_settings = {
            'visualize_raw_data': False,
            'rgb_lim': [0,1],
            'rgb_ticks':np.linspace(0.2,0.8,3),
            'lab_viewing_angle': [30,-25],
            'lab_ticks':[-60,0,60],
            'lab_lim_margin': 10,
            'fontsize':10,
            'fig_name':'Isothreshold_contour',
            }
        
        # Update plot parameters with method-specific settings and external configurations.
        self.pltP.update(method_specific_settings)
        self.pltP.update(kwargs)  
        self.ndims = 3
        plt.rcParams['font.sans-serif'] = ['Arial']
        
        # Create a new figure and axes if not provided.
        if ax is None:
            fig, ax =  plt.subplots(2, 1, dpi = self.pltP['dpi'], 
                                    figsize = (4,10), subplot_kw={'projection': '3d'})
        else:
            fig = ax.figure
            
        #colormap
        cmap_rgb = np.moveaxis(ref_rgb, 0, -1)
        colors_flat = cmap_rgb.reshape(-1, 3)
            
        #RGB space
        r_slc, g_slc, b_slc = ref_rgb
        r_slc_f = r_slc.flatten()
        g_slc_f = g_slc.flatten()
        b_slc_f = b_slc.flatten()
        ax[0].scatter(r_slc_f, g_slc_f, b_slc_f, c = colors_flat)
        self._update_axes_limits(ax[0], lim = self.pltP['rgb_lim'])
        self._update_axes_labels(ax[0], self.pltP['rgb_ticks'], self.pltP['rgb_ticks'],nsteps =1)
        ax[0].set_xlabel('R'); ax[0].set_ylabel('G'); ax[0].set_zlabel('B')
        ax[0].set_box_aspect([1,1,1])
        ax[0].set_title('RGB space')

        #CIELAB SPACE
        L_slc, A_slc, B_slc = ref_lab
        L_slc_f = L_slc.flatten()
        A_slc_f = A_slc.flatten()
        B_slc_f = B_slc.flatten()
        ax[1].scatter(A_slc_f, B_slc_f, L_slc_f, c = colors_flat, 
                   marker = 'o',s=5, edgecolor='none')

        xymin = np.min([np.min(A_slc_f), np.min(B_slc_f)])
        xymax = np.max([np.max(A_slc_f), np.max(B_slc_f)])
        xylim = np.array([-1,1])* np.max([np.abs(xymin), xymax]) + np.array([-1,1])*self.pltP['lab_lim_margin']
        zmin = np.min(L_slc_f)
        zmax = np.max(L_slc_f)
        zlim = np.array([zmin, zmax]) + np.array([-1,1])*self.pltP['lab_lim_margin']
        # Project the surface onto the XY plane (Z = min(Z))
        ax[1].scatter(A_slc_f, B_slc_f, zlim[0] * np.ones_like(B_slc_f),
                        c=colors_flat, edgecolor='none',  
                        marker = 'o',s = 2, alpha=0.05)

        # Project the surface onto the XZ plane (Y = max(Y))
        ax[1].scatter(A_slc_f, xylim[1] * np.ones_like(B_slc_f), L_slc_f,
                        c=colors_flat, edgecolor='none', 
                        marker = 'o',s= 4,  alpha=0.02)

        # Project the surface onto the YZ plane (X = min(X))
        ax[1].scatter(xylim[0] * np.ones_like(A_slc_f), B_slc_f, L_slc_f,
                        c=colors_flat, edgecolor='none', 
                        marker = 'o',s = 4,  alpha=0.02)

        ax[1].set_xlim(xylim); ax[1].set_ylim(xylim); ax[1].set_zlim(zlim)
        ax[1].set_xticks(self.pltP['lab_ticks']); ax[1].set_yticks(self.pltP['lab_ticks'])
        ax[1].set_xlabel('a'); ax[1].set_ylabel('b'); ax[1].set_zlabel('L')
        ax[1].set_box_aspect([1,1,1])
        ax[1].set_title('CIELab space')
        ax[1].view_init(*self.pltP['lab_viewing_angle'])
        # Save the plot with bbox_inches='tight' to ensure labels are not cropped
        if len(self.fig_dir) !=0 and self.save_fig:
            plt.savefig(self.fig_dir + self.pltP['fig_name'], bbox_inches='tight',
                    pad_inches=0.3)

        plt.show()
        return fig, ax

    def plot_2D(self, grid_est, fitEllipse, rawData = None, ax = None, **kwargs):
        #default values for optional parameters
        method_specific_settings = {
            'visualize_raw_data': False,
            'rgb_background': None,
            'ref_mc':[0,0,0],
            'ref_ms': 40,
            'ref_lw':2,
            'ell_lc':[0,0,0],
            'ell_ls':'-',
            'ell_lw':2,
            'data_m':'o',
            'data_alpha':1,
            'data_ms':20,
            'data_mc':[0.5,0.5,0.5],
            'ticks': np.linspace(0,1,5),
            'fontsize':15,
            'figName':'Isothreshold_contour',
            }
        
        # Update plot parameters with method-specific settings and external configurations.
        self.pltP.update(method_specific_settings)
        self.pltP.update(kwargs)  
        plt.rcParams['font.sans-serif'] = ['Arial']
        num_grid_pts_x, num_grid_pts_y = grid_est.shape[0:2]
        
        # Create a new figure and axes if not provided.
        if ax is None:
            fig, ax =  plt.subplots(1, self.sim_CIE.nPlanes,figsize=(20, 6),
                                    dpi= self.pltP['dpi'])
        else:
            fig = ax.figure
                
        for p in range(self.sim_CIE.nPlanes):
            if self.pltP['rgb_background'] is not None:
                #fill in RGB color
                ax[p].imshow(self.pltP['rgb_background'][p], extent = [0,1,0,1],
                             origin='lower')
            
            #Ground truth
            for i in range(num_grid_pts_x):
                for j in range(num_grid_pts_y):
                    
                    #reference location 
                    ax[p].scatter(*grid_est[i,j],s = self.pltP['ref_ms'],
                                  c = self.pltP['ref_mc'],
                                  marker ='+',linewidth = self.pltP['ref_lw'])
                    
                    #ellipses
                    ax[p].plot(*fitEllipse[p,i,j],
                              linestyle = self.pltP['ell_ls'],
                              color = self.pltP['ell_lc'],
                              linewidth = self.pltP['ell_lw'])
                        
                    #thresholds
                    if self.pltP['visualize_raw_data'] and rawData is not None:
                        ax[p].scatter(*rawData[p,i,j],\
                                          marker = self.pltP['data_m'],
                                          color = self.pltP['data_mc'],
                                          s = self.pltP['data_ms'],
                                          alpha = self.pltP['data_alpha'])
            self._update_axes_limits(ax[p], lim = [0,1])
            self._update_axes_labels(ax[p], self.pltP['ticks'], self.pltP['ticks'],nsteps =1)
            self.pltP['plane_2D'] = self.sim_CIE.plane_2D_list[p]
            self._configure_labels_and_title(ax[p])
        # Show the figure after all subplots have been drawn
        if len(self.fig_dir) !=0 and self.save_fig:
            plt.savefig(self.fig_dir + self.pltP['fig_name'])
        plt.show()
        return fig, ax