#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 11:15:34 2024

@author: fangfang
"""

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from plotting.wishart_plotting import WishartModelBasicsVisualization 
from core import viz  

#%%
class CIELabVisualization(WishartModelBasicsVisualization):
    def __init__(self, sim_CIE, fig_dir='', save_fig=False, save_gif=False):
        """


        """
        super().__init__(fig_dir, save_fig, save_gif)
        self.sim_CIE = sim_CIE

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
            'data_ms':40,
            'data_mc':[0.7,0.7,0.7],
            'ticks': np.linspace(0,1,5),
            'fontsize':15,
            'figName':'Isothreshold_contour',
            }
        
        # Update plot parameters with method-specific settings and external configurations.
        self.pltP.update(method_specific_settings)
        self.pltP.update(kwargs)  

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
                ax[p].imshow(self.pltP['rgb_background'][p], extent = [0,1,0,1])
            
            #Ground truth
            for i in range(num_grid_pts_x):
                for j in range(num_grid_pts_y):
                    
                    #reference location 
                    ax[p].scatter(*grid_est[i,j],s = self.pltP['ref_ms'],
                                  c = self.pltP['ref_mc'],
                                  marker ='+',linewidth = self.pltP['ref_lw'])
                    
                    #ellipses
                    ax[p].plot(*fitEllipse[p,i,j],
                              linestyle = self.pltP['ell_ls'],\
                              color = self.pltP['ell_lc'],\
                              linewidth = self.pltP['ell_lw'])
                        
                    #individual ellipse
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
        plt.show()
        return fig, ax