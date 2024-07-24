#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 17:39:59 2024

@author: fangfang
"""

import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/"+\
                "Python version/wishart_visualization/")
from wishart_plotting import wishart_model_basics_visualization   

#%%
class sampling_ref_comp_pair_visualization(wishart_model_basics_visualization):
    def __init__(self, fig_dir='', save_fig=False, save_gif=False):
        """
        Initialize an instance of sampling_ref_comp_pair_visualization, a subclass of
        wishart_model_basics_visualization, which extends its functionality for specific
        visualization tasks related to sampling reference and comparison stimulus pairs.

        """
        super().__init__(fig_dir, save_fig, save_gif)

    
    def plot_2D_sampling(self, ax, fig, xref, xcomp, idx_fixedPlane,\
                                        fixedVal, bounds, **kwargs):
        """
        This function plots the sampled pairs of reference stimulus and comparison
        stimulus in a selected 2D plane.
    
        Parameters:
        ----------
        ax : matplotlib.axes.Axes
            The axes object for the plot. This will contain the visual representation of the data.
        fig : matplotlib.figure.Figure
            The figure object that encapsulates the axes.
        xref : np.array, shape: (N, 2)
            Coordinates for the reference stimuli in the 2D plane of the RGB space.
        xcomp : np.array, shape: (N, 2)
            Coordinates for the comparison stimuli that are paired with the reference stimuli.
        idx_fixedPlane : int
            Index indicating which of the RGB dimensions is fixed (0 for R, 1 for G, 2 for B).
        fixedVal : float
            The value at which the fixed dimension is held, must be between 0 and 1.
        bounds : list of float
            The lower and upper bounds [lbd, ubd] for the stimuli values within the plane.

        """
        # Initialize method-specific settings and update with any additional configurations provided.
        method_specific_settings = {
            'visualize_bounds': True,
            'bounds_alpha':0.2,
            'linealpha':0.5,
            'ref_marker': '+',
            'ref_markersize': 20,
            'ref_markeralpha': 0.8,
            'comp_marker': 'o',
            'comp_markersize': 4,
            'comp_markeralpha': 0.8,     
            'plane_2D':'',
            'flag_rescale_axes_label':True,
            'flag_add_trialNum_title':True,
            'fontsize':8,
            'figName':'RandomSamples'}
        
        self.pltP.update(method_specific_settings)
        self.pltP.update(kwargs)  # Apply any external configurations
        # Mapping reference points to colors in RGB space, including the fixed dimension.
        cmap = (xref+1)/2
        cmap = np.insert(cmap, idx_fixedPlane, np.ones((xref.shape[0],))*fixedVal, axis=1)
        
        # Optional visualization of bounds as a grey patch on the plot.
        if self.pltP['visualize_bounds']:
            rectangle = Rectangle((bounds[0], bounds[0]), bounds[1] - bounds[0],\
                                  bounds[1] - bounds[0], facecolor='grey',\
                                  alpha= self.pltP['bounds_alpha'])  # Adjust alpha for transparency
            rectangle.set_label('Bounds for the reference')  # Set the label here
            ax.add_patch(rectangle)
        
        # Plotting the reference and comparison points.
        ax.scatter(xref[:,0],xref[:,1], c = cmap, marker = self.pltP['ref_marker'],\
                   s = self.pltP['ref_markersize'], alpha = self.pltP['ref_markeralpha'],\
                   label = 'Reference stimulus')
        ax.scatter(xcomp[:,0], xcomp[:,1], c = cmap, marker = self.pltP['comp_marker'],\
                   s = self.pltP['comp_markersize'], alpha = self.pltP['comp_markeralpha'],\
                   label = 'Comparison stimulus') 
            
        # Drawing lines connecting reference and comparison points.
        for l in range(xref.shape[0]):
            ax.plot([xref[l,0],xcomp[l,0]], [xref[l,1],xcomp[l,1]], c = cmap[l],\
                    alpha = self.pltP['linealpha'],lw = 0.5)
        
        # Configuring grid, aspect ratio, and ticks based on the plotting parameters.
        plt.grid(alpha = 0.2)
        ax.set_aspect('equal', adjustable='box')
        self._update_axes_limits(ax)
        
        # Configure tick marks for axes.
        ticks = np.sort(np.concatenate((np.linspace(-0.5, 0.5, 3), np.array([-0.85, 0.85]))))
        if self.pltP['flag_rescale_axes_label']:
            self._update_axes_labels(ax, ticks, (ticks+1)/2, nsteps = 1)
        else:
            self._update_axes_labels(ax, ticks, ticks, nsteps = 1)
        ax.tick_params(axis='both', which='major', labelsize=self.pltP['fontsize'])

        # Optionally add a trial number to the title.
        plane_2D_copy = self.pltP['plane_2D']
        if self.pltP['flag_add_trialNum_title'] :
            self.pltP['plane_2D'] += ' (n = ' +str(xref.shape[0])+')' 
        self._configure_labels_and_title(ax)
        self.pltP['plane_2D'] = plane_2D_copy
        
        # Set the legend with a custom location and show the plot.
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.47),\
                   fontsize = self.pltP['fontsize'])
        fig.tight_layout(); plt.show()
        
        # Save the figure if the directory is set and saving is enabled.
        if len(self.fig_dir) !=0 and self.save_fig:
            self._save_figure(fig, self.pltP['fig_name'])
            
            
    def plot_3D_sampling(self, ax, fig, xref, xcomp, **kwargs):
        """
        This function plots the sampled pairs of reference stimulus and comparison
        stimulus in a 3D RGB cube.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes object on which the data will be plotted.
        fig : matplotlib.figure.Figure
            The figure object that contains the axes.
        xref : np.array, shape: (N, 3)
            The reference stimulus coordinates in RGB space, where N is the number of pairs.
        xcomp : np.array, shape: (N, 3)
            The comparison (odd) stimulus coordinates, paired with the reference stimulus.

        """
        self.ndims = 3
        # Default parameters for ellipsoid fitting. Can be overridden by kwargs.
        method_specific_settings = {
            'linealpha':0.5,
            'ref_marker': '+',
            'ref_markersize': 20,
            'ref_markeralpha': 0.8,
            'comp_marker': 'o',
            'comp_markersize': 4,
            'comp_markeralpha': 0.8,        
            'fontsize':8,
            'plane_3D': 'RGB space',
            'flag_rescale_axes_label':True,
            'flag_add_trialNum_title':True,
            'fig_name':'3D_randRef_nearContourComp'} 
        self.pltP.update(method_specific_settings)
        self.pltP.update(kwargs)  # Apply any external configurations
        
        # Mapping the RGB data to a [0,1] range suitable for display as colors.
        color_map_ref = (xref + 1) / 2  
        color_map_comp = (xcomp + 1) / 2
        
        # Plotting the reference stimuli.
        ax.scatter(xref[:, 0], xref[:, 1], xref[:, 2], c=color_map_ref,\
                   marker= self.pltP['ref_marker'], s= self.pltP['ref_markersize'], \
                   alpha= self.pltP['ref_markeralpha'], label = 'Reference stimulus')
        # Plotting the comparison stimuli.
        ax.scatter(xcomp[:, 0], xcomp[:, 1], xcomp[:, 2], c=color_map_comp,\
                   marker=self.pltP['comp_marker'], s= self.pltP['comp_markersize'],\
                   alpha= self.pltP['comp_markeralpha'], label = 'Comparison stimulus')
        
        # Optionally draw lines between reference and comparison points.
        for l in range(xref.shape[0]):
            ax.plot([xref[l, 0], xcomp[l, 0]],[xref[l, 1], xcomp[l, 1]],\
                    [xref[l, 2], xcomp[l, 2]], color= np.array(color_map_ref[l]),\
                    alpha= self.pltP['linealpha'], lw= 0.5)

        # Configuring ticks and labels based on the settings.
        ticks = np.unique(xref); nsteps = 1
        if ticks.shape[0] > 5: ticks = np.linspace(-1,1,5); nsteps = 2
        if self.pltP['flag_rescale_axes_label']:
            self._update_axes_labels(ax, ticks, (ticks+1)/2, nsteps = nsteps)
        else:
            self._update_axes_labels(ax, ticks, ticks, nsteps = nsteps)
        # Setting the title and labels for the axes based on 3D plane settings.
        plane_3D_copy = self.pltP['plane_3D']
        if self.pltP['flag_add_trialNum_title'] :
            self.pltP['plane_3D'] += ' (n = ' +str(xref.shape[0])+')' 
        self._configure_labels_and_title(ax)
        self._update_axes_limits(ax)
        self.pltP['plane_3D'] = plane_3D_copy
        ax.grid(True)
        ax.set_aspect('equal')
        ax.tick_params(axis='both', which='major', labelsize= self.pltP['fontsize'])
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.24),\
                   fontsize = self.pltP['fontsize'])
        # Save the figure if required.
        if len(self.fig_dir) !=0 and self.save_fig:
            self._save_figure(fig, self.pltP['fig_name'],bbox_inches='tight', pad_inches=0.3)


