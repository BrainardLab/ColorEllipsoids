#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 21:01:22 2024

@author: fangfang
"""

#%% import modules
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import sys
import numpy as np

sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids")
from plotting.wishart_plotting import WishartModelBasicsVisualization 

#%%
class ZrefZ1Visualization(WishartModelBasicsVisualization):
    def __init__(self, W, model, rgb_ref, rgb_comp_pts, rgb_comp_contour,
                 fig_dir='', save_fig=False):
        """
        W (numpy.ndarray): Weight matrix for the model.
        model (object): Model object with methods to compute transformations.
        rgb_ref (numpy.ndarray; size: 2, ): 
            RGB values for the reference stimulus.
        rgb_comp (numpy.ndarray; size: 2, N): 
            RGB values for the comparison stimuli. N: #chromatic directions
        nTheta: N #chromatic directions
        rgb_comp_contour (numpy.ndarray; size: 2, M):
            RGB values for more finely sampled comparison stimuli. 
        nTheta_finer: M #chromatic directions
        """
        self.W = W
        self.model = model
        self.rgb_ref = rgb_ref
        self.rgb_comp_pts = rgb_comp_pts
        self.nTheta = rgb_comp_pts.shape[-1]
        self.rgb_comp_contour = rgb_comp_contour
        self.nTheta_finer = rgb_comp_contour.shape[-1]
        self.fig_dir = fig_dir
        self.save_fig = save_fig
        self.pltP = {'dpi': 256,
                     'target_pC': 2/3,
                     'lb_pC':1/3,
                     'ub_pC':1}
        
    def simulate_zref_z0_z1(self, **kwargs):
        """
        Simulate stimuli in perceptual space for a reference and comparison stimuli,
        and calculate the probability of choosing one comparison over another based on
        their Euclidean distances from the reference stimulus.
        
        Note: this function is very similar to a function in Alex's code (oddity_task),
        but that function doesn't return sampled zref, z0 and z1. This script relies 
        on z's for computation, so that's why I made this function.
        
        Returns:
            tuple: Contains arrays of simulated reference, comparison stimuli (z0, z1),
                   distances to reference, difference in distances, and 
                   probability of choosing x1.
        """
    
        # Define default parameters
        params = {
            'mc_samples':2000,
            'bandwidth': 5e-3,
            'opt_key':jax.random.PRNGKey(444),
        }
        # Update default parameters with any additional keyword arguments provided
        params.update(kwargs)
                
        # Compute U (weighted sum of basis functions) for the reference stimulus
        Uref      = self.model.compute_U(self.W, self.rgb_ref)
        shape_init = (self.nTheta_finer, params['mc_samples'], 2)
        zref_all       = np.full(shape_init, np.nan)
        z0_all         = np.full(shape_init, np.nan)
        z1_all         = np.full(shape_init, np.nan)
        z0_to_zref_all = np.full(shape_init[0:2], np.nan)
        z1_to_zref_all = np.full(shape_init[0:2], np.nan)
        zdiff_all      = np.full(shape_init[0:2], np.nan)
        pChoosingX1    = np.full((self.nTheta_finer), np.nan)
    
        # Iterate over each chromatic direction
        for i in range(self.nTheta_finer):        
            # Current RGB composition for the comparison
            rgb_comp_i = self.rgb_comp_contour[:,i]
            U1 = self.model.compute_U(self.W, rgb_comp_i)
                
            # Generate random draws from isotropic, standard gaussians
            keys = jax.random.split(params['opt_key'], num=6)
            nnref = jax.random.normal(keys[0], shape=(params['mc_samples'], Uref.shape[1]))
            nn0 = jax.random.normal(keys[1], shape=(params['mc_samples'], Uref.shape[1]))
            nn1 = jax.random.normal(keys[2], shape=(params['mc_samples'], Uref.shape[1]))
        
            zref = nnref @ Uref.T + self.rgb_ref[None, :] 
            z0 = nn0 @ Uref.T + self.rgb_ref[None, :]
            z1 = nn1 @ U1.T + rgb_comp_i[None, :]
            zref_all[i], z0_all[i], z1_all[i] = zref, z0, z1
        
            # Compute covariances
            S0 = Uref @ Uref.T 
            S1 = U1 @ U1.T 
        
            # Average covariances
            Sbar = (2 / 3) * S0 + (1 / 3) * S1
        
            # Compute squared Mahalanobis distances
            r01 = zref - z0
            r02 = zref - z1
            r12 = z0 - z1
        
            z0_to_zref_all[i] = jnp.sum(r01 * jnp.linalg.solve(Sbar, r01.T).T, axis=1)
            z1_to_zref_all[i] = jnp.sum(r02 * jnp.linalg.solve(Sbar, r02.T).T, axis=1)
            z0_to_z1 = jnp.sum(r12 * jnp.linalg.solve(Sbar, r12.T).T, axis=1)
            
            zdiff_all[i] = z0_to_zref_all[i] - jnp.minimum(z1_to_zref_all[i], z0_to_z1)
                    
            # Approximate the cumulative distribution function for the trial
            pChoosingX1[i] = np.sum(zdiff_all[i]<0)/params['mc_samples']
            
        return zref_all, z0_all, z1_all, z0_to_zref_all, z1_to_zref_all, zdiff_all, pChoosingX1
    
    #%%
    def plot_sampled_comp(self, gt = None, ax = None, **kwargs):
        """
        Plots sampled Z1 data for various chromatic directions and one set of 
        sampled Zref. Highlights the dispersion of Z1 around the reference stimulus 
        Zref.
        
        """
    
        # Default plot parameters; can be updated with kwargs to customize plot behavior
        method_specific_settings = {
            'alpha':1,
            'marker':'o',
            'line_alpha': 1,
            'markersize': 60,
            'figName':'SanityCheck_sampled_comp'}  # Default figure name
        # Update plot parameters with method-specific settings and external configurations.
        self.pltP.update(method_specific_settings)
        self.pltP.update(kwargs) 
        
        plt.rcParams['font.sans-serif'] = ['Arial']
        plt.rcParams.update({'font.size': 10})
        
        # Define color map and color assignments for plots
        default_cmap = plt.get_cmap('tab20b') 
        values       = np.linspace(0, 1, self.nTheta)
        colors_array = default_cmap(np.append(values[3:], values[:3]))
        
        # Initialize plot
        # Create a new figure and axes if not provided.
        if ax is None:
            fig, ax = plt.subplots(1, 1, dpi= self.pltP['dpi'])
        else:
            fig = ax.figure
            
        # Plot each set of Z1 samples
        if gt is not None:
            ax.plot(gt[0], gt[1], c = 'gray', alpha = self.pltP['line_alpha'],lw=2)
        ax.scatter(self.rgb_ref[0], self.rgb_ref[1], c = 'k',
                    marker = '+', s = self.pltP['markersize'],lw=2)
        for i in range(self.nTheta-1):
            ax.scatter(self.rgb_comp_pts[0,i], self.rgb_comp_pts[1,i],
                       marker = self.pltP['marker'], c = colors_array[i],
                       s = self.pltP['markersize'],lw=1)
        ax.grid(True)
        ax.set_aspect('equal')
        ax.set_xlim(self.rgb_ref[0] + jnp.array([-0.4, 0.4]))
        ax.set_ylim(self.rgb_ref[1] + jnp.array([-0.4, 0.4]))
        plt.tight_layout()
        # Save the figure if the directory is set and saving is enabled.
        if len(self.fig_dir) !=0 and self.save_fig:
            self._save_figure(fig, self.pltP['figName']+'.pdf')
        return fig, ax
            
    
    def plot_probC(self, pC, skipping_step = 10, ax = None, **kwargs):
        """
        This method plots the simulated probability of correct responses as a
        function of different chromatic directions
        """
        # Default plot parameters; can be updated with kwargs to customize plot behavior
        method_specific_settings = {
            'alpha':1,
            'line_alpha':0.5,
            'markersize':30,
            'figName':'SanityCheck_pC'}  # Default figure name
        # Update plot parameters with method-specific settings and external configurations.
        self.pltP.update(method_specific_settings)
        self.pltP.update(kwargs) 
        plt.rcParams['font.sans-serif'] = ['Arial']
        plt.rcParams.update({'font.size': 10})
        
        # Create a new figure and axes if not provided.
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize = (4,2.2), dpi= self.pltP['dpi'])
        else:
            fig = ax.figure
        
        default_cmap = plt.get_cmap('tab20b')
        values       = np.linspace(0, 1, self.nTheta)
        colors_array = default_cmap(np.append(values[3:], values[:3]))
        #pC = np.sort(pC)
        ax.plot(np.array(list(range(self.nTheta_finer))), pC, 
                color = 'gray', lw = 2)
        for i in range(self.nTheta):
            idx = np.max([0, int(i*skipping_step)-1])
            ax.scatter(idx, pC[idx], c = colors_array[i], s = self.pltP['markersize'])
        ax.set_ylim([self.pltP['lb_pC'], self.pltP['ub_pC']+0.05])
        ax.set_yticks(np.round([self.pltP['lb_pC'], self.pltP['target_pC'],
                                self.pltP['ub_pC']],3))
        ax.set_xticks([])
        ax.set_xlabel('Chromatic direction')
        ax.set_ylabel('p(correct)\n('+r'$x_1$'+' is the odd stimulus)')
        ax.set_xlim([-3, self.nTheta_finer])
        ax.grid(True)
        plt.tight_layout()
        # Save the figure if the directory is set and saving is enabled.
        if len(self.fig_dir) !=0 and self.save_fig:
            self._save_figure(fig, self.pltP['figName']+'.pdf')
        return fig, ax
        
    
    def plot_sampled_zref_z1(self, Zref, Z0, Z1, gt = None, sim = None, ax = None, **kwargs):
        """
        Plots sampled Z1 data for various chromatic directions and one set of 
        sampled Zref. Highlights the dispersion of Z1 around the reference stimulus 
        Zref.
        
        Parameters:
            Z1 (numpy.ndarray; size: N x M x 2): 
                The sampled comparison stimuli across different chromatic directions.
                N: # chromatic directions
                M: #MC samples
                2 dimensions
            Zref (numpy.ndarray):
                The sampled reference stimulus (with the same size as Z1).
            rgb_ref (numpy.ndarray; 2,): RGB values for the reference stimulus.
            **kwargs: Additional parameters for plot customization including saving options.
        
        Returns:
            float: The maximum bound used to set the x and y limits on the plot, centered on rgb_ref.
        """
    
        # Default plot parameters; can be updated with kwargs to customize plot behavior
        method_specific_settings = {
            'alpha':0.5,
            'markersize':30,
            'max_dots':Z1.shape[1],
            'legends':None,   # List of legends for each chromatic direction
            'figName':'SanityCheck_sampled_zref_z1'}  # Default figure name
        # Update plot parameters with method-specific settings and external configurations.
        self.pltP.update(method_specific_settings)
        self.pltP.update(kwargs) 
        plt.rcParams['font.sans-serif'] = ['Arial']
        plt.rcParams.update({'font.size': 10})
        
        # Define color map and color assignments for plots
        default_cmap = plt.get_cmap('tab20b') 
        numDirPts    = Z1.shape[0]
        values       = np.linspace(0, 1, 16)
        colors_array = default_cmap(np.append(values[3:], values[:3]))
        colors_ref   = np.array([0.5,0.5,0.5])
                
        # Create a new figure and axes if not provided.
        if ax is None:
            fig, ax = plt.subplots(1, 1, dpi= self.pltP['dpi'])
        else:
            fig = ax.figure
            
        z1_x_bds = 0 # Initialize the boundary for plot limits
        # Plot each set of Z1 samples
        for i in range(numDirPts):
            if self.pltP['legends'] is not None: lgd = self.pltP['legends'][i]; 
            else: lgd = None
            ax.scatter(Z1[i,:self.pltP['max_dots'],0],Z1[i,:self.pltP['max_dots'],1],
                        c = colors_array[i],s = self.pltP['markersize'], 
                        alpha = self.pltP['alpha'],
                        edgecolor = [1,1,1], label = lgd)
            ax.scatter(self.rgb_comp_pts[0,i], self.rgb_comp_pts[1,i], 
                       c = colors_array[i], marker = '+',
                       s = self.pltP['markersize']*3, lw = 3)
            if gt is not None:
                ax.plot(gt[0]+self.rgb_comp_pts[0,i]-self.rgb_ref[0],
                         gt[1]+self.rgb_comp_pts[1,i]-self.rgb_ref[1],
                         c = colors_array[i], alpha = self.pltP['alpha'])
            if sim is not None:
                ax.plot(sim[0]+self.rgb_comp_pts[0,i]-self.rgb_ref[0],
                         sim[1]+self.rgb_comp_pts[1,i]-self.rgb_ref[1], ls = '--', 
                         c = colors_array[i], alpha = self.pltP['alpha'])
            # Update the maximum boundary for plot limits based on Z1 data
            z1_x_bds = np.max([z1_x_bds, np.max(np.abs(Z1[i,:,0] - self.rgb_ref[0])),
                               np.max(np.abs(Z1[i,:,1] - self.rgb_ref[1]))])        
        ax.scatter(Zref[0,:self.pltP['max_dots'],0], Zref[0,:self.pltP['max_dots'],1],
                    c=colors_ref,s = self.pltP['markersize'],
                    alpha = self.pltP['alpha'], edgecolor = [1,1,1])
        ax.scatter(Z0[0,:self.pltP['max_dots'],0], Z0[0,:self.pltP['max_dots'],1],
                    c=colors_ref*0,s = self.pltP['markersize'], marker = '^',
                    alpha = self.pltP['alpha'], edgecolor = [1,1,1])
        if gt is not None:
            ax.plot(gt[0], gt[1], c = colors_ref, alpha = self.pltP['alpha'])
        if sim is not None:
            ax.plot(sim[0], sim[1], c = colors_ref, ls = '--',alpha = self.pltP['alpha'])
        #add centers
        ax.scatter(self.rgb_ref[0], self.rgb_ref[1], c = 'k', marker = '+',
                   s = self.pltP['markersize']*3, lw = 3)
        ax.grid(True)
        ax.set_aspect('equal')
        ax.set_xlim(self.rgb_ref[0] + jnp.array([-0.4, 0.4]))
        ax.set_ylim(self.rgb_ref[1] + jnp.array([-0.4, 0.4]))
        if self.pltP['legends'] is not None:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),ncol = 1,\
                   title='Chromatic \ndirection (deg)')
        plt.tight_layout()
        # Save the figure if the directory is set and saving is enabled.
        if len(self.fig_dir) !=0 and self.save_fig:
            self._save_figure(fig, self.pltP['figName']+'.pdf')
        return fig, ax
    
    def plot_EuclideanDist_hist(self, Z0_to_zref, Z1_to_zref,bin_edges, ax = None, **kwargs):
        """
        Plots histograms of the squared Euclidean distances from Z0 and Z1 to Zref
        for various chromatic directions in a multi-panel figure.
    
        Parameters:
            Z0_to_zref (numpy.ndarray; size: N x M): 
                Array of squared distances from Z0 to Zref for each direction.
                N: #chromatic directions
                M: #MC trials
            Z1_to_zref (numpy.ndarray size: N x M): 
                Array of squared distances from Z1 to Zref for each direction.
                (same size as Z0_to_zref)
            bin_edges (numpy.ndarray): 
                Array of bin edges for the histograms.
            **kwargs: Optional keyword arguments for plot customization such as 
                legends and saving.
    
        """
        
        # Set default plot parameters, update with any provided keyword arguments
        method_specific_settings = {
            'legends':None,
            'figName':'SanityCheck_sampled_zref_z1'} 
        # Update plot parameters with method-specific settings and external configurations.
        self.pltP.update(method_specific_settings)
        self.pltP.update(kwargs) 
        plt.rcParams['font.sans-serif'] = ['Arial']
        plt.rcParams.update({'font.size': 10})
        
        #define color map
        default_cmap = plt.get_cmap('tab20b')
        numDirPts    = Z0_to_zref.shape[0]
        values       = np.linspace(0, 1, numDirPts)
        colors_array = default_cmap(np.append(values[3:], values[:3]))
        colors_ref   = [0.5,0.5,0.5]
        
        # Create a new figure and axes if not provided.
        if ax is None:
            fig, ax = plt.subplots(4,4,figsize = (10,6), dpi= self.pltP['dpi'])
        else:
            fig = ax.figure

        # Upper bound for histogram x-axis
        bin_ub = bin_edges[-1] #0.01
        
        # Plot histograms for each direction
        for i in range(numDirPts):
            row_idx = i//4
            col_idx = i%4
            if self.pltP['legends'] is not None: lgd = 'cDir = '+self.pltP['legends'][i]+' deg'
            else: lgd = None
            # Configure axis for the current subplot
            ax[row_idx, col_idx].set_xlim([0, np.around(bin_ub,2)])
            # Plot histogram for Z0 to Zref distances
            ax[row_idx, col_idx].hist(Z0_to_zref[i], bins = bin_edges,
                                color= colors_ref, alpha = 0.5, 
                                edgecolor = colors_ref)
            # Plot histogram for Z1 to Zref distances
            ax[row_idx, col_idx].hist(Z1_to_zref[i], bins = bin_edges,
                                color=colors_array[i],alpha = 0.5, 
                                edgecolor = colors_ref, label = lgd)
             # Adjust ticks and labels for clarity and aesthetics
            if col_idx !=0: ax[row_idx, col_idx].set_yticks([])
            if row_idx != 3: ax[row_idx, col_idx].set_xticks([]); 
            else: ax[row_idx, col_idx].set_xticks(np.round(np.linspace(0, np.around(bin_ub,2),3),2)); 
            ax[row_idx, col_idx].tick_params(axis='x')
            ax[row_idx, col_idx].tick_params(axis='y')
            if self.pltP['legends'] is not None:
                lgd = ax[row_idx, col_idx].legend()
                lgd.set_frame_on(False)
            #ax[row_idx, col_idx].grid(True)
        fig.suptitle(r'$d_M(z_0, z_{ref})^2 vs. d_M(z_1, z_{ref})^2$')
        plt.tight_layout()
        # Save the figure if the directory is set and saving is enabled.
        if len(self.fig_dir) !=0 and self.save_fig:
            self._save_figure(fig, self.pltP['figName']+'.pdf')
        return fig, ax
    
    def plot_EuclieanDist_diff_hist(self, Z_diff, bin_edges, ax = None, **kwargs):
        """
        Plots histograms of the difference between squared Euclidean distances from 
        Z0 and Z1 to Zref for various chromatic directions in a multi-panel figure.
    
        Parameters:
            Z_diff (numpy.ndarray; size: N x M): 
                Array of squared distances from Z0 to Zref for each direction.
                N: #chromatic directions
                M: #MC trials
            bin_edges (numpy.ndarray): 
                Array of bin edges for the histograms.
            **kwargs: Optional keyword arguments for plot customization such as 
                legends and saving.
        """
        method_specific_settings = {
            'pC': None,
            'legends':None,
            'ylim':[0,500],
            'figName':'SanityCheck_sampled_zref_z1'} 
        # Update plot parameters with method-specific settings and external configurations.
        self.pltP.update(method_specific_settings)
        self.pltP.update(kwargs) 
        plt.rcParams['font.sans-serif'] = ['Arial']
        plt.rcParams.update({'font.size': 10})
        
        #define color map
        default_cmap = plt.get_cmap('tab20b')
        numDirPts    = Z_diff.shape[0]
        values       = np.linspace(0, 1, numDirPts)
        colors_array = default_cmap(np.append(values[3:], values[:3]))
        
        # Create a new figure and axes if not provided.
        if ax is None:
            fig, ax = plt.subplots(4,4,figsize = (10,6), dpi= self.pltP['dpi'])
        else:
            fig = ax.figure

        #bin_edges= np.linspace(-bin_ub,bin_ub/2,30)
        for i in range(numDirPts):
            row_idx = i//4
            col_idx = i%4
            if self.pltP['legends'] is not None: lgd = 'cDir = '+self.pltP['legends'][i]+' deg'; 
            else: lgd = None
            ax[row_idx, col_idx].set_xlim(np.around([bin_edges[0], bin_edges[-1]],2))
            ax[row_idx, col_idx].set_ylim(self.pltP['ylim'])
            ax[row_idx, col_idx].plot([0,0],[0,Z_diff.shape[-1]/2],c = 'k', linewidth=2)
            ax[row_idx, col_idx].hist(Z_diff[i], bins = bin_edges,
                                color=colors_array[i],alpha = 0.5,
                                edgecolor = colors_array[i],
                                label = lgd)
            if self.pltP['pC'] is not None: 
                ax[row_idx, col_idx].text(-25,200,f"pC = {self.pltP['pC'][i]}")
            if col_idx !=0: ax[row_idx, col_idx].set_yticks([])
            if row_idx != 3: ax[row_idx, col_idx].set_xticks([]); 
            else:
                xticks = np.around(np.linspace(bin_edges[0], bin_edges[-1],3),2)
                xticks = np.sort(np.append(xticks, 0))
                ax[row_idx, col_idx].set_xticks(xticks); 
            ax[row_idx, col_idx].tick_params(axis='x')
            ax[row_idx, col_idx].tick_params(axis='y')
            if self.pltP['legends'] is not None:
                lgd_all = ax[row_idx, col_idx].legend()
                lgd_all.set_frame_on(False)
            #ax[row_idx, col_idx].grid(True)
        fig.suptitle(r'$d_M(z_0, z_{ref})^2 - min([d_M(z_1, z_{ref})^2, d_M(z_1, z_0)^2])$')
        plt.tight_layout()
        # Save the figure if the directory is set and saving is enabled.
        if len(self.fig_dir) !=0 and self.save_fig:
            self._save_figure(fig, self.pltP['figName']+'.pdf')
        return fig, ax
    
    def plot_nLL(self, nLL_avg, nLL_lb, nLL_ub, nLL_target_pC_avg,
                 nLL_target_pC_lb, nLL_target_pC_ub, ax = None, **kwargs):
        method_specific_settings = {
            'x_err_plt': 1,
            'figName':'nLL'} 
        # Update plot parameters with method-specific settings and external configurations.
        self.pltP.update(method_specific_settings)
        self.pltP.update(kwargs) 
        # Define the corner points of the rectangle (clockwise or counterclockwise)
        rectangle_points = [(0, nLL_target_pC_lb), (0, nLL_target_pC_ub),
                            (4, nLL_target_pC_ub), (4, nLL_target_pC_lb)]

        # Create a polygon using the rectangle points
        rectangle = Polygon(rectangle_points, closed=True, edgecolor='none',
                            facecolor= 'gray', alpha = 0.5)

        # Create a new figure and axes if not provided.
        if ax is None:
            fig, ax = plt.subplots(1,1,figsize = (2.1,2.7), dpi= self.pltP['dpi'])
        else:
            fig = ax.figure
        plt.rcParams['font.sans-serif'] = ['Arial']
        plt.rcParams.update({'font.size': 10})
        
        ax.plot([0,4], nLL_target_pC_avg*np.ones((2)),c = 'k')
        # Add the rectangle to the axis
        ax.add_patch(rectangle)
        ax.errorbar(self.pltP['x_err_plt'], nLL_avg, 
                    yerr = [[nLL_ub - nLL_avg], [nLL_avg - nLL_lb]],
                    marker = 'o', c = 'k') 
        ax.set_xlim([0,4])
        ax.set_xticks([1,2,3])
        ax.set_ylim([0.4, 2])
        ax.set_yticks(np.linspace(0.4,2,5))
        ax.set_xlabel('Hypothesized\nWeight matrix')
        ax.set_ylabel('Negative log likelihood')
        plt.tight_layout()
        # Save the figure if the directory is set and saving is enabled.
        if len(self.fig_dir) !=0 and self.save_fig:
            self._save_figure(fig, self.pltP['figName']+'.pdf')
        return fig, ax

#%%
    @staticmethod
    def compute_nLL_w_simData(nIters, pC_givenW, nTheta_finer=160, target_pC=2/3, seed=0):
        """
        Computes the negative log-likelihood (nLL) based on simulated data.
        
        Parameters:
        nIters: int
            Number of iterations (simulations) to perform.
        pC_givenW: array-like
            Probability of correct response (pC) given W for each angle (theta).
        nTheta_finer: int, optional
            Number of finer discretization points (angles, theta) in the simulation.
            Default is 160.
        target_pC: float, optional
            Target probability for generating binary responses (default is 2/3).
        seed: int, optional
            Random seed for reproducibility (default is 0).
        
        Returns:
        nLL_mean: array
            Array of mean negative log-likelihoods over all iterations.
        """
        
        # Set the random seed for reproducibility
        np.random.seed(seed)
        # Generate random numbers for binary response simulation (nTheta_finer x nIters)
        randNum = np.random.rand(nTheta_finer, nIters)        
        # Simulate binary responses (1 if random number < target_pC, else 0)
        binaryResp = (randNum < target_pC).astype(int)        
        # Initialize array to hold mean nLL for each iteration
        nLL_mean = np.full((nIters), np.nan)
        
        # Loop through each iteration to compute nLL
        for i in range(nIters):
            # Calculate negative log-likelihood for each binary response
            nLL = binaryResp[:, i] * jnp.log(pC_givenW) +\
                  (1 - binaryResp[:, i]) * jnp.log(1 - pC_givenW)            
            # Filter out infinite and NaN values from nLL
            nLL_filtered = nLL[~np.isinf(nLL) & ~np.isnan(nLL)]            
            # Compute the mean of the filtered nLL and store it
            nLL_mean[i] = -np.mean(nLL_filtered)        
        # Return the mean nLL for all iterations
        return nLL_mean
        
    @staticmethod
    def nLL_avg_95CI(nLL, bds = [0.025, 0.975]):
        """
        Computes the average negative log-likelihood (nLL) and the 95% confidence interval (CI).
        
        Parameters:
        nLL: array
            Array of negative log-likelihood values.
        bds: list, optional
            List containing the lower and upper bounds for the confidence interval percentiles.
            Default is [0.025, 0.975], corresponding to the 2.5th and 97.5th percentiles, 
            which represent the 95% confidence interval.
        
        Returns:
        nLL_avg: float
            The mean of the negative log-likelihood values.
        nLL_lb: float
            The lower bound of the 95% confidence interval.
        nLL_ub: float
            The upper bound of the 95% confidence interval.
        """
        
        # Compute the mean of the negative log-likelihood (nLL) values
        nLL_avg = np.mean(nLL)
        # Sort the nLL values in ascending order to compute percentiles
        nLL_sort = np.sort(nLL)
        # Calculate the lower bound of the 95% confidence interval
        nLL_lb = nLL_sort[int(len(nLL) * bds[0])]
        # Calculate the upper bound of the 95% confidence interval
        nLL_ub = nLL_sort[int(len(nLL) * bds[1])]
        # Return the average nLL and the bounds of the 95% confidence interval
        return nLL_avg, nLL_lb, nLL_ub
        
        
        
        
        