#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 14:36:54 2024

@author: fangfang
"""

import jax
jax.config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
import numpy as np
import os
import imageio.v2 as imageio
from numpy.polynomial.chebyshev import chebval2d

#%%
class wishart_model_basics_visualization:
    def __init__(self, fig_dir = '', save_fig = False, save_gif = False):
        self.fig_dir  = fig_dir
        self.save_fig = save_fig
        self.save_gif = save_gif
        self.ndims    = 2
        self.pltP = {
            'xticks':[],
            'yticks':[],
            'xticklabels':[],
            'yticklabels':[],            
            'fontsize':12,
            'cmap':'PRGn',
            'dpi':256
                }
        
    def _save_figure(self, fig, fig_name):
        """
        Saves the given figure to the specified directory with a provided filename.
    
        Parameters:
            fig (Figure): The matplotlib figure object to save.
            fig_name (str): The filename under which to save the figure.
        """
        if os.path.exists(self.fig_dir):
            full_path = os.path.join(self.fig_dir, fig_name)
            fig.savefig(full_path, dpi=self.pltP['dpi'])
        else:
            raise FileNotFoundError(f"The directory {self.fig_dir} does not exist.")

    def _save_gif(self, gif_name, fig_name_start, fig_name_end = '.png', fps = 2):
        """
        Compiles a sequence of images into a GIF and saves it to the specified directory.
        
        Parameters:
            gif_name (str): The filename for the GIF.
            fig_name_start (str): The beginning pattern of filenames to include in the GIF.
            fig_name_end (str): The ending pattern of filenames to include in the GIF.
            fps (int): Frames per second, defining the speed of the GIF.
        """

        images = [img for img in os.listdir(self.fig_dir) \
                  if img.startswith(fig_name_start) and img.endswith(fig_name_end)]
        images.sort()  # Sort the images by name (optional)
        image_list = [imageio.imread(f"{self.fig_dir}/{img}") for img in images]
        # Create a GIF
        output_path = f"{self.fig_dir}{gif_name}.gif"
        imageio.mimsave(output_path, image_list, fps= fps)  
        
    def _configure_colormap(self, val):
        """
        Configures the color map bounds based on the absolute maximum value from the provided values.
        
        Parameters:
            val (array): Array of values from which to determine the color map bounds.
        """
        max_val = np.max(np.abs(val))
        self.pltP['cmap_bds'] = [-max_val, max_val]
        
    def _update_axes_limits(self, ax, lim):
        """
        Sets uniform limits for axes of a plot, extending to 3D if applicable.
        
        Parameters:
            ax (Axes): The matplotlib axes object to modify.
            lim (list): The limits to set for the x and y (and z if 3D) axes.
        """
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        if self.ndims == 3: 
            ax.set_zlim(lim)
        
    def _update_axes_labels(self, ax, unit_true, unit_show, nsteps = 2):
        """
        Sets ticks and labels for plot axes, adapting to the number of dimensions.
        
        Parameters:
            ax (Axes): The matplotlib axes object to modify.
            unit_true (list): Actual data points for tick positions.
            unit_show (list): Values to display at tick positions.
            nsteps (int): Interval for selecting ticks and labels.
        """

        if len(unit_true) == 0:
            ax.set_xticks([])
            ax.set_yticks([])
            if self.ndims == 3: ax.set_zticks([])
        else:
            ax.set_xticks(unit_true[::nsteps])
            ax.set_yticks(unit_true[::nsteps])
            ax.set_xticklabels([f"{x:.2f}" for x in unit_show[::nsteps]])
            ax.set_yticklabels([f"{x:.2f}" for x in unit_show[::nsteps]])
            if self.ndims == 3: 
                ax.set_zticks(unit_true[::nsteps])
                ax.set_zticklabels([f"{x:.2f}" for x in unit_show[::nsteps]])
        self.pltP['xticks'] = ax.get_xticks()
        self.pltP['yticks'] = ax.get_yticks()
        self.pltP['xticklabels'] = ax.get_xticklabels()
        self.pltP['yticklabels'] = ax.get_yticklabels()
        if self.ndims == 3:
            self.pltP['zticks'] = ax.get_zticks()
            self.pltP['zticklabels'] = ax.get_zticklabels()            
        
    def _configure_labels_and_title(self, ax):
        """
        Configures labels and title for a plot based on predefined plane settings.
        
        Parameters:
            ax (Axes): The matplotlib axes object to modify.
        """

        if self.pltP['plane_2D'] in ['RG plane', 'GB plane', 'RB plane']:
            ax.set_xlabel(self.pltP['plane_2D'][0], fontsize=self.pltP['fontsize'])
            ax.set_ylabel(self.pltP['plane_2D'][1], fontsize=self.pltP['fontsize'])
            ax.set_title(self.pltP['plane_2D'], fontsize=self.pltP['fontsize'])
        else:
            # Default labels when plane_2D is not specified
            ax.set_xlabel('Dim 1', fontsize=self.pltP['fontsize'])
            ax.set_ylabel('Dim 2', fontsize=self.pltP['fontsize'])
            ax.set_title('2D plane', fontsize=self.pltP['fontsize'])
            
#%% 
    def plot_basis_function_1d(self, degree, grid, cheby_func, **kwargs):
        """
        Plot a series of 1D Chebyshev polynomial basis functions.
    
        This function generates and displays a series of plots, each showing a 
        single Chebyshev polynomial of a specified degree evaluated at provided grid points.
        It allows for customization of the plots through keyword arguments.
    
        Parameters
        ----------
        degree : int
            The degree of the Chebyshev polynomial. This also determines the number of subplots,
            as each degree from 0 to `degree-1` will be plotted.
        grid : array-like, shape (N,)
            The grid points at which the Chebyshev polynomials are evaluated. These points should
            cover the domain of interest, typically [-1, 1] for Chebyshev polynomials.
        cheby_func : array-like, shape (N, degree)
            The values of the Chebyshev polynomials at each point in `grid`. Each column corresponds
            to a polynomial of increasing degree.
        kwargs : dict, optional
            Additional keyword arguments to customize the plots, such as 'linewidth' or 'fig_size'.
            These settings will override the method-specific defaults.

        """
        # Indicate that this is a 1D plot.
        self.ndims = 1
        # Copy the default settings and update them with any method-specific settings.        
        method_specific_settings = {
            'fig_size': (2,8),
            'linewidth': 2,
            'fig_name':'Chebyshev_basis_functions_1D.png',
        }
        self.pltP.update(method_specific_settings)
        self.pltP.update(kwargs) 
        
        # Create a figure and a set of subplots with shared x and y axes.
        fig, axes = plt.subplots(degree, 1, figsize=self.pltP['fig_size'],\
                                 sharex=True, sharey=True)
        for i in range(degree):
            axes[i].plot(grid,cheby_func[:,i], color = 'k', \
                         linewidth = self.pltP['linewidth'])
            axes[i].set_aspect('equal')
        plt.show()
        if len(self.fig_dir) !=0 and self.save_fig:
            self._save_figure(fig, self.pltP['fig_name'])

#%%
    def plot_2D_covMat(self, axes, fig, Sigmas_est, ellipses, xgrid_N_unit, **kwargs):
        """
        Visualizes the covariance matrices and corresponding ellipses to represent the shape
        and orientation influenced by the estimated sigma values at specific grid locations.
    
        This function divides the plot into two main sections: 
            The left side displays individual components of the covariance matrices 
            (\sigma_x^2, \sigma_xy, \sigma_y^2) at each grid point.
            The right side shows ellipses that correspond to these covariance matrices, 
            illustrating their geometric interpretation.
    
        Parameters
        ----------
        axes : array of AxesSubplot
            The array of matplotlib subplot axes objects for displaying the covariance matrix components.
        fig : Figure
            The matplotlib figure object that holds all subplots.
        Sigmas_est : np.array, shape (N, N, 2, 2)
            The estimated covariance matrices for each grid point, where N is the number of finely 
            sampled grid points.
        ellipses : np.array, shape (n, n, 2, 200)
            Coordinates for elliptical contours derived from covariance matrices, sampled at a coarser grid.
            The last dimension (200) represents the points that define each ellipse.
        xgrid_N_unit : np.array
            Normalized grid units used for scaling and plotting grid lines and ticks.

        """
        # Initialize method-specific settings and update with any additional configurations provided.
        method_specific_settings = {
            'slc_idx_dim1': 0,
            'slc_idx_dim2': 0,
            'bds_W_unit': [-1, 1],
            'plane_2D': '',
            'title_list': [[r'$\sigma^2_{dim1}$', r'$\sigma_{(dim1,dim2)}$'],
                           [r'$\sigma_{(dim2,dim1)}$', r'$\sigma^2_{dim2}$']],
            'cmap_bds': [],
            'flag_rescale_axes_label': True,
            'figName_ext': ''
        }
        self.pltP.update(method_specific_settings)
        self.pltP.update(kwargs)  # Apply any external configurations
        # Configure the colormap based on the values in Sigmas_est.
        self._configure_colormap(Sigmas_est)
        
        # Number of finely sampled grid points.
        num_grid_fine = Sigmas_est.shape[0]
        # Number of coarsely sampled grid points.
        num_grid = ellipses.shape[0]
        # Adjusted grid unit scale for plotting.
        xgrid_W_unit = xgrid_N_unit*2 - 1
                
        for i in range(self.ndims):
            for j in range(self.ndims):      
                # Plot the covariance matrix using a heatmap
                im = axes[i, j].imshow(Sigmas_est[:,:,i,j], cmap = self.pltP['cmap'],\
                                  vmin = self.pltP['cmap_bds'][0], vmax = self.pltP['cmap_bds'][1])
                # Calculate the scaled position for the horizontal line
                xgrid_scaled_p = xgrid_N_unit[num_grid-self.pltP['slc_idx_dim1']-1]*num_grid_fine
                # Draw horizontal line
                axes[i, j].plot([0, num_grid_fine], [xgrid_scaled_p,xgrid_scaled_p],\
                                c = 'grey',lw = 0.5)
                # Calculate the scaled position for the vertical line
                xgrid_scaled_q = xgrid_N_unit[self.pltP['slc_idx_dim2']]*num_grid_fine
                # Draw vertical line
                axes[i, j].plot([xgrid_scaled_q, xgrid_scaled_q],\
                                [0, num_grid_fine], c = 'grey',lw = 0.5)
                # Mark the intersection point
                axes[i, j].scatter(xgrid_scaled_q, xgrid_scaled_p, c = 'k', s = 10)
                # ticks and title
                if self.pltP['flag_rescale_axes_label']:
                    self._update_axes_labels(axes[i,j], xgrid_N_unit*num_grid_fine,\
                                             xgrid_N_unit, nsteps = 2)
                else:
                    self._update_axes_labels(axes[i,j], xgrid_N_unit*num_grid_fine,\
                                             xgrid_W_unit, nsteps = 2)  
                self._update_axes_limits(axes[i,j], [0,num_grid_fine-1])
                axes[i, j].set_title(self.pltP['title_list'][i][j])
        
        # Remove the axes that will be merged into the big plot
        plt.delaxes(axes[0, 2])
        plt.delaxes(axes[0, 3])
        plt.delaxes(axes[1, 2])
        plt.delaxes(axes[1, 3])
        
        cbar_ax = fig.add_axes([0.065, 0.1, 0.4, 0.02])  # [left, bottom, width, height]
        fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
        
        # Add a new large subplot that spans the last two columns of both rows
        # This creates a subplot that spans the right half of the figure
        ax_ell = fig.add_subplot(1, 2, 2)  
        for pp in range(num_grid):
            for qq in range(num_grid):
                # Plot ellipses based on the model predictions
                if pp < self.pltP['slc_idx_dim1'] or (pp == self.pltP['slc_idx_dim1'] \
                                                 and qq <= self.pltP['slc_idx_dim2']):
                    ax_ell.plot(ellipses[num_grid-1-pp,qq,0],\
                                ellipses[num_grid-1-pp,qq,1],c='k')
        # ticks and title
        self._update_axes_limits(ax_ell, self.pltP['bds_W_unit']) 
        if self.pltP['flag_rescale_axes_label']:
            self._update_axes_labels(ax_ell, xgrid_W_unit, xgrid_N_unit, nsteps = 2)
        else:
            self._update_axes_labels(ax_ell, xgrid_W_unit, xgrid_W_unit, nsteps = 2)  
        self._configure_labels_and_title(ax_ell)
        ax_ell.grid(True, alpha=0.5)
        ax_ell.set_aspect('equal')
        # Show the plot
        plt.subplots_adjust(left=0.05, right=0.95, top=0.925, bottom=0.2)
        plt.show()
        fig_counter = self.pltP['slc_idx_dim1']*num_grid+self.pltP['slc_idx_dim2']
        self.pltP['fig_name'] = 'CovarianceMatrix_'+self.pltP['plane_2D']+\
            self.pltP['figName_ext']+ f'_{fig_counter:02d}.png' 
        if len(self.fig_dir) !=0 and self.save_fig:
            self._save_figure(fig, self.pltP['fig_name'])
            
#%%
    def plot_basis_function_2D(self, degree, grid, **kwargs):
        """
        Plot 2D Chebyshev basis functions on a specified grid.
    
        This function generates a grid of subplots where each subplot represents 
        one of the basis functions for a 2D Chebyshev polynomial of specified 
        degrees. It demonstrates the effect of individual polynomial terms in 
        two dimensions, each identified by a pair of indices (i, j).

        Parameters
        ----------
        degree : int
            The degree of the Chebyshev polynomial. This also determines the number of subplots,
            as each degree from 0 to `degree-1` will be plotted.
        grid : array-like, shape (N,)
            The grid points at which the Chebyshev polynomials are evaluated. These points should
            cover the domain of interest, typically [-1, 1] for Chebyshev polynomials.

        """
        
        method_specific_settings = {
            'cmap_bds': [-1,1],
            'fig_size': (8,8),
            'fontsize': 12,
            'fig_name':'Chebyshev_basis_functions_2D.png',
        }
        self.pltP.update(method_specific_settings)
        self.pltP.update(kwargs)  # Apply any external configurations
        
        # Create a 2D mesh grid using the provided 1D grid array.
        xg, yg = np.meshgrid(grid, grid)
        # Initialize a grid for storing coefficients of the 2D polynomials.
        cg = np.zeros((degree, degree))
        
        # Create a figure with subplots arranged in a square grid.
        fig, axes = plt.subplots(degree, degree, figsize= self.pltP['fig_size'],\
                                 sharex=True, sharey=True)
        for i in range(degree):
            for j in range(degree):
                # Activate the (i, j)th term by setting its coefficient to 1.
                cg[i, j] = 1.0
                # Evaluate the 2D polynomial at the grid points.
                zg_2d = chebval2d(xg, yg, cg)
                
                # Display the result as an image in the corresponding subplot.
                axes[i, j].imshow(zg_2d, cmap = self.pltP['cmap'], \
                                  vmin = self.pltP['cmap_bds'][0], \
                                  vmax = self.pltP['cmap_bds'][1])
                # Reset the coefficient for the next iteration.
                cg[i, j] = 0.0
                
                # Update axis labels and limits to make the plots cleaner.
                self._update_axes_labels(axes[i,j], [], [])
                self._update_axes_limits(axes[i,j], [0,grid.shape[0]-1])
                
                # Set a title for each subplot to indicate the polynomial degrees.
                axes[i, j].set_title(f"({i}, {j})", fontsize = self.pltP['fontsize'])
        plt.tight_layout()
        if len(self.fig_dir) !=0 and self.save_fig:
            self._save_figure(fig, self.pltP['fig_name'])

#%%
    def plot_basis_functions_3D(self, XG, YG, ZG, M, **kwargs):
        """
            Visualizes selected slices of 3D Chebyshev basis functions over time.
        
            Due to the complexity of visualizing high-dimensional data, this function simplifies
            the representation by displaying one 2D slice at a time from the 3D 
            Chebyshev basis functions. Each slice is treated as a time step, providing 
            a series of 2D plots that represent how the basis functions evolve over 
            the third dimension, conceptualized as time.
            
            This script is also used to visualize selected slices of weigthed sum
            of basis functions (U), as well as 3D covariance matrices (Sigmas)
        
            Parameters
            ----------
            XG, YG, ZG : array-like, shape (N, N, N)
                3D grids representing the x, y, and z coordinates in the cube, respectively. 
                These grids define the points at which the basis functions are evaluated.
            M : array-like, shape (N, N, N, degree, degree)
                The values of the basis functions evaluated at every point in the 3D grid 
                (XG, YG, ZG) for each combination of polynomial degrees up to the specified 
                degree.

        """
        # Indicate the number of dimensions being visualized.
        self.ndims = 3
        # Copy the default settings and update them with any method-specific settings.        
        method_specific_settings = {
            'fig_size': (8,9),
            'xyzlim':[-1.05, 1.05],
            'fig_name':'Chebyshev_basis_function',
        }
        self.pltP.update(method_specific_settings)
        self.pltP.update(kwargs) 
        cmap = plt.get_cmap(self.pltP['cmap'])
        
        nbins = M.shape[0]  # Number of bins (slices) in the third dimension.
        ndim1 = M.shape[-2]
        ndim2 = M.shape[-1]
        for l in range(nbins):
            # Create a new figure with 3D subplots for each time point.
            fig, ax = plt.subplots(ndim1, ndim2, dpi = self.pltP['dpi'],\
                                   figsize = self.pltP['fig_size'],\
                                   subplot_kw={'projection': '3d'})
            for i in range(ndim1):
                for j in range(ndim2): 
                    max_val = np.max([-np.min(M), np.max(M)])
                    # Plot the basis functions.
                    ax[i, j].plot_surface(XG[:,:,l], ZG[:,:,l], YG[:,:,l],\
                        facecolors=cmap(((M[:,:,l,i,j] + max_val)/ (2*max_val+1e-10))),
                        rstride=1, cstride=1)
                    # Set aspect ratio and limits for each subplot.
                    self._update_axes_labels(ax[i, j], [], [])
                    self._update_axes_limits(ax[i,j], self.pltP['xyzlim'])
                    ax[i, j].view_init(20,-75)
                    ax[i, j].set_aspect('equal')
                    ax[i, j].set_autoscale_on(False)
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None,\
                                wspace=-0.1, hspace=-0.1)
            plt.show()
            fig_name = self.pltP['fig_name']+ f'_slice{l:02}.png'
            if len(self.fig_dir) !=0 and self.save_fig:
                self._save_figure(fig, fig_name)
        if self.save_gif:
            self._save_gif(self.pltP['fig_name'], self.pltP['fig_name'])

#%% 
    def plot_W_selected_slice(self, degree, W, basis_orders, slc_slice=[0,0], **kwargs):
        """
        Plots selected slices of the weight matrix for 3D Chebyshev polynomial basis functions.
    
        This function visualizes slices of the weight matrix applied to Chebyshev basis functions,
        emphasizing the variation of weights across different polynomial orders. Each
        plotted figure shows a 2D slice of the weight matrix for a specific degree of polynomial,
        annotated with the highest polynomial order of the basis functions at each point.
    
        Parameters
        ----------
        degree : int
            The degree of the polynomial
        W : array, shape (degree, degree, degree, ndims, ndims+1)
            The weight matrix that modifies the basis functions, structured to accommodate 
            multiple polynomial degrees and dimensions.
        basis_orders : array, shape (degree, degree, degree)
            Specifies the highest polynomial order for each basis function, used for annotation in plots.
        slc_slice : list of int, optional
            Indices specifying the slice of the last two dimensions of W to be visualized. Defaults to [0,0].

        """
        
        # Copy the default settings and update them with any method-specific settings.   
        self.ndims = 2
        method_specific_settings = {
            'fig_size': (5,5),
            'cmap':'RdBu',
            'fig_name':'EstimatedWeightMatrix',
        }
        self.pltP.update(method_specific_settings)
        self.pltP.update(kwargs)  # Apply any external configurations
        self._configure_colormap(W)  #color map

        # Plot each degree as a separate figure.
        for i in range(degree): #23d dimension
            fig, ax = plt.subplots(1, 1, figsize= self.pltP['fig_size'],\
                                   sharex=True, sharey=True)
            ax.imshow(W[:,:,i,*(slc_slice)], cmap = self.pltP['cmap'],\
                      vmin = self.pltP['cmap_bds'][0], vmax = self.pltP['cmap_bds'][1])
            # Annotate each cell in the plot with the highest polynomial order.
            for j in range(degree):
                for k in range(degree):
                    # Display text over the image
                    ax.text(j, k, str(basis_orders[j,k,i]), color='black',\
                            fontsize=20, ha='center', va='center')
            self._update_axes_labels(ax, [],[])
            ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
            if len(self.fig_dir) !=0 and self.save_fig:
                self._save_figure(fig, self.pltP['fig_name']+f'_degree{i}_{slc_slice}.png')





        
        