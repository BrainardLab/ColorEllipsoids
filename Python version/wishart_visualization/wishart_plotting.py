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

#%%
class wishart_model_basics_visualization:
    def __init__(self, fig_dir = '', save_fig = False, save_gif = False):
        self.fig_dir = fig_dir
        self.save_fig = save_fig
        self.save_gif = save_gif
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
        full_path = os.path.join(self.fig_dir, fig_name)
        fig.savefig(full_path, dpi=self.pltP['dpi'])
        
    def _save_gif(self, gif_name, fig_name_start, fig_name_end = '.png', fps = 2):
        """
        Creates a GIF from a series of images stored in a directory that match the specified start and end patterns.
        
        Parameters:
            fig_name_start (str): Start pattern of the image filenames to include in the GIF.
            fig_name_end (str): End pattern of the image filenames to include in the GIF.
            gif_name (str): Filename for the resulting GIF.
            fps (int): Frames per second in the resulting GIF. Default is 2.
        """

        images = [img for img in os.listdir(self.fig_dir) \
                  if img.startswith(fig_name_start) and img.endswith(fig_name_end)]
        images.sort()  # Sort the images by name (optional)
        image_list = [imageio.imread(f"{self.fig_dir}/{img}") for img in images]
        # Create a GIF
        output_path = f"{self.fig_dir}{gif_name}.gif"
        imageio.mimsave(output_path, image_list, fps= fps)  
        
    def _configure_colormap(self, val):
        max_val = np.max(np.abs(val))
        self.pltP['cmap_bds'] = [-max_val, max_val]
        
    def _update_axes_labels(self, ax, unit_true, unit_show, nsteps = 2):
        ax.set_xticks(unit_true[::nsteps])
        ax.set_yticks(unit_true[::nsteps])
        ax.set_xticklabels([f"{x:.2f}" for x in unit_show[::nsteps]])
        ax.set_yticklabels([f"{x:.2f}" for x in unit_show[::nsteps]])
        self.pltP['xticks'] = ax.get_xticks()
        self.pltP['yticks'] = ax.get_yticks()
        self.pltP['xticklabels'] = ax.get_xticklabels()
        self.pltP['yticklabels'] = ax.get_yticklabels()
        
    def _configure_labels_and_title(self, ax):
        if self.pltP['plane_2D'] in ['RG plane', 'GB plane', 'RB plane']:
            ax.set_xlabel(self.pltP['plane_2D'][0], fontsize=self.pltP['fontsize'])
            ax.set_ylabel(self.pltP['plane_2D'][1], fontsize=self.pltP['fontsize'])
            ax.set_title(self.pltP['plane_2D'], fontsize=self.pltP['fontsize'])
        else:
            # Default labels when plane_2D is not specified
            ax.set_xlabel('Dim 1', fontsize=self.pltP['fontsize'])
            ax.set_ylabel('Dim 2', fontsize=self.pltP['fontsize'])
            ax.set_title('2D plane', fontsize=self.pltP['fontsize'])

    def plot_2D_covMat(self, axes, fig, Sigmas_est, ellipses, xgrid_N_unit, **kwargs):
        """
        Sigmas_est (np.array, size: N x N x 2 x 2 where N is the number of grid poitns sampled finely)
        ellipses (np.array, size: n x n x 2 x 200 where n is the number of grid points sampled coarsely
                  and 200 is the sampled points for each elliptical contour)
        """
        # Copy the default settings and update them with any method-specific settings.        
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
        #color map
        self._configure_colormap(Sigmas_est)
        
        degree = 2 
        num_grid_fine = Sigmas_est.shape[0]
        num_grid = ellipses.shape[0]
        xgrid_W_unit = xgrid_N_unit*2 - 1
                
        for i in range(degree):
            for j in range(degree):      
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
                axes[i, j].set_xlim([0,num_grid_fine-1])
                axes[i, j].set_ylim([0,num_grid_fine-1])
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
        ax_ell.set_xlim(self.pltP['bds_W_unit'])
        ax_ell.set_ylim(self.pltP['bds_W_unit'])        
        if self.pltP['flag_rescale_axes_label']:
            self._update_axes_labels(ax_ell, xgrid_W_unit, xgrid_N_unit, nsteps = 2)
        else:
            self._update_axes_labels(ax_ell, xgrid_W_unit, xgrid_W_unit, nsteps = 2)  
        ax_ell.grid(True, alpha=0.5)
        self._configure_labels_and_title(ax_ell)
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
    def plot_basis_functions_3D(XG, YG, ZG, M, **kwargs):
        pltP = {
            'figSize':(8,9),
            'xlim':[-1.05, 1.05],
            'ylim':[-1.05, 1.05],
            'zlim':[-1.05, 1.05],
            'saveFig':False,
            'saveGif':False,
            'figDir':'',
            'figName':'Chebyshev_basis_function',
            } 
        pltP.update(kwargs)
        fig_outputDir = pltP['figDir']
        cmap = plt.get_cmap('PRGn')
        
        nbins = M.shape[0]
        num_dim1, num_dim2 = M.shape[3:5]
        # Color map
        for l in range(nbins):
            plt.rcParams['figure.dpi'] = 250 
            # Create a 3D plot
            #since we can only visualize 2D basis function, the 3rd dimension is 
            #illustrated as time dimension
            fig, ax = plt.subplots(num_dim1, num_dim2, figsize= pltP['figSize'],\
                                   subplot_kw={'projection': '3d'})
            for i in range(num_dim1):
                for j in range(num_dim2): 
                    max_val = np.max([-np.min(M), np.max(M)])
                    ax[i, j].plot_surface(XG[:,:,l], ZG[:,:,l], YG[:,:,l],\
                        facecolors=cmap(((M[:,:,l,i,j] + max_val)/ (2*max_val+1e-10))),
                        rstride=1, cstride=1)
                    
                    ax[i, j].set_xticks([])
                    ax[i, j].set_xticklabels([])
                    ax[i, j].set_yticklabels([])
                    ax[i, j].set_zticks([])
                    ax[i, j].set_zticklabels([])
                    ax[i, j].set_xlim(pltP['xlim'])
                    ax[i, j].set_ylim(pltP['ylim'])
                    ax[i, j].set_zlim(pltP['zlim'])
                    ax[i, j].view_init(20,-75)
                    ax[i, j].set_aspect('equal')
                    ax[i, j].set_autoscale_on(False)
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None,\
                                wspace=-0.1, hspace=-0.1)
            plt.show()
            if pltP['saveFig'] and pltP['figDir'] != '':
                fig_name = pltP['figName'] + f'_slice{l:02}'
                full_path = os.path.join(pltP['figDir'],fig_name+'.png') 
                fig.savefig(full_path)   
        if pltP['saveFig'] and pltP['figDir'] != '' and pltP['saveGif']:
            # make a gif
            images = [img for img in os.listdir(pltP['figDir']) if img.startswith(fig_name[:-2])]
            images.sort()  # Sort the images by name (optional)
            image_list = [imageio.imread(f"{fig_outputDir}/{img}") for img in images]
            # Create a GIF
            gif_name = fig_name[:-8] + '.gif'
            output_path = f"{fig_outputDir}{gif_name}" 
            imageio.mimsave(output_path, image_list, fps=2)  
            
            
        
        