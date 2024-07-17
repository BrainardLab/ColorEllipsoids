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

#%%
def plot_2D_covMat(axes, fig, Sigmas_est, ellipses, xgrid_N_unit, **kwargs):
    """
    Sigmas_est (np.array, size: N x N x 2 x 2 where N is the number of grid poitns sampled finely)
    ellipses (np.array, size: n x n x 2 x 200 where n is the number of grid points sampled coarsely
              and 200 is the sampled points for each elliptical contour)
    """
    
    pltP = {
        'slc_idx_dim1':0,
        'slc_idx_dim2':0,
        'bds_W_unit': [-1,1],  
        'cmap': 'PRGn',
        'plane_2D': '',
        'title_list':[[r'$\sigma^2_{dim1}$', r'$\sigma_{(dim1,dim2)}$'],\
                      [r'$\sigma_{(dim2,dim1)}$', r'$\sigma^2_{dim2}$']],
        'cmap_bds': [],
        'fontsize': 12,
        'flag_rescale_axes_label':True,
        'saveFig':False,
        'figDir':'',
        'figName_ext':''} 
    pltP.update(kwargs)
    
    degree = 2 
    num_grid_fine = Sigmas_est.shape[0]
    num_grid = ellipses.shape[0]
    xgrid_W_unit = xgrid_N_unit*2 - 1
        
    #color map
    # Get a colormap
    cmap = plt.get_cmap(pltP['cmap'])
    # Calculate the maximum absolute value from the covariance matrices for color scaling
    if len(pltP['cmap_bds']) == 0:
        sig_bds = np.max(np.abs(Sigmas_est))
        cmap_bds = [-sig_bds, sig_bds]
    else:
        cmap_bds = pltP['cmap_bds']
            
    for i in range(degree):
        for j in range(degree):      
            # Plot the covariance matrix using a heatmap
            im = axes[i, j].imshow(Sigmas_est[:,:,i,j], cmap = cmap,\
                              vmin = cmap_bds[0], vmax = cmap_bds[1])
            # Calculate the scaled position for the horizontal line
            xgrid_scaled_p = xgrid_N_unit[num_grid-pltP['slc_idx_dim1']-1]*num_grid_fine
            # Draw horizontal line
            axes[i, j].plot([0, num_grid_fine], [xgrid_scaled_p,xgrid_scaled_p],\
                            c = 'grey',lw = 0.5)
            # Calculate the scaled position for the vertical line
            xgrid_scaled_q = xgrid_N_unit[pltP['slc_idx_dim2']]*num_grid_fine
            # Draw vertical line
            axes[i, j].plot([xgrid_scaled_q, xgrid_scaled_q],\
                            [0, num_grid_fine], c = 'grey',lw = 0.5)
            # Mark the intersection point
            axes[i, j].scatter(xgrid_scaled_q, xgrid_scaled_p, c = 'k', s = 10)
            # ticks and title
            axes[i, j].set_xticks(xgrid_N_unit[::2] * num_grid_fine)
            axes[i, j].set_yticks(xgrid_N_unit[::2] * num_grid_fine)
            if pltP['flag_rescale_axes_label']:
                axes[i, j].set_xticklabels(xgrid_N_unit[::2])
                axes[i, j].set_yticklabels(xgrid_N_unit[::2])
            else:
                axes[i, j].set_xticklabels(xgrid_W_unit[::2])
                axes[i, j].set_yticklabels(xgrid_W_unit[::2])                
            axes[i, j].set_xlim([0,num_grid_fine-1])
            axes[i, j].set_ylim([0,num_grid_fine-1])
            axes[i, j].set_title(pltP['title_list'][i][j])
    
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
            if pp < pltP['slc_idx_dim1'] or (pp == pltP['slc_idx_dim1'] \
                                             and qq <= pltP['slc_idx_dim2']):
                ax_ell.plot(ellipses[num_grid-1-pp,qq,0],\
                            ellipses[num_grid-1-pp,qq,1],c='k')
    # ticks and title
    ax_ell.set_xlim(pltP['bds_W_unit'])
    ax_ell.set_ylim(pltP['bds_W_unit'])
    ax_ell.set_xticks(xgrid_W_unit)
    ax_ell.set_yticks(xgrid_W_unit)
    if pltP['flag_rescale_axes_label']:
        ax_ell.set_xticklabels('{:.2f}'.format(x) for x in xgrid_N_unit)
        ax_ell.set_yticklabels('{:.2f}'.format(x) for x in xgrid_N_unit)
    else:
        ax_ell.set_xticklabels('{:.2f}'.format(x) for x in xgrid_W_unit)
        ax_ell.set_yticklabels('{:.2f}'.format(x) for x in xgrid_W_unit)
    ax_ell.grid(True, alpha=0.5)
    if pltP['plane_2D'] != '':
        ax_ell.set_xlabel(pltP['plane_2D'][0], fontsize = pltP['fontsize'])
        ax_ell.set_ylabel(pltP['plane_2D'][1], fontsize = pltP['fontsize'])
        ax_ell.set_title(pltP['plane_2D'], fontsize = pltP['fontsize'])
    else:
        ax_ell.set_xlabel('Dim 1', fontsize = pltP['fontsize'])
        ax_ell.set_ylabel('Dim 2', fontsize = pltP['fontsize'])
        ax_ell.set_title('2D plane', fontsize = pltP['fontsize'])        
    ax_ell.set_aspect('equal')
    # Show the plot
    plt.subplots_adjust(left=0.05, right=0.95, top=0.925, bottom=0.2)
    plt.show()
    fig_counter = pltP['slc_idx_dim1']*num_grid+pltP['slc_idx_dim2']
    fig_name = 'CovarianceMatrix_'+pltP['plane_2D']+pltP['figName_ext']+f'_{fig_counter:02d}.png' 
    if len(pltP['figDir']) !=0 and pltP['saveFig']:
        figDir = pltP['figDir']
        full_path = f"{figDir}{fig_name}"
        fig.savefig(full_path) 