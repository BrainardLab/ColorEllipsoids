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
import ast
sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/"+\
                "Python version/wishart_visualization/")
from wishart_plotting import wishart_model_basics_visualization   

import jax
jax.config.update("jax_enable_x64", True)

#load functions from other modules
sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids")
from core import model_predictions
sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/"+\
                "Python version")
from simulations_CIELab import PointsOnEllipseQ
sys.path.append("/Users/fangfang/Documents/GitHub/META_AEPsych/Tests_simulatedExperiments/")
from plot_slice_4d import extract_sliced_xy

# load AEPsych
path_str = '/Users/fangfang/Documents/MATLAB/toolboxes/aepsych-main'
sys.path.append(path_str)
from aepsych.plotting import plot_slice, plot_strat

#%%
class AEPsych_predictions_visualization(wishart_model_basics_visualization):
    def __init__(self, fig_dir='', save_fig=False, save_gif=False):
        """
        Initialize an instance of sampling_ref_comp_pair_visualization, a subclass of
        wishart_model_basics_visualization, which extends its functionality for specific
        visualization tasks related to sampling reference and comparison stimulus pairs.

        """
        super().__init__(fig_dir, save_fig, save_gif)
        # Override the default color map used in plots
        self.pltP['cmap'] = 'twilight'
    
    def _rgb_to_N_unit(self, rgb, round_decimal = 2):
        """
        Normalize RGB values from a standard 0-255 range to a 0-1 range.

        Parameters:
        rgb : array_like
            The original RGB values.
        round_decimal : int, optional
            The number of decimal places to round to.

        Returns:
        array_like
            Normalized RGB values rounded to the specified precision.
        """
        rgb_N_unit = np.around(rgb/255, round_decimal)
        return rgb_N_unit
    
    def _rgb_to_W_unit(self, rgb, round_decimal = 2):
        """
        Normalize RGB values from a standard 0-255 range to a -1 to 1 range.

        Parameters:
        rgb : array_like
            The original RGB values.
        round_decimal : int, optional
            The number of decimal places to round to.

        Returns:
        array_like
            Normalized RGB values rounded to the specified precision.
        """
        rgb_W_unit = np.around((rgb/255) * 2 - 1, round_decimal)
        return rgb_W_unit
        
    def _convert_2Dcov_to_points_on_ellipse(self, cov2D, scaler = 1, ref_x = 0, ref_y = 0):
        """
        Convert a 2D covariance matrix into points on an ellipse, scaled and recentered.
        
        Parameters:
        cov2D : array_like
            2x2 covariance matrix.
        scaler : float, optional
            Scaling factor for the size of the ellipse.
        ref_x : float, optional
            X-coordinate to recenter the ellipse.
        ref_y : float, optional
            Y-coordinate to recenter the ellipse.
        
        Returns:
        tuple
            Two arrays representing the x and y coordinates of the ellipse points.
        """

        #axes loength and rotation angle
        _,_,axisLength, rotAngle = model_predictions.covMat_to_ellParamsQ(cov2D)
        #poitns on ellipses
        ell_2d_x, ell_2d_y = PointsOnEllipseQ(*axisLength, rotAngle, 0,0)
        ell_2d_x_scaled = ell_2d_x*scaler
        ell_2d_y_scaled = ell_2d_y*scaler
        ell_2d_x_recentered = ell_2d_x_scaled + ref_x
        ell_2d_y_recentered = ell_2d_y_scaled + ref_y
        return ell_2d_x_recentered, ell_2d_y_recentered

    def _update_subplots_axes_labels(self, ax, num_grid_pts, row_idx, col_idx,
                                     unit_true_x, unit_show_x, unit_true_y,
                                     unit_show_y, fixed_dim, fixed_val):
        """
        Update subplot axes labels, ticks, and titles for clarity in multi-panel visualizations.

        Parameters:
        ax : matplotlib.axes.Axes
            The axis of the subplot to update.
        num_grid_pts : int
            The number of rows or columns in the subplot grid.
        row_idx : int
            The index of the current row in the subplot grid.
        col_idx : int
            The index of the current column in the subplot grid.
        unit_true_x : array_like
            The positions for x-axis ticks.
        unit_show_x : array_like
            The labels for the x-axis ticks.
        unit_true_y : array_like
            The positions for y-axis ticks.
        unit_show_y : array_like
            The labels for the y-axis ticks.
        fixed_dim : int
            The dimension that remains constant in this subplot.
        fixed_val : float
            The value of the fixed dimension for labeling.

        Notes:
        - Only the last row displays x-axis ticks and labels.
        - Only the first column displays y-axis ticks and labels.
        - Titles are displayed at the top center subplot only.
        """
        
        varying_dim = list(range(self.ndims)) 
        varying_dim.remove(fixed_dim)
        if row_idx == num_grid_pts -1:
            ax.set_xticks(unit_true_x)
            ax.set_xticklabels(unit_show_x)
            ax.set_xlabel(self.pltP['str_dim'][varying_dim[0]])
        else:
            ax.set_xlabel(''); ax.set_xticks([])
        if col_idx == 0:
            ax.set_yticks(unit_true_y)
            ax.set_yticklabels(unit_show_y)
            ax.set_ylabel(self.pltP['str_dim'][varying_dim[1]])
        else:
            ax.set_ylabel(''); ax.set_yticks([])
        if col_idx == int(np.floor(num_grid_pts/2)) and row_idx == 0:
            ax.set_title(self.pltP['str_dim'][fixed_dim] +f' = {fixed_val:.2f}')
        else:
            ax.set_title('')
            
    def plot_2D_predictions(self, server, pseudo_order, xgrid, ax = None, **kwargs):
        self.ndims = 2
        # Initialize method-specific settings and update with any additional configurations provided.
        method_specific_settings = {
            'visualize_gt': False,
            'gt_2D_ellipse': np.array([]),
            'xlabel': 'dim1',
            'ylabel': 'dim2',
            'yes_label':'correctly identifying\nthe odd stimulus',
            'no_label': 'incorrect responses',
            'fig_name':''}
        
        self.pltP.update(method_specific_settings)
        self.pltP.update(kwargs)  # Apply any external configurations
        
        plt.set_cmap(self.pltP['cmap'])
        
        if ax is None:
            fig, ax = plt.subplots(dpi= self.pltP['dpi'], figsize=(5,4))
        else:
            fig = ax.figure
        
        # Select the configuration and strategy based on a pseudo-random order
        config_org = server._configs[pseudo_order]
        server_org = server._strats[pseudo_order]
        
        # Parse the lower and upper bounds from the configuration's common section
        lb_bds = ast.literal_eval(config_org['common']['lb'])
        ub_bds = ast.literal_eval(config_org['common']['ub'])
        
        # Set the y and x ticks with appropriate scaling and labels
        xticks = np.linspace(lb_bds[0], ub_bds[0], 5)
        yticks = np.linspace(lb_bds[1], ub_bds[1], 5)
        self._update_axes_labels(ax, xticks, self._rgb_to_N_units(xticks))
        self._update_axes_labels(ax, yticks, self._rgb_to_N_units(yticks))
        
        if self.pltP['visualize_gt'] and self.pltP['gt_2D_ellipse'].shape[0] != 0:
            # Calculate the center of the RGB bounds and convert it to a normalized range
            center_rgb = (np.array(ub_bds) - np.array(lb_bds))/ 2 + np.array(lb_bds)
            center = self._rgb_to_W_units(center_rgb)
            
            # Find indices of the closest grid points to the center in the x and y directions
            x_idx = np.argmin(np.abs(np.unique(xgrid) - center[1]))
            y_idx = np.argmin(np.abs(np.unique(xgrid) - center[0]))
            
            # Recenter the grid points around the calculated center, scale and adjust for plotting
            gt_recenter = self.pltP['gt_2D_ellipse'][x_idx,y_idx] - \
                np.reshape(xgrid[x_idx,y_idx],(2,1))
            gt_recenter = gt_recenter/2 * 255 + np.reshape(center_rgb,(2,1))
            ax.plot(gt_recenter[0], gt_recenter[1], c = 'white', lw = 4)
        
        output_filename = self.fig_dir +  self.pltP['fig_name']
        plot_strat(server_org, ax = ax, target_level = None, gridsize = 100,\
               xlabel = self.pltP['xlabel'], ylabel = self.pltP['ylabel'],\
               save_path = output_filename, include_colorbar=True, show=False,\
               yes_label = self.plt['yes_label'], no_label = self.plt['no_label'])
            
        return fig, ax
    
            
    def plot_3D_predictions(self, strat, ref, indices, lb_comp, ub_comp,
                            fixed_dim, ax = None, **kwargs):
        """
        Visualizes predictions in a 3D space by displaying a 2D slice (subplot) from a
        set of 3D experiment results. This function focuses on displaying AEPsych's
        threshold estimates, optional ground truth data, and the trials selected by
        AEPsych.

        Parameters:
        ----------
        ax : matplotlib.axes.Axes
            Array of subplot axes for displaying the results.
        fig : matplotlib.figure.Figure
            Figure object where the subplots are drawn.
        strat : object
            AEPsych strategy object containing the model and experiment data.
        ref : np.array
            An array of shape (N, 3), where N is the number of interleaved experiments
            specifying the reference locations for each experiment.
        indices : np.array
            Array of indices of shape (M,), where M is num_grid_pts**2, representing
            the order of strategies shuffled during experiments.
        lb_comp : np.array
            Lower bounds of the comparison stimuli across three dimensions.
        ub_comp : np.array
            Upper bounds of the comparison stimuli across three dimensions.
        fixed_dim : int
            Specifies the fixed dimension across color spaces (e.g., R, G, or B).

        """
        self.ndims = 3
        # Initialize method-specific settings and update with any additional configurations provided.
        method_specific_settings = {
            'str_dim': ['R', 'G', 'B'],
            'str_plane': ['GB plane', 'RB plane', 'RG plane'],
            'visualize_gt': False,
            'gt_2D_ellipse': np.array([]),
            'gt_2D_ellipse_scaler':5,
            'gt_ls':'-',
            'gt_color':np.array([240,128,128])/255,
            'gt_lw':3,
            'visualize_data': True,
            'data_yes_color': np.array([107,142,35])/255,
            'data_no_color': np.array([128,0,0])/255,
            'data_mc': 20,
            'data_marker':'.',
            'data_alpha':1,
            'flag_rescale_axes_label':True,
            'fontsize':8,
            'fig_name':'RandomSamples'}
        
        self.pltP.update(method_specific_settings)
        self.pltP.update(kwargs)  # Apply any external configurations
        plt.set_cmap(self.pltP['cmap'])

        num_subplots = np.size(indices)
        # Determine grid size from number of indices.
        num_grid_pts = int(np.sqrt(num_subplots))
        # Determine varying dimensions by excluding the fixed one.
        varying_dim = list(range(self.ndims)) 
        varying_dim.remove(fixed_dim)
        
        if ax is None:
            fig, ax = plt.subplots(nrows = num_grid_pts, ncols = num_grid_pts,\
                                    dpi= self.pltP['dpi'], figsize=(9.5, 9.5))
        else:
            fig = ax.figure
        
        for idx_n, n in enumerate(indices):
            # Calculate the row and column index for each subplot based on the shuffled indices.
            col_idx = idx_n // num_grid_pts
            row_idx = num_grid_pts - (idx_n % num_grid_pts) -1 
        
            # Plot the estimated threshold from AEPsych strategy.
            plot_slice(ax[row_idx, col_idx], strat[n], self.pltP['str_dim'],\
                       slice_dim = fixed_dim, slice_val = ref[n][fixed_dim],\
                       vmin = 1/3, vmax = 1, contour_levels = [], gridsize = 100)
            
            # Compute and plot the ground truth contour if enabled.
            if self.pltP['visualize_gt'] and self.pltP['gt_2D_ellipse'].shape[0] != 0:
                ell_2d_x, ell_2d_y = self._convert_2Dcov_to_points_on_ellipse(\
                        self.pltP['gt_2D_ellipse'][n], scaler = self.pltP['gt_2D_ellipse_scaler']*255,\
                        ref_x = ref[n][varying_dim[0]], ref_y = ref[n][varying_dim[1]])            
                # Plot the ground truth contour on the same subplot
                ax[row_idx, col_idx].plot(ell_2d_x, ell_2d_y, c= self.pltP['gt_color'],\
                                          linestyle = self.pltP['gt_ls'],\
                                          lw = self.pltP['gt_lw'])
                
            # Plot the data points (responses) nearby if enabled.
            if self.pltP['visualize_data']:
                slice_dim = dict(zip([fixed_dim],[ref[n][fixed_dim]]))
                dp_dim1, dp_dim2 = extract_sliced_xy(strat[n], remaining_ax= slice_dim,\
                                                     tolpct=0.05)
                yes_resp = np.vstack((dp_dim1[dp_dim2 == 1, varying_dim[0]].numpy(),\
                                 dp_dim1[dp_dim2 == 1, varying_dim[1]].numpy()))
                no_resp = np.vstack((dp_dim1[dp_dim2 == 0, varying_dim[0]].numpy(),\
                                dp_dim1[dp_dim2 == 0, varying_dim[1]].numpy()))
    
                ax[row_idx, col_idx].scatter(yes_resp[0,:], yes_resp[1,:], 
                                             marker= self.pltP['data_marker'], \
                                             color = self.pltP['data_yes_color'],\
                                             alpha= self.pltP['data_alpha'],\
                                             label='yes', s= self.pltP['data_mc'])
                ax[row_idx, col_idx].scatter(no_resp[0,:], no_resp[1,:],\
                                             marker= self.pltP['data_marker'], \
                                             color = self.pltP['data_no_color'],\
                                             alpha= self.pltP['data_alpha'],\
                                             label='no', s= self.pltP['data_mc'])

            # Set tick labels and labels based on position in grid
            lb_n = lb_comp[n]
            ub_n = ub_comp[n]
            ticks_x = np.array([lb_n[varying_dim[0]], ref[n][varying_dim[0]], ub_n[varying_dim[0]]])
            ticks_y = np.array([lb_n[varying_dim[1]], ref[n][varying_dim[1]], ub_n[varying_dim[1]]])
            ticks_x_show = self._rgb_to_N_unit(ticks_x)
            ticks_y_show = self._rgb_to_N_unit(ticks_y)
            ref_n = self._rgb_to_N_unit(ref[n][fixed_dim])
            self._update_subplots_axes_labels(ax[row_idx, col_idx], num_grid_pts,\
                                              row_idx, col_idx,\
                                              ticks_x, ticks_x_show, ticks_y, \
                                              ticks_y_show, fixed_dim, ref_n)
        # Save the figure if the directory is set and saving is enabled.
        if len(self.fig_dir) !=0 and self.save_fig:
            str_ext = '_'+ self.pltP['str_plane'][fixed_dim]+'_'+ \
                self.pltP['str_dim'][fixed_dim] +str(ref_n)
            self._save_figure(fig, self.pltP['fig_name'] + str_ext)
            
        return fig, ax

