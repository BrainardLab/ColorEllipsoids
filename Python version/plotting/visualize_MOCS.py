#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 15:08:22 2025

@author: fangfang
"""

import numpy as np
import matplotlib.pyplot as plt
import os

#%%
class MOCSTrialsVisualization():
    def __init__(self, fit_PMF_MOCS, color_thres, fig_dir='', save_fig=False, **kwargs):
        """
        Visualize models fits to 

        """
        self.fit_PMF_MOCS = fit_PMF_MOCS
        self.color_thres = color_thres
        self.fig_dir = fig_dir
        self.save_fig = save_fig
        # Default parameters for ellipsoid fitting. Can be overridden by kwargs.
        self.pltP = {
            'dpi':1024,
            } 
        # Update parameters with user-defined kwargs
        self.pltP.update(kwargs)
        
    def plot_PMF(self, slc_idx, pX1_Wishart_slc=None, pX1_indv_slc=None, ax=None, **kwargs):
        """
        Plots the psychometric function (PMF) for a selected condition.
        
        Parameters
        ----------
        slc_idx : int
            Index specifying which MOCS condition to visualize.
        pX1_Wishart_slc : np.ndarray, optional
            Probability of selecting X1 as the odd stimulus, predicted by the Wishart fit
            (jointly fitted across all reference stimuli).
        pX1_indv_slc : np.ndarray, optional
            Probability of selecting X1 as the odd stimulus, predicted by an individual Wishart fit
            (each ellipse fitted separately for each reference stimulus).
        ax : matplotlib.axes.Axes, optional
            Matplotlib axis on which to plot. If None, a new figure and axis are created.
        **kwargs : dict
            Additional keyword arguments for plot customization.
        
        Returns
        -------
        fig : matplotlib.figure.Figure
            The generated figure object.
        ax : matplotlib.axes.Axes
            The axis object containing the plot.
        """
        # Update plot parameters with method-specific settings and external configurations.
        method_specific_settings = {
            'fig_size': (4, 5.5),
            'alpha_CI_area': 0.2,
            'xref': None,
            'PMF_label': 'Best-fit psychometric function to MOCS trials',
            'CI_area_label': '95% bootstrap CI of PMF',
            'CI_thres_label': '95% bootstrap CI of threshold',
            'lw_Wishart': 0.2,
            'xlabel': 'Vector length between xref and x1 in W space',
            'show_ref_in_title': True,
            'fontsize': 10,
            'fig_name': 'Mahalanobis_distance'
        }
        
        self.pltP.update(method_specific_settings)
        self.pltP.update(kwargs)  
        plt.rcParams['font.sans-serif'] = ['Arial']
        
        # Extract the selected PMF condition
        slc_PMF_MOCS = self.fit_PMF_MOCS[slc_idx]
        
        # Create a new figure and axis if none are provided
        if ax is None:
            fig, ax = plt.subplots(figsize=self.pltP['fig_size'], dpi=self.pltP['dpi'])
        else:
            fig = ax.figure
        
        # Define color mapping
        if self.pltP['xref'] is not None:
            cmap_i = self.color_thres.M_2DWToRGB @ np.append(slc_PMF_MOCS.unique_stim[-1] + self.pltP['xref'], 1)
        else:
            cmap_i = np.array([0, 0, 0])
        
        # Plot the PMF curve
        ax.grid(True, color='grey', linewidth=0.1)
        ax.plot(slc_PMF_MOCS.fineVal, slc_PMF_MOCS.fine_pC, c=cmap_i, label=self.pltP['PMF_label'])
        
        # Scatter plot for observed data points (excluding the first filler point)
        ax.scatter(np.sort(slc_PMF_MOCS.unique_stim_L2norm)[1:],
                   np.sort(slc_PMF_MOCS.pC_perLevel)[1:],
                   c=cmap_i)
        
        # Fill 95% confidence interval area
        ax.fill_between(slc_PMF_MOCS.fineVal,
                        slc_PMF_MOCS.fine_pC_95btstCI[0],
                        slc_PMF_MOCS.fine_pC_95btstCI[1],
                        color=cmap_i, alpha=self.pltP['alpha_CI_area'],
                        label=self.pltP['CI_area_label'])
        
        # Add error bars for estimated threshold
        ax.errorbar(slc_PMF_MOCS.stim_at_targetPC,
                    slc_PMF_MOCS.target_pC,
                    xerr=slc_PMF_MOCS.stim_at_targetPC_95btstErr[:, np.newaxis],
                    c=cmap_i, lw=2, capsize=4,
                    label=self.pltP['CI_thres_label'])
        
        # Plot Wishart model predictions if available
        if pX1_Wishart_slc is not None:
            ax.plot(slc_PMF_MOCS.fineVal, pX1_Wishart_slc, color='k',
                    lw=self.pltP['lw_Wishart'], 
                    label='Predictions by Wishart Process model')
        
        # Plot individual Wishart fit if available
        if pX1_indv_slc is not None and not np.isnan(pX1_indv_slc[0]):
            ax.plot(slc_PMF_MOCS.fineVal, pX1_indv_slc, color='yellow',
                    lw=self.pltP['lw_Wishart'], 
                    label='Predictions by Wishart Process (individual fit) model')
        
        ax.set_xlabel(self.pltP['xlabel'])
        ax.set_ylabel('Percent correct')
        
        # Set title if reference stimulus is provided
        if self.pltP['show_ref_in_title'] and self.pltP['xref'] is not None:
            ax.set_title(f"Ref = [{np.round(self.pltP['xref'][0], 2)}, {np.round(self.pltP['xref'][1], 2)}]",
                         fontsize=self.pltP['fontsize'])
        
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.8), fontsize='small')
        plt.tight_layout()
        
        if self.save_fig:
            fig.savefig(os.path.join(self.fig_dir, self.pltP['fig_name']), bbox_inches='tight')            
        
        plt.show()
        return fig, ax


    def plot_comparison_thres(self, thres_Wishart, slope_mean, slope_CI, ax = None,
                              **kwargs):
        # Update plot parameters with method-specific settings and external configurations.
        method_specific_settings = {
            'fig_size': (4.5, 6),
            'x_bds':np.array([0, 0.14]),
            'alpha_CI_area': 0.1,
            'corr_coef_mean': None,
            'corr_coef_CI': None,
            'corr_text_loc': [0.025, 0.13],
            'slope_mean': None,
            'slope_CI':None,
            'xlabel': "Predicted vector length between xref and x1 for 66.7% correct \n(MOCS trials, Weibull function)",
            'ylabel': "Predicted vector length between xref and x1 for 66.7% correct \n(AEPsych trials, Wishart model)",
            'show_ref_in_title': True,
            'fontsize': 10,
            'fig_name': 'Mahalanobis_distance'
        }
        
        self.pltP.update(method_specific_settings)
        self.pltP.update(kwargs)  
        plt.rcParams['font.sans-serif'] = ['Arial']
        
        fig2, ax2 = plt.subplots(1, 1, figsize= self.pltP['fig_size'], dpi= self.pltP['dpi'])
        #add the best-fit line
        ax2.fill_between(self.pltP['x_bds'], self.pltP['x_bds'] * slope_CI[0], 
                        self.pltP['x_bds'] * slope_CI[1], 
                        color= [0,0,0], alpha= self.pltP['alpha_CI_area'],
                        label='95% bootstrap CI of a line fit')
        ax2.plot(self.pltP['x_bds'], self.pltP['x_bds']*slope_mean, 
                 color = 'gray', label = 'Best line fit')
        for n in range(len(self.fit_PMF_MOCS)):
            ax2.errorbar(
                self.fit_PMF_MOCS[n].stim_at_targetPC, 
                thres_Wishart[n],
                xerr= self.fit_PMF_MOCS[n].stim_at_targetPC_95btstErr[:, np.newaxis], 
                marker='o',
                c = [0,0,0],#cmap[n],
                ms = 7,#10
                lw = 2 #3
            )
        #add stats to the figure
        #0.035, 0.015
        if (self.pltP['corr_coef_mean'] is not None) and (self.pltP['corr_coef_CI'] is not None):
            ax2.text(self.pltP['corr_text_loc'],
                     f"Corr coef = {self.pltP['corr_coef_mean']:.2f}; 95% CI:"
                     f" [{self.pltP['corr_coef_CI'][0]:.2f}, {self.pltP['corr_coef_CI'][1]:.2f}]",
                     fontsize = 9.5)
        #0.035, 0.0075
        ax2.text(0.025, 0.123, f"Slope = {slope_mean:.2f}; 95% CI:"
                 f" [{slope_CI[0]:.2f}, {slope_CI[1]:.2f}]", fontsize = 9.5)
        # Add diagonal line for reference
        ax2.set_xlim(self.pltP['x_bds'])
        ax2.set_ylim(self.pltP['x_bds'])
        ax2.plot(self.pltP['x_bds'], self.pltP['x_bds'], ls='--', c='k', label = 'Identity line')
        ax2.set_xticks(np.linspace(*self.pltP['x_bds'],6))
        ax2.set_yticks(np.linspace(*self.pltP['x_bds'],6))

        # Set axis square and add grid
        ax2.set_aspect('equal', adjustable='box')  # Make the axis square
        ax2.grid(True, color='grey',linewidth=0.2)   # Add grid lines

        # Add labels, title, etc. (optional)
        ax2.set_xlabel(self.pltP['xlabel'])
        ax2.set_ylabel(self.pltP['ylabel'])

        # Add legend outside the plot at the bottom
        ax2.legend(
            loc='lower center', 
            bbox_to_anchor=(0.5, -0.45),  # Center below the plot, with some space
            ncol=1,  # Arrange legend entries in 2 columns
            fontsize='medium'
        )

        plt.tight_layout()
        # if load_actualData:
        #     fig2.savefig(output_figDir+f"{file_name[:-8]}_comparison_btw_MOCS_WishartPredictions.pdf",
        #                  format='pdf', bbox_inches='tight')
        # else:
        #     fig2.savefig(output_figDir+f"/{file_name[:-4]}_comparison_btw_MOCS_WishartPredictions.pdf",
        #                  format='pdf', bbox_inches='tight')    

    
#%%
def plot_MOCS_conditions(ndims, xref_unique, comp_unique, color_thres_data, **kwargs):
    """
    Plot MOCS (Method of Constant Stimuli) conditions in 2D or 3D.

    Parameters
    ----------
    ndims : int
        Dimensionality of the color space.
        - 2: 2D plane
        - 3: 3D RGB cube
    xref_unique : np.ndarray, shape (M, ndims)
        Reference locations for all MOCS conditions, where M is the number of conditions.
    comp_unique : np.ndarray, shape (M, N, ndims)
        Comparison stimuli for each reference location, where:
        - M is the number of reference locations
        - N is the number of comparison levels
    color_thres_data : object
        Object containing color transformation methods.
    **kwargs : dict
        Additional plotting parameters.

    Returns
    -------
    None
    """
    pltP = {
        'fig_size': (4, 4),  # Default figure size
        'ticks': np.linspace(-0.6, 0.6, 5),  # Tick marks on axes
        'xlabel': 'Wishart dimension 1',
        'ylabel': 'Wishart dimension 2',
        'zlabel': 'Wishart dimension 3',
        'title': 'Isoluminant plane',  # Default title for 2D plots
        'fontsize': 10,
        'fig_name': '',
        'output_dir': '',
        'save_fig': False
    }
    
    pltP.update(kwargs) 
    plt.rcParams['font.family'] = 'Arial'
    
    if ndims == 2:
        # Create a 2D figure
        fig, ax = plt.subplots(figsize=pltP['fig_size'], dpi=1024)
        
        for idx_slc in range(len(xref_unique)):
            xref = xref_unique[idx_slc]
            comp = comp_unique[idx_slc]
            
            # Plot comparison stimuli
            ax.scatter(comp[:, 0], comp[:, 1], marker='.', color='k', s=1)
            ax.plot([xref[0], comp[-1, 0]],
                    [xref[1], comp[-1, 1]], lw=0.4, color='gray')
            ax.scatter(*comp[-1,:], marker='o', facecolor='none', edgecolor='r', alpha=0.5, s=10)
            
            # Color mapping for reference points
            cmap_n = color_thres_data.M_2DWToRGB @ np.append(xref, 1)
            ax.scatter(*xref, color=cmap_n, marker='+', lw=3, s=100)
            
        # Set plot limits and labels
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_xticks(pltP['ticks'])
        ax.set_yticks(pltP['ticks'])
        ax.set_aspect('equal', adjustable='box')  # Ensure equal scaling
        ax.grid(True, linewidth=0.2)
        ax.set_xlabel(pltP['xlabel'])
        ax.set_ylabel(pltP['ylabel'])
        ax.set_title(pltP['title'])
        plt.show()
    
    else:
        # Create a 3D figure
        fig = plt.figure(figsize=pltP['fig_size'], dpi=1024)
        ax = fig.add_subplot(111, projection='3d')
        
        for idx_slc in range(len(xref_unique)):
            xref = xref_unique[idx_slc]
            comp = comp_unique[idx_slc]
            
            # Mapping reference color in RGB cube
            color_map_ref = color_thres_data.W_unit_to_N_unit(xref)
            
            # Plot reference stimuli
            ax.scatter(*xref, c=color_map_ref, marker='+', lw=3, s=100)
            
            # Plot comparison stimuli
            ax.scatter(comp[:, 0], comp[:, 1], comp[:, 2], marker='.', color='k', s=1)
            ax.plot(comp[:, 0], comp[:, 1], comp[:, 2], lw=0.4, color='gray')
            ax.scatter(*comp[-1, :], marker='o', facecolor='none', edgecolor='r', alpha=0.5, s=10)
        
        # Set plot limits, labels, and aspect ratio
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xticks(pltP['ticks'])
        ax.set_yticks(pltP['ticks'])
        ax.set_zticks(pltP['ticks'])
        ax.set_box_aspect([1, 1, 1])  # Maintain equal aspect ratio
        ax.grid(True, linewidth=0.1)
        ax.set_xlabel(pltP['xlabel'])
        ax.set_ylabel(pltP['ylabel'])
        ax.set_zlabel(pltP['zlabel'])
        ax.set_title(pltP['title'])
        
        # Adjust layout to prevent labels from getting cut off
        fig.tight_layout(pad=4.0)
        fig.set_size_inches(4.5, 5)
    
    # Save the figure if required
    if pltP['save_fig'] and os.path.exists(pltP['output_dir']) and pltP['fig_name']:
        fig.savefig(os.path.join(pltP['output_dir'], pltP['fig_name']))
    plt.show()

    
    