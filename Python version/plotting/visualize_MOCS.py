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
    def __init__(self, fit_PMF_MOCS, fig_dir='', save_fig=False, **kwargs):
        """
        Visualize models fits to 

        """
        self.fit_PMF_MOCS = fit_PMF_MOCS
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
            'cmap': np.array([0, 0, 0]),
            'xref': None,
            'filler_pts': None,
            'yticks': [0.33, 0.67, 1],
            'PMF_label': 'Best-fit psychometric function to MOCS trials',
            'CI_area_label': '95% bootstrap CI of PMF',
            'CI_thres_label': '95% bootstrap CI of threshold',
            'lw_Wishart': 0.2,
            'xlabel': 'Euclidean distance between ref and comp in W space',
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
        
        # Plot the PMF curve
        ax.grid(True, color='grey', linewidth=0.1)
        ax.plot(slc_PMF_MOCS.fineVal, slc_PMF_MOCS.fine_pC, c=self.pltP['cmap'],
                label=self.pltP['PMF_label'])
        
        # Scatter plot for observed data points 
        ax.scatter(slc_PMF_MOCS.unique_stim_L2norm,
                   slc_PMF_MOCS.pC_perLevel,
                   c=self.pltP['cmap'])
        #(excluding the first filler point)
        if self.pltP['filler_pts'] is not None:
            ax.scatter(*self.pltP['filler_pts'], c = 'white')
        
        # Fill 95% confidence interval area
        ax.fill_between(slc_PMF_MOCS.fineVal,
                        slc_PMF_MOCS.fine_pC_95btstCI[0],
                        slc_PMF_MOCS.fine_pC_95btstCI[1],
                        color=self.pltP['cmap'], alpha=self.pltP['alpha_CI_area'],
                        label=self.pltP['CI_area_label'])
        
        # Add error bars for estimated threshold
        ax.errorbar(slc_PMF_MOCS.stim_at_targetPC,
                    slc_PMF_MOCS.target_pC,
                    xerr=slc_PMF_MOCS.stim_at_targetPC_95btstErr[:, np.newaxis],
                    c=self.pltP['cmap'], lw=2, capsize=4,
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
        ax.set_yticks(self.pltP['yticks'])
        
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


    def plot_comparison_thres(self, thres_Wishart, slope_mean, slope_CI, xref_unique,
                              ax = None,thres_Wishart_CI=None, **kwargs):
        # Update plot parameters with method-specific settings and external configurations.
        method_specific_settings = {
            'fig_size': (4.5, 6),
            'bds':np.array([0, 0.14]), 
            'alpha_CI_area': 0.1,
            'corr_coef_mean': None,
            'corr_coef_CI': None,
            'cmap': None,
            'corr_text_loc': [0.025, 0.13],
            'slope_text_loc': [0.025, 0.123],
            'ms': 7,
            'lw': 2,
            'alpha':1,
            'xlabel': "Predicted Euclidean distance between ref and comp \nfor 66.7% correct (MOCS trials, Weibull function)",
            'ylabel': "Predicted Euclidean distance between ref and comp \nfor 66.7% correct (AEPsych trials, Wishart model)",
            'show_ref_in_title': True,
            'fontsize': 9.5,
            'fig_name': ''
        }
        
        self.pltP.update(method_specific_settings)
        self.pltP.update(kwargs)  
        plt.rcParams['font.sans-serif'] = ['Arial']
        
        # Create a new figure and axis if none are provided
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize= self.pltP['fig_size'], dpi= self.pltP['dpi'])
        else:
            fig = ax.figure
            
        #add the best-fit line
        ax.fill_between(self.pltP['bds'], self.pltP['bds'] * slope_CI[0], 
                        self.pltP['bds'] * slope_CI[1], 
                        color= [0,0,0], alpha= self.pltP['alpha_CI_area'],
                        label='95% bootstrap CI of a line fit')
        ax.plot(self.pltP['bds'], self.pltP['bds']*slope_mean, 
                 color = 'gray', label = 'Best line fit')
        for n in range(len(self.fit_PMF_MOCS)):
            if self.pltP['cmap'] is None:
                cmap_n = np.array([0,0,0])
            else:
                cmap_n = self.pltP['cmap'][n]
            
            if thres_Wishart_CI is not None:
                ax.errorbar(
                    self.fit_PMF_MOCS[n].stim_at_targetPC, 
                    thres_Wishart[n],
                    xerr= self.fit_PMF_MOCS[n].stim_at_targetPC_95btstErr[:, np.newaxis], 
                    yerr= thres_Wishart_CI[n][:, np.newaxis],
                    marker='o',
                    c = cmap_n,
                    alpha = self.pltP['alpha'],
                    ms = self.pltP['ms'],#10
                    lw = self.pltP['lw'] #3
                )
            else:
                ax.errorbar(
                    self.fit_PMF_MOCS[n].stim_at_targetPC, 
                    thres_Wishart[n],
                    xerr= self.fit_PMF_MOCS[n].stim_at_targetPC_95btstErr[:, np.newaxis], 
                    marker='o',
                    c = cmap_n,
                    alpha = self.pltP['alpha'],
                    ms = self.pltP['ms'],#10
                    lw = self.pltP['lw'] #3
                )                
        #add stats to the figure
        if (self.pltP['corr_coef_mean'] is not None) and (self.pltP['corr_coef_CI'] is not None):
            ax.text(*self.pltP['corr_text_loc'],
                     f"Corr coef = {self.pltP['corr_coef_mean']:.2f}; 95% CI:"
                     f" [{self.pltP['corr_coef_CI'][0]:.2f}, {self.pltP['corr_coef_CI'][1]:.2f}]")
        ax.text(*self.pltP['slope_text_loc'],
                     f"Slope = {slope_mean:.2f}; 95% CI: [{slope_CI[0]:.2f}, {slope_CI[1]:.2f}]")
        # Add diagonal line for reference
        ax.set_xlim(self.pltP['bds'])
        ax.set_ylim(self.pltP['bds'])
        ax.plot(self.pltP['bds'], self.pltP['bds'], ls='--', c='k', label = 'Identity line')
        ax.set_xticks(np.linspace(*self.pltP['bds'],6))
        ax.set_yticks(np.linspace(*self.pltP['bds'],6))

        # Set axis square and add grid
        ax.set_aspect('equal', adjustable='box')  # Make the axis square
        ax.grid(True, color='grey',linewidth=0.2)   # Add grid lines

        # Add labels, title, etc. (optional)
        ax.set_xlabel(self.pltP['xlabel'])
        ax.set_ylabel(self.pltP['ylabel'])

        # Add legend outside the plot at the bottom
        ax.legend(
            loc='lower center', 
            bbox_to_anchor=(0.5, -0.45),  # Center below the plot, with some space
            ncol=1,  # Arrange legend entries in 2 columns
            fontsize='medium'
        )

        plt.tight_layout()
        
        if self.save_fig:
            fig.savefig(os.path.join(self.fig_dir, self.pltP['fig_name']), bbox_inches='tight')            
        
        plt.show()
        return fig, ax

    
#%%
def plot_MOCS_conditions(ndims, xref_unique, comp_unique, color_thres_data,
                         ax = None, **kwargs):
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
        'xlabel': 'Wishart space dimension 1',
        'ylabel': 'Wishart space dimension 2',
        'zlabel': 'Wishart space dimension 3',
        'title': 'Isoluminant plane',  # Default title for 2D plots
        'fontsize': 10,
        'ref_ms': 100,
        'ref_lw':3,
        'comp_ms':10,
        'easyTrials_highlight': True,
        'fig_name': '',
        'output_dir': '',
        'save_fig': False
    }
    
    pltP.update(kwargs) 
    plt.rcParams['font.family'] = 'Arial'
    
    if ndims == 2:
        # Create a new figure and axis if none are provided
        if ax is None:
            fig, ax = plt.subplots(figsize=pltP['fig_size'], dpi=1024)
        else:
            fig = ax.figure
        
        for idx_slc in range(len(xref_unique)):
            xref = xref_unique[idx_slc]
            comp = comp_unique[idx_slc]
            
            # Plot comparison stimuli
            ax.scatter(comp[:, 0], comp[:, 1], marker='.', color='k', s=pltP['comp_ms'])
            ax.plot([xref[0], comp[-1, 0]],
                    [xref[1], comp[-1, 1]], lw=0.4, color='gray')
            if pltP['easyTrials_highlight']:
                ax.scatter(*comp[-1,:], marker='o', facecolor='none', edgecolor='r', alpha=0.5, s=10)
            
            # Color mapping for reference points
            cmap_n = color_thres_data.M_2DWToRGB @ np.append(xref, 1)
            ax.scatter(*xref, color=cmap_n, marker='+', lw=pltP['ref_lw'], s=pltP['ref_ms'])
            
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
        plt.tight_layout()
        #plt.show()
    
    else:
        # Create a 3D figure
        # Create a new figure and axis if none are provided
        if ax is None:
            fig = plt.figure(figsize=pltP['fig_size'], dpi=1024)
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.figure
        
        for idx_slc in range(len(xref_unique)):
            xref = xref_unique[idx_slc]
            comp = comp_unique[idx_slc]
            
            # Mapping reference color in RGB cube
            color_map_ref = color_thres_data.W_unit_to_N_unit(xref)
            
            # Plot reference stimuli
            ax.scatter(*xref, c=color_map_ref, marker='+', lw=3, s=100)
            
            # Plot comparison stimuli
            
            ax.scatter(comp[:, 0], comp[:, 1], comp[:, 2], marker='.', color='k', s= pltP['comp_ms'])
            ax.plot(comp[:, 0], comp[:, 1], comp[:, 2], lw=0.4, color='gray')
            if pltP['easyTrials_highlight']:
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
    
    return fig, ax
    