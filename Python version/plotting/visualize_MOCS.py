#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 15:08:22 2025

@author: fangfang
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Union
from plotting.wishart_plotting import PlottingTools, PlotSettingsBase

@dataclass
class PlotPMFSettings(PlotSettingsBase):
    fig_size: Tuple[float, float] = (4, 5.5)
    cmap: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    filler_pts: Optional[np.ndarray] = None
    yticks: List[float] = field(default_factory=lambda: [0.33, 0.67, 1])
    PMF_label: str = 'Best-fit psychometric function to MOCS trials'
    CI_area_label: str = '95% bootstrap CI of PMF'
    CI_thres_label: str = '95% bootstrap CI of threshold'
    Wishart_pred_lw: float = 0.2
    Wishart_pred_lc: Union[str, np.ndarray, List[float]] = field(default_factory=lambda: [0,0,0])
    Wishart_pred_label: str = 'Predictions by Wishart Process model'
    Wishart_indv_pred_label: str = 'Predictions by Wishart Process (individual fit) model'
    xlabel: str = 'Euclidean distance between ref and comp in W space'
    ylabel: str = 'Percent correct'
    show_ref_in_title: bool = True
    fig_name: str = 'Mahalanobis_distance'
    
@dataclass
class PlotThresCompSettings:
    fig_size: Tuple[float, float] = (4.5, 6)
    bds: np.ndarray = field(default_factory=lambda: np.array([0, 0.14]))
    cmap: Optional[Union[np.ndarray, List[float]]] = None
    corr_text_loc: List[float] = field(default_factory=lambda: [0.025, 0.13])
    slope_text_loc: List[float] = field(default_factory=lambda: [0.025, 0.123])
    ms: int = 7
    lw: int = 2
    alpha: float = 1.0
    marker: str = 'o'
    xlabel: str = ("Predicted Euclidean distance between ref and comp \n"
                   "for 66.7% correct (MOCS trials, Weibull function)")
    ylabel: str = ("Predicted Euclidean distance between ref and comp \n"
                   "for 66.7% correct (AEPsych trials, Wishart model)")
    num_ticks: int = 6
    line_fit_CI_alpha: float = 0.1
    line_fit_CI_lc: Union[str, np.ndarray, List[float]] = field(default_factory=lambda: [0,0,0])
    line_fit_CI_label: str = '95% bootstrap CI of a line fit'
    line_fit_mean_lc: Union[str, np.ndarray, List[float]] = 'grey'
    line_fit_mean_label: str = 'Best line fit'
    show_ref_in_title: bool = True
    fig_name: str = ''

@dataclass
class PlotCondSettings:
    fig_size: Tuple[float, float] = (4, 4)  # Default figure size
    ticks: np.ndarray = field(default_factory=lambda: np.linspace(-0.6, 0.6, 5))  # Tick marks on axes
    title: str = 'Isoluminant plane'  # Default title for 2D plots
    ref_ms: int = 100  # Marker size for reference stimulus
    ref_lw: int = 3    # Line width for reference
    ref_marker: str = '+'
    comp_ms: int = 10  # Marker size for comparison stimulus
    comp_marker: str = '.'
    comp_mc: Union[str, np.ndarray, List[float]] = 'k'
    comp_lc: Union[str, np.ndarray, List[float]] = 'grey'
    comp_lw: float = 0.4
    catch_marker: str = 'o'
    catch_alpha: float = 0.5
    catch_ms: int = 10
    catch_ec: Union[str, np.ndarray, List[float]] = 'r'
    easyTrials_highlight: bool = True  # Whether to highlight easy trials
    fig_name: str = ''  # Filename to save the figure
    
#%%
class MOCSTrialsVisualization(PlottingTools):
    def __init__(self, fit_PMF_MOCS, settings: PlotSettingsBase, 
                 save_fig=False, save_format = 'pdf'):
        super().__init__(settings, save_fig, save_format)
        self.fit_PMF_MOCS = fit_PMF_MOCS
        plt.rcParams['font.sans-serif'] = settings.fontstyle
        plt.rcParams['font.size'] = settings.fontsize
        
    def plot_PMF(self, slc_idx, settings: PlotPMFSettings, xref = None,
                 pX1_Wishart_slc=None, pX1_indv_slc=None, ax=None):
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
        
        # Extract the selected PMF condition
        slc_PMF_MOCS = self.fit_PMF_MOCS[slc_idx]
        
        # Create a new figure and axis if none are provided
        if ax is None:
            fig, ax = plt.subplots(figsize=settings.fig_size, dpi=settings.dpi)
        else:
            fig = ax.figure
        
        # Plot the PMF curve
        ax.grid(True, color='grey', linewidth=0.1)
        ax.plot(slc_PMF_MOCS.fineVal, slc_PMF_MOCS.fine_pC, c=settings.cmap,
                label=settings.PMF_label)
        
        # Scatter plot for observed data points 
        ax.scatter(slc_PMF_MOCS.unique_stim_L2norm, slc_PMF_MOCS.pC_perLevel,
                   c=settings.cmap)
        #(excluding the first filler point)
        if settings.filler_pts is not None:
            ax.scatter(*settings.filler_pts, c = 'white')
        
        # Fill 95% confidence interval area
        ax.fill_between(slc_PMF_MOCS.fineVal,
                        slc_PMF_MOCS.fine_pC_95btstCI[0],
                        slc_PMF_MOCS.fine_pC_95btstCI[1],
                        color=settings.cmap, alpha=settings.alpha_CI_area,
                        label=settings.CI_area_label)
        
        # Add error bars for estimated threshold
        ax.errorbar(slc_PMF_MOCS.stim_at_targetPC,
                    slc_PMF_MOCS.target_pC,
                    xerr=slc_PMF_MOCS.stim_at_targetPC_95btstErr[:, np.newaxis],
                    c=settings.cmap, lw=2, capsize=4,
                    label=settings.CI_thres_label)
        
        # Plot Wishart model predictions if available
        if pX1_Wishart_slc is not None:
            ax.plot(slc_PMF_MOCS.fineVal, pX1_Wishart_slc, color= settings.Wishart_pred_lc,
                    lw=settings.Wishart_pred_lw, 
                    label=settings.Wishart_pred_label)
        
        # Plot individual Wishart fit if available
        if pX1_indv_slc is not None and not np.isnan(pX1_indv_slc[0]):
            ax.plot(slc_PMF_MOCS.fineVal, pX1_indv_slc, color='yellow',
                    lw=settings.lw_Wishart, 
                    label= settings.Wishart_indv_pred_label)
        
        ax.set_xlabel(settings.xlabel)
        ax.set_ylabel(settings.ylabel)
        ax.set_yticks(settings.yticks)
        
        # Set title if reference stimulus is provided
        if settings.show_ref_in_title and xref is not None:
            ax.set_title(f"Ref = [{np.round(xref[0], 2)}, {np.round(xref[1], 2)}]",
                         fontsize=settings.fontsize)
        
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.8),
                  fontsize= settings.fontsize - 1)
        plt.tight_layout()
        
        if settings.fig_dir and self.save_fig:
            self._save_figure(fig, settings.fig_name)
                 
        plt.show()
        return fig, ax

    def plot_comparison_thres(self, thres_Wishart, slope_mean, slope_CI, xref_unique,
                              settings: PlotThresCompSettings, ax = None,
                              thres_Wishart_CI=None, corr_coef_mean = None,
                              corr_coef_CI = None):
        # Create a new figure and axis if none are provided
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize= settings.fig_size, dpi= settings.dpi)
        else:
            fig = ax.figure
            
        #add the best-fit line
        ax.fill_between(settings.bds, settings.bds * slope_CI[0], 
                        settings.bds * slope_CI[1], 
                        color= settings.line_fit_CI_lc, alpha= settings.line_fit_CI_alpha,
                        label= settings.line_fit_CI_label)
        ax.plot(settings.bds, settings.bds*slope_mean, 
                 color = settings.line_fit_mean_lc, 
                 label = settings.line_fit_mean_label)
        for n in range(len(self.fit_PMF_MOCS)):
            if settings.cmap is None:
                cmap_n = np.array([0,0,0])
            else:
                cmap_n = settings.cmap[n]
            
            if thres_Wishart_CI is not None:
                ax.errorbar(
                    self.fit_PMF_MOCS[n].stim_at_targetPC, 
                    thres_Wishart[n],
                    xerr= self.fit_PMF_MOCS[n].stim_at_targetPC_95btstErr[:, np.newaxis], 
                    yerr= thres_Wishart_CI[n][:, np.newaxis],
                    marker= settings.marker,
                    c = cmap_n,
                    alpha = settings.alpha,
                    ms = settings.ms,#10
                    lw = settings.lw #3
                )
            else:
                ax.errorbar(
                    self.fit_PMF_MOCS[n].stim_at_targetPC, 
                    thres_Wishart[n],
                    xerr= self.fit_PMF_MOCS[n].stim_at_targetPC_95btstErr[:, np.newaxis], 
                    marker= settings.marker,
                    c = cmap_n,
                    alpha = settings.alpha,
                    ms = settings.ms,#10
                    lw = settings.lw #3
                )                
        #add stats to the figure
        if (corr_coef_mean is not None) and (corr_coef_CI is not None):
            ax.text(*settings.corr_text_loc,
                     f"Corr coef = {corr_coef_mean:.2f}; 95% CI:"
                     f" [{corr_coef_CI[0]:.2f}, {corr_coef_CI[1]:.2f}]")
        ax.text(*settings.slope_text_loc,
                     f"Slope = {slope_mean:.2f}; 95% CI: [{slope_CI[0]:.2f}, {slope_CI[1]:.2f}]")
        # Add diagonal line for reference
        ax.set_xlim(settings.bds)
        ax.set_ylim(settings.bds)
        ax.plot(settings.bds, settings.bds, ls='--', c='k', label = 'Identity line')
        ax.set_xticks(np.linspace(*settings.bds, settings.num_ticks))
        ax.set_yticks(np.linspace(*settings.bds, settings.num_ticks))

        # Set axis square and add grid
        ax.set_aspect('equal', adjustable='box')  # Make the axis square
        ax.grid(True, color='grey',linewidth=0.2)   # Add grid lines

        # Add labels, title, etc. (optional)
        ax.set_xlabel(settings.xlabel)
        ax.set_ylabel(settings.ylabel)

        # Add legend outside the plot at the bottom
        ax.legend(
            loc='lower center', 
            bbox_to_anchor=(0.5, -0.45),  # Center below the plot, with some space
            ncol=1,  # Arrange legend entries in 2 columns
            fontsize='medium'
        )

        plt.tight_layout()
        
        if settings.fig_dir and self.save_fig:
            self._save_figure(fig, settings.fig_name)        
        
        plt.show()
        return fig, ax

    
#%%
class MOCSConditionsVisualization(PlottingTools):
    def __init__(self, settings: PlotSettingsBase, save_fig=False, save_format = 'pdf'):
        super().__init__(settings, save_fig, save_format)
        plt.rcParams['font.sans-serif'] = settings.fontstyle
        plt.rcParams['font.size'] = settings.fontsize
        
    def plot_MOCS_conditions(self, ndims, xref_unique, comp_unique, color_thres_data,
                             settings: PlotCondSettings, ax = None):
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
    
        Returns
        -------
        None
        """
        
        if ndims == 2:
            # Create a new figure and axis if none are provided
            if ax is None:
                fig, ax = plt.subplots(figsize=settings.fig_size, dpi= settings.dpi)
            else:
                fig = ax.figure
            
            for idx_slc in range(len(xref_unique)):
                xref = xref_unique[idx_slc]
                comp = comp_unique[idx_slc]
                
                # Plot comparison stimuli
                ax.scatter(comp[:, 0], comp[:, 1], marker=settings.comp_marker, 
                           color=settings.comp_mc, s=settings.comp_ms)
                ax.plot([xref[0], comp[-1, 0]], [xref[1], comp[-1, 1]], 
                        lw= settings.comp_lw, color= settings.comp_lc)
                if settings.easyTrials_highlight:
                    ax.scatter(*comp[-1,:], marker=settings.catch_marker, facecolor='none', 
                               edgecolor= settings.catch_ec, 
                               alpha= settings.catch_alpha, s=settings.catch_ms)
                
                # Color mapping for reference points
                cmap_n = color_thres_data.M_2DWToRGB @ np.append(xref, 1)
                ax.scatter(*xref, color=cmap_n, marker=settings.ref_marker, 
                           lw=settings.ref_lw, s=settings.ref_ms)
        
        else:
            # Create a 3D figure
            # Create a new figure and axis if none are provided
            if ax is None:
                fig = plt.figure(figsize= settings.ig_size, dpi= settings.dpi)
                ax = fig.add_subplot(111, projection='3d')
            else:
                fig = ax.figure
            
            for idx_slc in range(len(xref_unique)):
                xref = xref_unique[idx_slc]
                comp = comp_unique[idx_slc]
                
                # Mapping reference color in RGB cube
                color_map_ref = color_thres_data.W_unit_to_N_unit(xref)
                
                # Plot reference stimuli
                ax.scatter(*xref, c=color_map_ref, marker=settings.ref_marker, 
                           lw=3, s=100)
                
                # Plot comparison stimuli
                ax.scatter(comp[:, 0], comp[:, 1], comp[:, 2], marker=settings.comp_marker, 
                           color= settings.comp_mc, s= settings.comp_ms)
                ax.plot(comp[:, 0], comp[:, 1], comp[:, 2], lw=settings.comp_lw,
                        color=settings.lc)
                if settings.easyTrials_highlight:
                    ax.scatter(*comp[-1, :], marker= settings.catch_marker,
                               facecolor='none', edgecolor=settings.catch_ec, 
                               alpha=settings.catch_alpha, s=settings.catch_ms)
            
        self._update_axes_limits(ax, ndims = ndims) 
        self._update_axes_labels(ax, settings.ticks, settings.ticks,
                                 nsteps = 1, ndims = ndims)  
        self._configure_labels_and_title(ax, title = settings.title, ndims = ndims)
        ax.grid(True, linewidth=0.2)
            
        if ndims == 2:
            ax.set_aspect('equal', adjustable='box')  # Ensure equal scaling
        else:
            ax.set_box_aspect([1, 1, 1])  # Maintain equal aspect ratio
            # Adjust layout to prevent labels from getting cut off
            fig.tight_layout(pad=4.0)
            fig.set_size_inches(4.5, 5)
        
        if settings.fig_dir and self.save_fig:
            self._save_figure(fig, settings.fig_name)
        plt.show()
        return fig, ax
    