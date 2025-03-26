#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 21:12:36 2024

@author: fangfang
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Tuple, Union
import sys
import os
from datetime import datetime
sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids")
from plotting.wishart_plotting import PlottingTools, PlotSettingsBase

#%%
@dataclass
class PlotWeibullPMFSettings(PlotSettingsBase):
    fig_size: Tuple[float, float] = (2.2, 2.2)
    xticks: List[float] = field(default_factory=lambda: [0, 2, 4, 6])
    xlabel: str = r'Perceptual difference ($\Delta E$)'
    ylabel: str = 'Percent correct'
    fig_name: str = field(default_factory=lambda: f'Weibull_PMF_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
   
@dataclass
class PlotTransformationSettings(PlotSettingsBase):
    fig_size: Tuple[float, float] = (9, 3)
    colorcode_resp: bool = False
    visualize_gt: bool = True
    lim_scaler: float = 1.25
    facecolor: Union[str, np.ndarray, List[float]] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    edgecolor: Union[str, np.ndarray, List[float]] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    facecolor_yes: Union[str, np.ndarray, List[float]] = field(default_factory=lambda: (np.array([107,142,35])/255).tolist())
    facecolor_no: Union[str, np.ndarray, List[float]] = field(default_factory=lambda: (np.array([178,34,34])/255).tolist())
    alpha: float = 1.0
    xlim: List[List[float]] = field(default_factory=lambda: [[] for _ in range(4)])
    ylim: List[List[float]] = field(default_factory=lambda: [[] for _ in range(4)])
    ms: float = 25
    xlabel: str = 'dim 1'
    ylabel: str = 'dim 2'
    fig_name: str = field(default_factory=lambda: f'samples_transformation_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    
@dataclass
class Plot2DSampledCompSettings(PlotSettingsBase):
    fig_size: Tuple[float, float] = (8,8)
    slc_x_grid_ref: List[int] = field(default_factory=list)
    slc_y_grid_ref: List[int] = field(default_factory=list)
    ground_truth: any = None  # adjust type if known (e.g., np.ndarray or Callable)
    xbds: List[float] = field(default_factory=lambda: [-0.025, 0.025])
    ybds: List[float] = field(default_factory=lambda: [-0.025, 0.025])
    xlabel: str = ''
    ylabel: str = ''
    nFinerGrid: int = 50
    lc: Union[str, np.ndarray, List[float]] = field(default_factory=lambda: (np.array([178, 34, 34]) / 255).tolist())
    WishartEllipsesColor: Union[str, np.ndarray, List[float]] = field(default_factory=lambda: (np.array([76, 153, 0]) / 255).tolist())
    m1: str = '.'
    m0: str = '*'
    ms: float = 5
    lw: float = 1
    ls: str = '--'
    alpha: float = 0.8
    mc1: Union[str, np.ndarray, List[float]] = field(default_factory=lambda: (np.array([173, 216, 230]) / 255).tolist())
    mc0: Union[str, np.ndarray, List[float]] = field(default_factory=lambda: (np.array([255, 179, 138]) / 255).tolist())
    fontsize: int = 10
    fig_name: str = field(default_factory=lambda: f'Sampled comparison stimuli_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    
@dataclass
class Plot3DSampledCompSettings(PlotSettingsBase):
    visualize_ellipsoid: bool = True
    visualize_samples: bool = True
    scaled_neg12pos1: bool = False
    nPhi: int = 100
    nTheta: int = 200
    slc_grid_ref_dim1: List[int] = field(default_factory=lambda: list(range(5)))
    slc_grid_ref_dim2: List[int] = field(default_factory=lambda: list(range(5)))
    surf_alpha: float = 0.3
    samples_alpha: float = 0.2
    markerSize_samples: float = 2
    default_viewing_angle: bool = False
    bds: float = 0.025
    fontsize: int = 15
    fig_size: Tuple[float, float] = (8, 8)
    title: str = ''
    fig_name: str = field(default_factory=lambda: f'Sampled_comparison_stimuli_3D_{datetime.now().strftime("%Y%m%d_%H%M%S")}')


#%%
class TrialPlacementVisualization(PlottingTools):
    def __init__(self, sim_trial, settings: PlotSettingsBase, save_fig = False, 
                 save_format = 'pdf'):

        super().__init__(settings, save_fig, save_format)
        self.sim_trial = sim_trial
        plt.rcParams['font.sans-serif'] = settings.fontstyle
        plt.rcParams['font.size'] = settings.fontsize
        
    def plot_WeibullPMF(self, x, settings: PlotWeibullPMFSettings, ax = None):        
        a, b = self.sim_trial.sim['alpha'], self.sim_trial.sim['beta']
        g = self.sim_trial.sim['guessing_rate']
        y = self.sim_trial.WeibullFunc(x, a, b, g)
        # Create a new figure and axes if not provided.
        if ax is None:
            fig, ax =  plt.subplots(1, 1, dpi = settings.dpi,
                                    figsize = settings.fig_size)
        else:
            fig = ax.figure
        
        ax.plot(x, y, color = 'k')
        ax.set_xticks(settings.xticks)
        ax.set_yticks(np.round(np.array([1/3, 2/3, 1]),2))
        ax.set_xlabel(settings.xlabel)
        ax.set_ylabel(settings.ylabel)
        ax.grid(True, alpha=0.5)
        plt.tight_layout()
        # Save the plot with bbox_inches='tight' to ensure labels are not cropped
        if settings.fig_dir and self.save_fig:
            plt.savefig(os.path.join(settings.fig_dir, settings.figname))
        plt.show()
        return fig, ax  

    def plot_transformation(self, ell0, ell1, ell2, ell_final, resp, gt,
                            settings = PlotTransformationSettings, ax = None):
        # Set default font size for all elements
        ell = [ell0, ell1, ell2, ell_final]
        
        # Create a new figure and axes if not provided.
        if ax is None:
            fig, ax =  plt.subplots(1, 4, dpi = settings.dpi, figsize = settings.fig_size)
        else:
            fig = ax.figure
            
        for i in range(4):
            mean_val = np.mean(gt[i], axis = 1)
            if settings.visualize_gt:
                ax[i].plot(gt[i][0], gt[i][1], color = 'k', alpha = 0.35, lw = 2)
                #plot the center
                ax[i].scatter(mean_val[0], mean_val[1],
                              marker = '+', color = 'k', s = 30, lw = 1)
            if settings.colorcode_resp:
                print(i)
                idx_yes = np.where(resp == 1)[0]
                idx_no  = np.where(resp == 0)[0]
                ell_yes = ell[i][:,idx_yes]
                ell_no  = ell[i][:,idx_no]
                ax[i].scatter(ell_yes[0], ell_yes[1], marker = 'o',
                              facecolor = settings.facecolor_yes,
                              edgecolor = settings.edgecolor,
                              s = settings.ms,
                              alpha = settings.alpha)
                ax[i].scatter(ell_no[0], ell_no[1], marker = 'o',
                              facecolor = settings.facecolor_no,
                              edgecolor = settings.edgecolor,
                              s = settings.ms,
                              alpha = settings.alpha)
            else:
                ax[i].scatter(ell[i][0], ell[i][1], 
                              facecolor = settings.facecolor,
                              edgecolor = settings.edgecolor,
                              s = settings.ms,
                              alpha = settings.alpha)
            lim_val = np.max(np.abs(ell[i] -  mean_val[:,None])) *settings.lim_scaler
            if len(settings.xlim[i]) == 0 or len(settings.ylim[i]) == 0:
                ax[i].set_xlim(np.array([-lim_val, lim_val] + mean_val[0]))
                ax[i].set_ylim(np.array([-lim_val, lim_val] + mean_val[1]))
                print(np.array([-lim_val, lim_val] + mean_val[0]))
                print(np.array([-lim_val, lim_val] + mean_val[1]))
            else:
                ax[i].set_xlim(settings.xlim[i])
                ax[i].set_ylim(settings.ylim[i])
            ax[i].set_xlabel(settings.xlabel)
            ax[i].set_ylabel(settings.ylabel)
            ax[i].grid(True, alpha=0.5)
            ax[i].set_aspect('equal')
        plt.tight_layout()
        # Save the plot with bbox_inches='tight' to ensure labels are not cropped
        if settings.fig_dir and self.save_fig:
            plt.savefig(settings.fig_dir + settings.fig_name)
        plt.show()
        return fig, ax       
    
    #%%
    def plot_2D_sampledComp(self, settings: Plot2DSampledCompSettings):
        sim = self.sim_trial.sim
        grid_ref_x = sim['grid_ref']
        grid_ref_y = sim['grid_ref']
        
        nGrid_x = len(grid_ref_x)
        nGrid_y = len(grid_ref_y)
        
        vIdx = sim['varying_RGBplane']
        
        #subplot
        fig, ax = plt.subplots(nGrid_x, nGrid_y, figsize = settings.fig_size, dpi = settings.dpi)
        for i in range(nGrid_x):
            for j in range(nGrid_y):
                # Access the subplot from bottom to top
                ax_idx = ax[nGrid_x - 1 - i, j]  # Reverse row order

                x_axis = np.linspace(*settings.xbds, settings.nFinerGrid) + grid_ref_x[j]
                y_axis = np.linspace(*settings.ybds, settings.nFinerGrid) + grid_ref_y[i]    
                
                #plot the ground truth
                if settings.ground_truth is not None:
                    ax_idx.plot(settings.ground_truth[i,j,0],
                                settings.ground_truth[i,j,1],
                                color= settings.lc,
                                linestyle = settings.ls, 
                                linewidth = settings.lw)
                
                #find indices that correspond to a response of 1 / 0
                idx_1 = np.where(sim['resp_binary'][i,j] == 1)
                idx_0 = np.where(sim['resp_binary'][i,j] == 0)
                ax_idx.scatter(sim['rgb_comp'][i, j, vIdx[0], idx_1],
                               sim['rgb_comp'][i, j, vIdx[1], idx_1],
                               s = settings.ms, 
                               marker= settings.m1,
                               c= settings.mc1,
                               alpha= settings.alpha)
                    
                ax_idx.scatter(sim['rgb_comp'][i, j, vIdx[0], idx_0], 
                               sim['rgb_comp'][i, j, vIdx[1], idx_0], 
                               s = settings.ms, 
                               marker= settings.m0, 
                               c= settings.mc0,
                               alpha= settings.alpha)
                
                ax_idx.set_xlim([x_axis[0], x_axis[-1]])
                ax_idx.set_ylim([y_axis[0], y_axis[-1]])
                if i == 0 and j == nGrid_y//2: ax_idx.set_xlabel(settings.xlabel)
                if i == nGrid_x//2 and j == 0: ax_idx.set_ylabel(settings.ylabel)
                
                if j == 0: ax_idx.set_yticks([grid_ref_y[i]])
                else: ax_idx.set_yticks([])
                
                if i == 0: ax_idx.set_xticks([grid_ref_x[j]])
                else: ax_idx.set_xticks([])
            
        plt.subplots_adjust(wspace = 0, hspace = 0)
        plt.tight_layout()
        plt.show()
        if self.save_fig and settings.fig_dir:
            full_path2 = f"{settings.fig_dir}/{settings.fig_name}.pdf"
            fig.savefig(full_path2) 
        return fig, ax
        
    @staticmethod
    def plot_3D_sampledComp(ref_points, fitEllipsoid_unscaled, sampledComp,
                            fixedPlane, fixedPlaneVal, settings: Plot3DSampledCompSettings,
                            save_fig = False):
        
        #Determine the indices of the reference points based on the fixed 
        # plane specified ('R', 'G', or 'B' for different color channels)
        if fixedPlane =='R':
            idx_x = np.where(np.abs(ref_points - fixedPlaneVal)< 1e-6)[0][0]
        elif fixedPlane == 'G':
            idx_y = np.where(np.abs(ref_points - fixedPlaneVal)< 1e-6)[0][0]
        elif fixedPlane == 'B':
            idx_z = np.where(np.abs(ref_points - fixedPlaneVal)< 1e-6)[0][0]
        else:
            return
        
        nGridPts_dim1 = len(settings.slc_grid_ref_dim1)
        nGridPts_dim2 = len(settings.slc_grid_ref_dim2)
        ref_points_idx = np.array(list(range(len(ref_points))))
        
        fig, axs = plt.subplots(nGridPts_dim2, nGridPts_dim1, subplot_kw={'projection': '3d'}, \
                                figsize=settings.fig_size)
        for j in range(nGridPts_dim2-1,-1,-1):
            jj = ref_points_idx[settings.slc_grid_ref_dim2[nGridPts_dim2-j-1]]
            for i in range(nGridPts_dim1):
                ii = ref_points_idx[settings.slc_grid_ref_dim1[i]]
                
                ax = axs[j, i]
                base_shape = (settings.nPhi, settings.nTheta)
                if fixedPlane == 'R':
                    slc_ref = np.array([fixedPlaneVal, ref_points[ii], ref_points[jj]])
                    slc_gt = fitEllipsoid_unscaled[idx_x, ii,jj,:,:]
                    slc_rgb_comp = sampledComp[idx_x, ii,jj,:,:]
                elif fixedPlane == 'G':
                    slc_ref = np.array([ref_points[ii], fixedPlaneVal, ref_points[jj]])
                    slc_gt = fitEllipsoid_unscaled[ii,idx_y,jj,:,:]
                    slc_rgb_comp = sampledComp[ii,idx_y,jj,:,:]
                elif fixedPlane == 'B':
                    slc_ref = np.array([ref_points[ii], ref_points[jj], fixedPlaneVal])
                    slc_gt = fitEllipsoid_unscaled[ii,jj,idx_z,:,:]
                    slc_rgb_comp = sampledComp[ii,jj,idx_z,:,:]
                slc_gt_x = np.reshape(slc_gt[0,:], base_shape)
                slc_gt_y = np.reshape(slc_gt[1,:], base_shape)
                slc_gt_z = np.reshape(slc_gt[2,:], base_shape)
                       
                #subplot
                if settings.visualize_ellipsoid:
                    if settings.scaled_neg12pos1: color_v = (slc_ref+1)/2
                    else: color_v = slc_ref
                    ax.plot_surface(slc_gt_x, slc_gt_y, slc_gt_z, \
                        color=color_v, edgecolor='none', alpha=0.5)
                    
                if settings.visualize_samples:
                    ax.scatter(slc_rgb_comp[0,:], slc_rgb_comp[1,:], slc_rgb_comp[2,:],\
                               s=settings.markerSize_samples, c= [0,0,0],
                               alpha=settings.samples_alpha)
                        
                ax.set_xlim(slc_ref[0]+np.array(settings.bds * np.array([-1,1]))); 
                ax.set_ylim(slc_ref[1]+np.array(settings.bds * np.array([-1,1])));  
                ax.set_zlim(slc_ref[2]+np.array(settings.bds * np.array([-1,1])));  
                ax.set_xlabel('');ax.set_ylabel('');ax.set_zlabel('');
                #set tick marks
                if fixedPlane == 'R':
                    ax.set_xticks([]); 
                else:
                    ax.set_xticks(np.round(slc_ref[0]+\
                        np.array(np.ceil(settings.bds*100)/100*\
                        np.array([-1,0,1])),2))
                    
                if fixedPlane == 'G':
                    ax.set_yticks([]); 
                else:
                    ax.set_yticks(np.round(slc_ref[1]+\
                        np.array(np.ceil(settings.bds*100)/100*\
                        np.array([-1,0,1])),2))
                    
                if fixedPlane == 'B':
                    ax.set_zticks([]);
                else:
                    ax.set_zticks(np.round(slc_ref[2]+\
                        np.array(np.ceil(settings.bds*100)/100*\
                        np.array([-1,0,1])),2))
                # Adjust viewing angle for better visualization
                if not settings.default_viewing_angle:
                    if fixedPlane == 'R': ax.view_init(0,0)
                    elif fixedPlane == 'G': ax.view_init(0,-90)
                    elif fixedPlane == 'B': ax.view_init(90,-90)
                else:
                    ax.view_init(30,-37.5)
                ax.grid(True)
                ax.set_aspect('equal')
        fig.suptitle(settings.title)
        plt.tight_layout()
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05,\
                            wspace=-0.05, hspace=-0.05)
        plt.show()
        if save_fig and settings.fig_dir:
            full_path2 = f"{settings.fig_dir}/{settings.fig_name}"
            fig.savefig(full_path2)            