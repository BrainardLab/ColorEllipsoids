#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 11:40:22 2024

@author: fangfang
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle

#%%
def computeQ_givenEllParam(a,b,theta):
    theta_rad = np.radians(theta)
    Q = np.array([[a**2 * np.cos(theta_rad)**2 + b**2 * np.sin(theta_rad)**2, \
                   (a**2 - b**2) * np.sin(theta_rad) * np.cos(theta_rad)],\
                  [(a**2 - b**2) * np.sin(theta_rad) * np.cos(theta_rad),\
                   a**2 * np.sin(theta_rad)**2 + b**2 *np.cos(theta_rad)**2]])
    return Q

def compute_BrierScore(p_gt, p_sims):
    nRepeats, nGrids = p_sims.shape
    squared_diff = (p_sims - p_gt)**2
    BrierScore = np.sum(squared_diff)/(nRepeats * nGrids)
    return BrierScore
   
def compute_optimalBandwidth_CI(bdw, bScore, opt_bScore, opt_bdw, tolerance = 0.005):
    idx = np.argmin(abs(bdw - opt_bdw))
    index_lb_closest = np.argmin(np.abs(bScore[0:idx+1] - opt_bScore - tolerance))
    index_ub_closest = np.argmin(np.abs(bScore[idx:-1] - opt_bScore + tolerance))
    return np.array([bdw[index_lb_closest], bdw[index_ub_closest+idx]])
     
def plot_optBandwidth(opt_bdw, CI_bdw, eigv, nSims_vec, tol, **kwargs):
    pltParams = {
        'saveFig': False,
        'figPath': '',
        'figName': 'SanityCheck_fig6'
        }
    #update default options with any keyword arguments provided
    pltParams.update(kwargs)
    
    fig, ax = plt.subplots(1,len(nSims_vec), figsize = [9,6])
    plt.rcParams['figure.dpi'] = 250 
    cmap = np.array([[128,181,140],[42,130,139],[4,54,94]])/255
    for i in range(len(nSims_vec)):
        eigv_log = np.log(eigv)/np.log(10)
        ax[i].plot(eigv_log, eigv_log, c='k',linestyle='--')
        ax[i].plot(eigv_log, np.log(opt_bdw[i,:])/np.log(10), c = cmap[i,:],\
                   linewidth=3, label = '#Sims = ' + str(int(nSims_vec[i])))
        
        CI_bdw_log = np.log(CI_bdw)/np.log(10)
        CI_lb_ij = CI_bdw_log[i,:,0]
        CI_ub_ij = CI_bdw_log[i,:,1]
        CI_bds = np.concatenate((CI_lb_ij,CI_ub_ij[::-1]), axis = 0)
        CI_bds = np.append(CI_bds, CI_lb_ij[0])
        eigv_repeat = np.concatenate((eigv_log,eigv_log[::-1]), axis = 0)
        eigv_repeat = np.append(eigv_repeat,eigv_repeat[0])
        points = np.column_stack((eigv_repeat, CI_bds))
        
        polygon = patches.Polygon(points, closed=True, linewidth=1, edgecolor=cmap[i,:],\
                        facecolor=cmap[i,:],alpha = 0.3, \
                        label = 'CI (smallest \nBrier score + '+str(tol)+')')
        # Add the polygon patch to the Axes
        ax[i].add_patch(polygon)
        if i == 1: ax[i].set_xlabel('Log eigenvalue of the cov matrix')
        if i == 0: ax[i].set_ylabel('Log optimal bandwidth')
        else: ax[i].set_yticklabels('')
        ax[i].set_ylim([-8,-2])
        ax[i].legend(fontsize = 10)
        ax[i].grid(True)
    plt.tight_layout()
    if pltParams['saveFig'] and pltParams['figPath'] != '':
        if not os.path.exists(pltParams['figPath']):
            os.makedirs(pltParams['figPath'])
        full_path = os.path.join(pltParams['figPath'], pltParams['figName']+'.png')
        fig.savefig(full_path)        
        
#%%
def plot_BrierScore(bdw_vec, nSims_vec, bScore, optimal_bdw, min_bScore, **kwargs):
    pltParams = {
        'xlim': [],
        'ylim': [],
        'figSize': [8,3.5],
        'flag_legend': True,
        'title': '',
        'saveFig': False,
        'figPath': '',
        'figName': 'SanityCheck_fig5'
        }
    #update default options with any keyword arguments provided
    pltParams.update(kwargs)

    fig, ax = plt.subplots(1, 1, figsize = pltParams['figSize'])
    plt.rcParams['figure.dpi'] = 250 
    cmap = np.array([[128,181,140],[42,130,139],[4,54,94]])/255
    for i in range(len(nSims_vec)):
        plt.plot(np.log(bdw_vec)/np.log(10), bScore[i,:], label = str(int(nSims_vec[i])),\
                 c = cmap[i,:], linewidth= 2)
        plt.scatter(np.log(optimal_bdw[i])/np.log(10), min_bScore[i], \
                    marker ='*', s = 200, c = cmap[i,:])
    if pltParams['flag_legend']: plt.legend(title = '#Sims')
    plt.xlabel('log(bandwidth)')
    plt.ylabel('Brier score\n' + r'$\frac{1}{R*N} \sum_r^{r=120,} \sum_n^{n=11} (f_{r,n} - p_n)^2$')
    if pltParams['title'] != '': plt.title(pltParams['title'])
    plt.grid(True)
    if pltParams['xlim'] != []: plt.xlim(pltParams['xlim'])
    if pltParams['ylim'] != []: plt.ylim(pltParams['ylim'])
    plt.tight_layout()
    plt.show()
    if pltParams['saveFig'] and pltParams['figPath'] != '':
        if not os.path.exists(pltParams['figPath']):
            os.makedirs(pltParams['figPath'])
        full_path = os.path.join(pltParams['figPath'], pltParams['figName']+'.png')
        fig.savefig(full_path)    
        
#%%
cov_scaler_all     = np.array([0.001, 0.0005, 0.0001, 5e-5, 1e-5])
len_cov_scaler_all = len(cov_scaler_all)
len_bandwidth_vec  = 28
nSims_vec          = np.array([10, 50, 100])
len_nSims_vec      = len(nSims_vec)

nBins     = 1000
nRepeats  = 120
bandwidth_all     = np.full((len_cov_scaler_all, len_bandwidth_vec), np.nan)
nSims_all         = np.full((len_cov_scaler_all, len_nSims_vec), np.nan)
bScore_all        = np.full((len_cov_scaler_all, len_nSims_vec, len_bandwidth_vec),\
                             np.nan)
optimal_bandwidth_all = np.full((len_cov_scaler_all,len_nSims_vec), np.nan)
min_bScore_all        = np.full((len_cov_scaler_all,len_nSims_vec), np.nan)
CI_bScore_all = np.full((len_cov_scaler_all, len_nSims_vec, 2), np.nan)

fig_outputDir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                'ELPS_analysis/SanityChecks_FigFiles'
try:
    for s in range(len_cov_scaler_all):
        figName_part1_s = '_covMat_identity' + str(cov_scaler_all[s])
        file_name_s = 'SanityChecks_covScaler' +str(cov_scaler_all[s]) +\
            figName_part1_s + '.pkl'
        import_path = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
            'ELPS_analysis/SanityChecks_DataFiles/'
        full_path = f"{import_path}{file_name_s}"
        with open(full_path, 'rb') as f:
            # Load the object from the file
            data_load = pickle.load(f)
            bandwidth_all[s,:] = data_load['bandwidth_vec'].ravel()
            nSims_all[s,:] = data_load['nSims_vec']
            bScore_all[s,:,:] = data_load['bScore']
            optimal_bandwidth_all[s,:] = data_load['optimal_bandwidth'].ravel()
            min_bScore_all[s,:] = data_load['min_bScore'].ravel()
            
        plot_BrierScore(bandwidth_all[s,:], nSims_all[s,:], bScore_all[s,:,:],\
                        optimal_bandwidth_all[s,:], min_bScore_all[s,:], xlim = [-8,1],ylim=[0,0.1],\
                        title = 'cov = [' + str(cov_scaler_all[s]) +',0; 0,' + \
                        str(cov_scaler_all[s]) + ']',  flag_legend = True,\
                        saveFig = True, figPath = fig_outputDir, figName = \
                            'BrierScore'+figName_part1_s)
        for n in range(len(nSims_vec)):
            CI_bScore_all[s,n,:] = compute_optimalBandwidth_CI(bandwidth_all[s,:],\
                bScore_all[s,n,:], min_bScore_all[s,n], optimal_bandwidth_all[s,n])
    plot_optBandwidth(optimal_bandwidth_all.T, np.transpose(CI_bScore_all,(1, 0, 2)),\
                      cov_scaler_all, nSims_vec, tol = 0.005, saveFig = True, \
                          figPath = fig_outputDir, figName = 'OptBandwidth'+figName_part1_s)
except:
    print('No such file!')
    
#%%
file_name = 'Isothreshold_contour_CIELABderived.pkl'
path_str = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
        'ELPS_analysis/Simulation_DataFiles/'
full_path = f"{path_str}{file_name}"
os.chdir(path_str)

#Here is what we do if we want to load the data
with open(full_path, 'rb') as f:
    # Load the object from the file
    data_load = pickle.load(f)
results = data_load[2]
paramE_2D = results['ellParams']
longAxis_all, shortAxis_all, angle_all = paramE_2D[:,:,:,2], paramE_2D[:,:,:,3], paramE_2D[:,:,:,4]

ellipses_cov = np.full((3,5,5,2,2), np.nan)
cov_eigval = np.full((3,5,5,2), np.nan)
for s in range(3):
    for i in range(5):
        for j in range(5):
            ellipses_cov[s,i,j,:,:] = computeQ_givenEllParam(longAxis_all[s,i,j],\
                                                  shortAxis_all[s,i,j],\
                                                  angle_all[s,i,j])
            cov_eigval[s,i,j,:], _ = np.linalg.eig(ellipses_cov[s,i,j,:,:])


longAxis_all     = longAxis_all.flatten()
shortAxis_all    = shortAxis_all.flatten()
angle_all        = angle_all.flatten()
axisRatio_all    = longAxis_all/shortAxis_all
axisRatio_maxIdx = np.argmax(axisRatio_all)
axisRatio_minIdx = np.argmin(axisRatio_all)

longAxis_thin    = longAxis_all[axisRatio_maxIdx]
shortAxis_thin   = shortAxis_all[axisRatio_maxIdx]
angle_thin       = angle_all[axisRatio_maxIdx]
ellipse_cov      = computeQ_givenEllParam(longAxis_thin, shortAxis_thin, angle_thin)
eigenvalues, eigenvectors = np.linalg.eig(ellipse_cov)

# longAxis_short  = longAxis_all[axisRatio_minIdx]
# shortAxis_short = shortAxis_all[axisRatio_minIdx]
# angle_short     = angle_all[axisRatio_minIdx]
# ellipse_cov        = computeQ_givenEllParam(longAxis_short, shortAxis_short, angle_short)

#plot_ECDF_gt(diff_sorted_gt, ecdf_gt, x1_vec,saveFig = True, \
#             figPath = fig_outputDir, figName = 'ECDF_gt_samplingMethod_'+\
#                 sMethod+'_covMat_'+'CIELab_set'+str(axisRatio_maxIdx))
