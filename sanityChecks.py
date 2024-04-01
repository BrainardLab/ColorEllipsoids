#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 17:20:55 2024

@author: fangfang
"""

import os
import numpy as np
from scipy.stats import logistic
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle

#import functions from the other script
func_path = '/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/Python version/'
os.chdir(func_path)
from Simulate_probCorrectResp import sample_rgb_comp_2DNearContour

#%%
def computeQ_givenEllParam(a,b,theta):
    theta_rad = np.radians(theta)
    #theta has to be in deg
    Q = np.array([[a**2 * np.cos(theta_rad)**2 + b**2 * np.sin(theta_rad)**2, \
                   (a**2 - b**2) * np.sin(theta_rad) * np.cos(theta_rad)],\
                  [(a**2 - b**2) * np.sin(theta_rad) * np.cos(theta_rad),\
                   a**2 * np.sin(theta_rad)**2 + b**2 *np.cos(theta_rad)**2]])
    return Q

def sample_Gaussian(mu, sig, nS):
    samples = np.random.multivariate_normal(mu, sig, size = (1, nS))
    return samples

def sample_NearContour(mu, paramEllipse, nS, jitter):
    samples = sample_rgb_comp_2DNearContour(mu, [0,1], np.nan, nS, paramEllipse,\
                                            jitter)
    return samples

def sim_prob_choosing_x0(xref, x0, x1, sig, nS, sampling_method = 'Gaussian', 
                         flag_smooth = False, bandwidth = 0.1, nbins = 1000,
                         bins_lb = -0.005, bins_ub = 0.002, jitter = 0.1):
    nGrids = xref.shape[1]
    ecdf, diff_sorted = [np.full((nS,nGrids), np.nan) for _ in range(2)]
    pChoosingX0 = np.full((nGrids), np.nan)
    zref, z0, z1 = [np.full((nGrids, nS, 2), np.nan) for _ in range(3)]
    
    if flag_smooth:
        bins_range, logit_pdf, logit_cdf = [np.full((nbins,nGrids), np.nan) \
                                            for _ in range(3)]
    
    for i in range(nGrids):
        if sampling_method == 'Gaussian':
            zref[i,:,:] = sample_Gaussian(xref[:,i], sig, nS) 
            z0[i,:,:] = sample_Gaussian(x0[:,i], sig, nS) 
            z1[i,:,:] = sample_Gaussian(x1[:,i], sig, nS) 
        elif sampling_method == 'NearContour':
            paramE = np.array([0.2, 0.2, sig[0,0], sig[1,1], 0])
            zref_temp = sample_NearContour(0.2+xref[:,i], paramE, nS, jitter) 
            zref[i,:,:] = zref_temp[0:2,:].T 
            z0_temp = sample_NearContour(0.2+x0[:,i], paramE, nS, jitter) 
            z0[i,:,:] = z0_temp[0:2,:].T 
            z1_temp  = sample_NearContour(0.2+x1[:,i], paramE, nS, jitter)     
            z1[i,:,:] = z1_temp[0:2,:].T            
        
        # Compute squared distance of each probe stimulus to reference
        z0_to_zref = np.sum((z0[i,:,:] - zref[i,:,:]) ** 2, axis=-1)
        z1_to_zref = np.sum((z1[i,:,:] - zref[i,:,:]) ** 2, axis=-1)
        
        #signed difference
        diff = z0_to_zref - z1_to_zref
        diff = diff.flatten()
        
        #Sort the data in ascending order
        diff_sorted[:,i] = np.sort(diff)
        
        ecdf[:,i] = np.arange(1, len(diff) + 1) / len(diff)
        
        if flag_smooth:
            bins_range[:,i] = np.linspace(bins_lb, bins_ub, nbins)
            logit_pdf_i = np.zeros((nbins))
            for n in range(nS):
                logit_pdf_i += logistic.pdf(bins_range[:,i], diff_sorted[n,i],\
                                            bandwidth)
            logit_pdf[:,i] = logit_pdf_i
            logit_cdf[:,i] = np.cumsum(logit_pdf[:,i])
            logit_cdf[:,i] = logit_cdf[:,i]/logit_cdf[-1,i]
            min_idx        = np.argmin(np.abs(bins_range[:,i]))
            pChoosingX0[i] = logit_cdf[min_idx,i]/logit_cdf[-1,i]
        else:
            #find the value that's closest to 0
            min_idx = np.argmin(np.abs(diff_sorted[:,i]))
        
            #probability of correct
            pChoosingX0[i] = ecdf[min_idx, i]
    if flag_smooth == False:
        return ecdf, diff_sorted, pChoosingX0, zref, z0, z1
    else:
        return bins_range, logit_pdf, logit_cdf, pChoosingX0, zref, z0, z1

def compute_95CI(nRepeats, p):
    CI_lb_idx = int(np.ceil(nRepeats*0.025))
    CI_ub_idx = int(np.floor(nRepeats*0.975))
    mean_p = np.mean(p, axis = 0)
    p_sort = np.sort(p, axis = 0)
    CI_lb_p = p_sort[CI_lb_idx,:,:,:]
    CI_ub_p = p_sort[CI_ub_idx,:,:,:]
    return mean_p, p_sort, CI_lb_p, CI_ub_p
    
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
    
#%% plotting functions
def plot_ECDF_gt(sorted_x, ecdf, stim, **kwargs):
    pltParams = {
        'saveFig': False,
        'figPath': '',
        'figName': 'SanityCheck_fig1'
        }
    #update default options with any keyword arguments provided
    pltParams.update(kwargs)
    
    plt.rcParams['figure.dpi'] = 250 
    plt.rcParams['font.size'] = 10
    
    nGrids = diff_sorted_gt.shape[-1]
    nSims  = diff_sorted_gt.shape[0]
    fig, ax = plt.subplots(1, 1)
    for i in range(nGrids):
        plt.plot(sorted_x[:,i], ecdf[:,i], linestyle='-',\
                 label = str(np.round(stim[i],4)))
    plt.legend(title = 'Horizontal value of x1')
    plt.xlabel('||z0 - zref|| - ||z1 - zref||')
    plt.ylabel('Empirical Cumulative Distribution Function')
    plt.title('Ground truth (#Sims = '+str(nSims)+', bandwidth = 0)')
    plt.tight_layout()
    plt.grid(True)
    plt.show()
    if pltParams['saveFig'] and pltParams['figPath'] != '':
        if not os.path.exists(pltParams['figPath']):
            os.makedirs(pltParams['figPath'])
        full_path = os.path.join(pltParams['figPath'], pltParams['figName']+'.png')
        fig.savefig(full_path)

def plot_PMF_gt(x, p, **kwargs):
    pltParams = {
        'saveFig': False,
        'figPath': '',
        'figName': 'SanityCheck_fig2'
        }
    #update default options with any keyword arguments provided
    pltParams.update(kwargs)
    
    plt.rcParams['figure.dpi'] = 250 
    plt.rcParams['font.size'] = 10
    
    nGrids = len(p)
    x_lb = x.min()
    x_ub = x.max()
    fig, ax = plt.subplots(1,1)
    #line plot
    plt.plot(x, p, c = [0.5,0.5,0.5]) 
    for i in range(nGrids):
        plt.scatter(x[i], p[i], marker='o', s = 50, label = str(np.round(x[i],4)))
    plt.legend(ncol = 2, title = 'Horizontal value of x1')
    plt.xlim([x_lb, x_ub])
    plt.ylim([0,1])
    plt.xlabel('Horizontal value of x1')
    plt.ylabel('Probability of reporting x0 as more similar to xref')
    plt.title('Ground truth (#Sims = '+str(nSims_gt)+', bandwidth = 0)')
    plt.grid(True)
    if pltParams['saveFig'] and pltParams['figPath'] != '':
        if not os.path.exists(pltParams['figPath']):
            os.makedirs(pltParams['figPath'])
        full_path = os.path.join(pltParams['figPath'], pltParams['figName']+'.png')
        fig.savefig(full_path)

def plot_ECDF_sims(sorted_x, ecdf, nSims_vec, bandwidth_vec, slc_run = 0, **kwargs):
    pltParams = {
        'xlim':[],
        'saveFig': False,
        'figPath': '',
        'figName': 'SanityCheck_fig3'
        }
    #update default options with any keyword arguments provided
    pltParams.update(kwargs)
    
    nRepeats, len_nSims, len_bandwidth, _, nGrids = sorted_x.shape
    x_lb = sorted_x.min()
    x_ub = sorted_x.max()
    
    plt.rcParams['figure.dpi'] = 250
    fig, ax = plt.subplots(len_nSims, len_bandwidth, figsize = (16,9))
    plt.rcParams['font.size'] = 15
    for i in range(len_nSims):
        for j in range(len_bandwidth):
            for k in range(nGrids):
                ax[i][j].plot(sorted_x[slc_run,i,j,:,k], ecdf[slc_run,i,j,:,k],\
                              linestyle='-')        
            ax[i][j].set_yticks(np.linspace(0,1,5))
            ax[i][j].set_xticks(np.sort([x_lb, 0, x_ub]))
            ax[i][j].tick_params(axis='x', labelrotation=45)
            ax[i][j].tick_params(axis='both', labelsize=15)
            if j == 0:
                ax[i][j].text(-0.8, 0.5, '#Sims =' + str(nSims_vec[i]),\
                              transform=ax[i][j].transAxes, rotation=0,\
                              va='bottom',ha='left')
                if i == 1:
                    ax[i][j].set_ylabel('Empirical Cumulative Distribution Function',\
                                        labelpad=10,rotation=90, fontsize = 15)   
            else:
                ax[i][j].set_yticklabels('')
            if i == 0:
                ax[i][j].set_title('bandwidth =' + str(bandwidth_vec[j]))
            if i != len_nSims -1:
                ax[i][j].set_xticklabels('')
            else:
                ax[i][j].set_xlabel('||z0 - zref|| - ||z1 - zref||')
            ax[i][j].grid(True)
    plt.tight_layout()
    if pltParams['saveFig'] and pltParams['figPath'] != '':
        if not os.path.exists(pltParams['figPath']):
            os.makedirs(pltParams['figPath'])
        full_path = os.path.join(pltParams['figPath'], pltParams['figName']+'.png')
        fig.savefig(full_path)

def plot_PMF_sims(mean_p, p_gt, CI_lb_p, CI_ub_p, x1, nSims_vec, bandwidth_vec,
                  **kwargs):
    pltParams = {
        'saveFig': False,
        'figPath': '',
        'figName': 'SanityCheck_fig4'
        }
    #update default options with any keyword arguments provided
    pltParams.update(kwargs)
    
    len_nSims, _, nGrids = mean_pChoosingX0.shape
    len_bandwidth = len(bandwidth_vec)
    
    plt.rcParams['figure.dpi'] = 250 
    fig, ax = plt.subplots(len_nSims,len_bandwidth, figsize = (16,9))
    plt.rcParams['font.size'] = 15
    for i in range(len_nSims):
        for j in range(len_bandwidth):
            CI_lb_ij = CI_lb_p[i,j,:]
            CI_ub_ij = CI_ub_p[i,j,:]
            CI_bds = np.concatenate((CI_lb_ij,CI_ub_ij), axis = 0)
            CI_bds = np.append(CI_bds, CI_lb_ij[0])
            
            x1_temp = x1[0,:]
            x1_repeat = np.concatenate((x1_temp,x1_temp[::-1]), axis = 0)
            x1_repeat = np.append(x1_repeat,x1_temp[0])
            points = np.column_stack((x1_repeat, CI_bds))
            
            polygon = patches.Polygon(points, closed=True, linewidth=1, edgecolor='k',\
                            facecolor='lightblue',alpha = 0.3, label = '95% CI (for 120 \nrepetitions)')
            # Add the polygon patch to the Axes
            ax[i][j].add_patch(polygon)
            ax[i][j].plot(x1[0,:], mean_p[i,j,:], c ='k', label = 'Mean')
            for k in range(nGrids):
                ax[i,j].scatter(x1[0,k], p_gt[k], marker='o', s = 50)
            x1_lb = x1.min()
            x1_ub = x1.max()
            ax[i][j].set_xlim([x1_lb, x1_ub])
            ax[i][j].set_ylim([0,1])
            
            ax[i][j].set_yticks(np.linspace(0,1,5))
            ax[i][j].set_xticks([x1_lb, 0, x1_ub])
            ax[i][j].tick_params(axis='x', labelrotation=45)
            ax[i][j].tick_params(axis='both', labelsize=15)
            if j == 0:
                ax[i][j].text(-0.9, 0.5, '#Sims =' + str(nSims_vec[i]),\
                              transform=ax[i][j].transAxes, rotation=0,\
                              va='bottom',ha='left')
                if i == 1:
                    ax[i][j].set_ylabel('Probability of reporting x0 as more similar to xref',\
                                        labelpad=10,rotation=90, fontsize = 15)   
            else:
                ax[i][j].set_yticklabels('')
            if i == 0:
                ax[i][j].set_title('bandwidth =' + str(bandwidth_vec[j]))
            if i != len_nSims -1:
                ax[i][j].set_xticklabels('')
            else:
                ax[i][j].set_xlabel('Horizontal value of x1', fontsize = 15)
            if i == len_nSims-1 and j == 0:
                ax[i][j].legend()
            ax[i][j].grid(True)
    plt.tight_layout()
    if pltParams['saveFig'] and pltParams['figPath'] != '':
        if not os.path.exists(pltParams['figPath']):
            os.makedirs(pltParams['figPath'])
        full_path = os.path.join(pltParams['figPath'], pltParams['figName']+'.png')
        fig.savefig(full_path)
    
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
                        label = 'CI (smallest \nBrier score +/- '+str(tol)+')')
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
nGrids     = 11
xref       = np.zeros((2,nGrids))
x0         = np.zeros((2,nGrids))
sMethod    = 'Gaussian'
cov_scaler = 5e-4 
sig        = np.eye(2)*cov_scaler
#5e-5: [-0.05, 0.05]
#1e-5: [-0.025, 0.025]
#5e-4: [-0.14, 0.14]
#1e-4: [-0.07, 0.07]
#1e-3: [-0.20, 0.2]
x1_lb      = -0.14
x1_ub      = 0.14
x1_vec     = np.linspace(x1_lb, x1_ub, nGrids)
x1         = np.stack((x1_vec, np.zeros(nGrids)), axis = 0)

nSims_gt = int(1e5)

#ground truth
ecdf_gt, diff_sorted_gt, pChoosingX0_gt, _, _, _ = sim_prob_choosing_x0(xref,\
    x0, x1, sig, nSims_gt, sampling_method = sMethod)

fig_outputDir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                'ELPS_analysis/SanityChecks_FigFiles'
figName_part1 = '_samplingMethod_'+ sMethod+'_covMat_identity' +str(cov_scaler)
                
#visualize the CDF
plot_ECDF_gt(diff_sorted_gt, ecdf_gt, x1_vec,saveFig = True, \
             figPath = fig_outputDir, figName = 'ECDF_gt' + figName_part1)
#visualize the PMF
plot_PMF_gt(x1_vec, pChoosingX0_gt,saveFig = True, figPath = fig_outputDir, \
            figName = 'PMF_gt'+figName_part1)

#%% 
nSims_vec = np.array([10, 50, 100])
bandwidth_vec = 10**np.arange(np.log(cov_scaler)/np.log(10)-3,\
                              np.log(cov_scaler)/np.log(10)+4,0.25) #28 levels
nBins     = 1000
nRepeats  = 120
#binsize: [-0.002, 0.001] for scaler = 1e-5
#binsize: [-0.008, 0.003] for scaler = 5e-4
#binsize: [-0.012, 0.005] for scaler = 1e-4
#binsize: [-0.050, 0.025] for scaler = 5e-3
#binsize: [-0.100, 0.040] for scaler = 1e-3
bins_lb   = -0.050
bins_ub   = 0.025
bins_range, logit_pdf, logit_cdf = [np.full((nRepeats,len(nSims_vec), \
    len(bandwidth_vec), nBins,nGrids),np.nan) for _ in range(3)]
pChoosingX0 = np.full((nRepeats,len(nSims_vec), len(bandwidth_vec), nGrids), np.nan)

for n in range(nRepeats):
    print(n)
    for i in range(len(nSims_vec)):
        for j in range(len(bandwidth_vec)):
            bins_range[n,i,j,:,:], logit_pdf[n,i,j,:,:], logit_cdf[n,i,j,:,:],\
                pChoosingX0[n,i,j,:], _, _, _ = sim_prob_choosing_x0(xref, x0, x1, sig,\
                nSims_vec[i],bandwidth = bandwidth_vec[j],flag_smooth = True,\
                    sampling_method = sMethod, bins_lb = bins_lb, bins_ub =bins_ub) # -0.008, bins_ub = 0.003
            # bins_range[n,i,j,:,:], logit_pdf[n,i,j,:,:], logit_cdf[n,i,j,:,:],\
            #     pChoosingX0[n,i,j,:], _, _, _ = sim_prob_choosing_x0(xref, x0, x1, sig,\
            #     nSims_vec[i],flag_smooth = True, sampling_method = 'NearContour',\
            #     bandwidth = bandwidth_vec[j], bins_lb = -0.008, bins_ub = 0.003,\
            #     jitter = 0.3, nbins = 1000)
            
#%%
bandwidth_slc = 10**np.arange(np.log(cov_scaler)/np.log(10)-2,\
                              np.log(cov_scaler)/np.log(10)+2,1) 
bandwidth_slc_idx = [np.where(bandwidth_vec == element)[0][0] for element in bandwidth_slc]
#plot one instance 
plot_ECDF_sims(bins_range[:,:,bandwidth_slc_idx,:,:], logit_cdf[:,:,bandwidth_slc_idx,:,:], \
               nSims_vec, bandwidth_vec[bandwidth_slc_idx],\
               slc_run = 0, saveFig = True, figPath = fig_outputDir, \
               figName = 'ECDF_sim' + figName_part1)
                   
#compute mean and 95% confidence interval 
mean_pChoosingX0, _, CI_lb_pChoosingX0, CI_ub_pChoosingX0 = \
    compute_95CI(nRepeats, pChoosingX0)

# visualize simulations given difference combinations of #sims and bandwidth
plot_PMF_sims(mean_pChoosingX0[:,bandwidth_slc_idx,:], pChoosingX0_gt,\
              CI_lb_pChoosingX0[:,bandwidth_slc_idx,:],\
              CI_ub_pChoosingX0[:,bandwidth_slc_idx,:],\
              x1, nSims_vec, bandwidth_vec[bandwidth_slc_idx],\
              saveFig = True, figPath = fig_outputDir, figName = \
              'PMF_sim' + figName_part1)
     
#%%
bScore = np.full((len(nSims_vec), len(bandwidth_vec)), np.nan)
optimal_bandwidth, min_bScore = [np.full((len(nSims_vec),1),np.nan) for _ in range(2)]
for i in range(len(nSims_vec)):
    for j in range(len(bandwidth_vec)):
        bScore[i,j] = compute_BrierScore(pChoosingX0_gt, pChoosingX0[:,i,j,:])
    min_idx = np.argmin(bScore[i,:])
    optimal_bandwidth[i] = bandwidth_vec[min_idx]
    min_bScore[i] = bScore[i,min_idx]

plot_BrierScore(bandwidth_vec, nSims_vec, bScore, optimal_bandwidth, min_bScore,\
                title = 'cov = [' + str(cov_scaler) +',0; 0,' + str(cov_scaler) + ']',\
                saveFig = True,figPath = fig_outputDir, figName = \
                    'BrierScore'+figName_part1)
    
#%% save file
#save to CSV
file_name = 'SanityChecks_covScaler' +str(cov_scaler) + figName_part1 + '.pkl'
path_output = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
    'ELPS_analysis/SanityChecks_DataFiles/'
full_path = f"{path_output}{file_name}"

# Write the list of dictionaries to a file using pickle
with open(full_path, 'wb') as f:
    variable_names = ['nGrids', 'xref', 'x0', 'sMethod', 'x1_lb', 'x1_ub',\
                      'x1_vec', 'x1', 'cov_scaler', 'sig', 'nSims_gt',\
                      'ecdf_gt', 'diff_sorted_gt', 'pChoosingX0_gt',\
                      'nSims_vec', 'bandwidth_vec', 'nBins', 'nRepeats',\
                      'bins_range', 'logit_pdf', 'logit_cdf', 'pChoosingX0',\
                    'bScore', 'optimal_bandwidth', 'min_bScore']
    vars_dict = {var_name: globals()[var_name] for var_name in variable_names}
    pickle.dump(vars_dict, f)
    
#%%
cov_scaler_all    = np.array([0.001, 0.0005, 0.0001, 5e-5, 1e-5])
bandwidth_all     = np.full((len(cov_scaler_all), len(bandwidth_vec)), np.nan)
nSims_all         = np.full((len(cov_scaler_all), len(nSims_vec)), np.nan)
bScore_all        = np.full((len(cov_scaler_all), len(nSims_vec), len(bandwidth_vec)),\
                             np.nan)
optimal_bandwidth_all = np.full((len(cov_scaler_all),len(nSims_vec)), np.nan)
min_bScore_all        = np.full((len(cov_scaler_all),len(nSims_vec)), np.nan)
CI_bScore_all = np.full((len(cov_scaler_all), len(nSims_vec), 2), np.nan)

try:
    for s in range(len(cov_scaler_all)):
        figName_part1_s = '_samplingMethod_'+ sMethod+'_covMat_identity' +\
            str(cov_scaler_all[s])
        file_name_s = 'SanityChecks_covScaler' +str(cov_scaler_all[s]) +\
            figName_part1_s + '.pkl'
        full_path = f"{path_output}{file_name_s}"
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
            
    
#%% load files
# file_name = 'Isothreshold_contour_CIELABderived.pkl'
# path_str = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
#         'ELPS_analysis/Simulation_DataFiles/'
# full_path = f"{path_str}{file_name}"
# os.chdir(path_str)

# #Here is what we do if we want to load the data
# with open(full_path, 'rb') as f:
#     # Load the object from the file
#     data_load = pickle.load(f)
# results = data_load[2]
# paramE_2D = results['ellParams']
# longAxis_all, shortAxis_all, angle_all = paramE_2D[:,:,:,2], paramE_2D[:,:,:,3], paramE_2D[:,:,:,4]
# longAxis_all     = longAxis_all.flatten()
# shortAxis_all    = shortAxis_all.flatten()
# angle_all        = angle_all.flatten()
# axisRatio_all    = longAxis_all/shortAxis_all
# axisRatio_maxIdx = np.argmax(axisRatio_all)
# axisRatio_minIdx = np.argmin(axisRatio_all)

# longAxis_thin    = longAxis_all[axisRatio_maxIdx]
# shortAxis_thin   = shortAxis_all[axisRatio_maxIdx]
# angle_thin       = angle_all[axisRatio_maxIdx]
# ellipse_cov      = computeQ_givenEllParam(longAxis_thin, shortAxis_thin, angle_thin)
# eigenvalues, eigenvectors = np.linalg.eig(ellipse_cov)

# # longAxis_short  = longAxis_all[axisRatio_minIdx]
# # shortAxis_short = shortAxis_all[axisRatio_minIdx]
# # angle_short     = angle_all[axisRatio_minIdx]
# # ellipse_cov        = computeQ_givenEllParam(longAxis_short, shortAxis_short, angle_short)

# #plot_ECDF_gt(diff_sorted_gt, ecdf_gt, x1_vec,saveFig = True, \
# #             figPath = fig_outputDir, figName = 'ECDF_gt_samplingMethod_'+\
# #                 sMethod+'_covMat_'+'CIELab_set'+str(axisRatio_maxIdx))