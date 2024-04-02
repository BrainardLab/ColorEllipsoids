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

#%%
def sim_prob_choosing_x0(xref, x0, x1, sig, nS, 
                         flag_smooth = False, bandwidth = 0.1, nbins = 1000,
                         bins_lb = -0.005, bins_ub = 0.002):
    """
    Define a function to simulate the probability of choosing option x0 as more
    similar to a reference stimulus xref.
    The simulation accounts for variability in observations using a Gaussian 
    distribution model with specified parameters.

    Parameters:
    - xref (array; 2 x N): reference stimuli (1st col: x-val; 2nd col: yval)
        N is the number of selected stimuli
    - x0 (array; 2 x N): exactly the same as xref
    - x1 (array; 2 x N): comparison stimuli that are different from xref 
    - sig (array; 2 x 2): covariance matrix
    - nS (int): number of samples drawn from every triplet of xref, x0 and x1
    - flag_smooth (boolean): whether we want to apply a logistic density 
        function to smooth the cumulative distribution function
    - bandwidth (float): the width of the smoothing kernel
    - nbins (int): number of bins within a range
    - bins_lb (float): the lower bounds of the range
    - bins_ub (float): the upper bounds of the range

    Returns:
    - ecdf (array; nS x N): empirical cumulative distribution function
    - diff_sorted (array; nS x N): sorted ||z0 - zref|| - ||z1 - zref||
    - pChoosingX0 (array; N,): the probability of choosing X0 as more similar 
        to xref 
    - logit_pdf (array; nbins x N): ||z0 - zref|| - ||z1 - zref|| convoled 
        with a logistic density function
    - zref (array; N x nS x 2): randomly generated measurements given xref
    - z0 (array; N x nS x 2): randomly generated measurements given x0
    - z1 (array; N x nS x 2): randomly generated measurements given x1
    """
    
    #number of testing reference stimuli
    nStim = xref.shape[1]
    #initialization
    ecdf, diff_sorted = [np.full((nS, nStim), np.nan) for _ in range(2)]
    pChoosingX0 = np.full((nStim), np.nan)
    zref, z0, z1 = [np.full((nStim, nS, 2), np.nan) for _ in range(3)]
    
    if flag_smooth:
        bins_range, logit_pdf, logit_cdf = [np.full((nbins,nStim), np.nan) \
                                            for _ in range(3)]
    
    # Iterate over each reference stimulus to simulate data and calculate 
    #probabilities.
    for i in range(nStim):
        # Sample data for xref, x0, and x1 using a multivariate normal 
        #distribution based on provided parameters.
        zref[i,:,:] = np.random.multivariate_normal(xref[:,i], sig, size = (1, nS))    
        z0[i,:,:] = np.random.multivariate_normal(x0[:,i], sig, size = (1, nS))
        z1[i,:,:] = np.random.multivariate_normal(x1[:,i], sig, size = (1, nS))
        
        # Compute squared distance of each probe stimulus to reference
        z0_to_zref = np.sum((z0[i,:,:] - zref[i,:,:]) ** 2, axis=-1)
        z1_to_zref = np.sum((z1[i,:,:] - zref[i,:,:]) ** 2, axis=-1)
        
        #signed difference
        diff = z0_to_zref - z1_to_zref
        diff = diff.flatten()
        
        #Sort the data in ascending order
        diff_sorted[:,i] = np.sort(diff)
        
        #empirical cumulative distribution function 
        ecdf[:,i] = np.arange(1, len(diff) + 1) / len(diff)
        
        # If smoothing is requested, then we convolve each ||z0-zref|| - ||z1-zref||
        # with a logistic probability density, and sum them together
        if flag_smooth:
            # Define the range of bins for the logistic probability density 
            #function (PDF) smoothing, based on user-defined lower and upper 
            #bounds and number of bins.
            bins_range = np.linspace(bins_lb, bins_ub, nbins)
            # Initialize an array to hold the logistic PDF values for the 
            #current grid point, starting with all zeros.
            logit_pdf_i = np.zeros((nbins))
            
            # For each sample, add the logistic PDF values to logit_pdf_i. The 
            #logistic PDF is computed at each bin center, with a mean equal to 
            #the sorted difference and a specified bandwidth that controls the 
            #spread.
            for n in range(nS):
                logit_pdf_i += logistic.pdf(bins_range, diff_sorted[n,i],\
                                            bandwidth)
            # Store the sum of logistic PDFs for the current grid point in the 
            #overall logit_pdf matrix.
            logit_pdf[:,i] = logit_pdf_i
            # Compute the logistic cumulative distribution function (CDF) by 
            #cumulatively summing the logistic PDF values.
            logit_cdf[:,i] = np.cumsum(logit_pdf[:,i])
            # Normalize the logistic CDF so that its final value equals 1, 
            #ensuring it represents a proper CDF.
            logit_cdf[:,i] = logit_cdf[:,i]/logit_cdf[-1,i]
            # Identify the index of the bin that is closest to zero, which is 
            #used to find the probability of choosing x0.
            min_idx        = np.argmin(np.abs(bins_range))
            pChoosingX0[i] = logit_cdf[min_idx,i]
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
    plt.xlabel(r'$||z_0 - z_{ref}||^2 - ||z_1 - z_{ref}||^2$')
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
    
    nRepeats, len_nSims, len_bandwidth, _, nGrids = ecdf.shape
    x_lb = sorted_x.min()
    x_ub = sorted_x.max()
    
    plt.rcParams['figure.dpi'] = 250
    fig, ax = plt.subplots(len_nSims, len_bandwidth, figsize = (16,9))
    plt.rcParams['font.size'] = 15
    for i in range(len_nSims):
        for j in range(len_bandwidth):
            for k in range(nGrids):
                ax[i][j].plot(sorted_x[i,j,:], ecdf[slc_run,i,j,:,k],\
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
                ax[i][j].set_xlabel(r'$||z_0 - z_{ref}||^2 - ||z_1 - z_{ref}||^2$')
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
                            facecolor='lightblue',alpha = 0.3, \
                            label = '95% CI (for 120 \nrepetitions)')
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
                    ax[i][j].set_ylabel('Probability of reporting x0 as more'+\
                                        ' similar to xref',\
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
 
def plot_z1(z1, x1, x1_idx_slc, covMat, **kwargs):
    pltParams = {
        'saveFig': False,
        'figPath': '',
        'figName': 'SanityCheck_fig4b'
        }
    #update default options with any keyword arguments provided
    pltParams.update(kwargs)
    
    nStim, nS, dim = z1.shape
    x1_slc = x1[:,x1_idx_slc]
    
    #unit circle
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_points = np.array([np.cos(theta), np.sin(theta)])
    
    # Perform eigenvalue decomposition of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(covMat)
    
    # Scale and rotate the points according to the covariance matrix
    ellipse_points = (eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ circle_points).T
    
    plt.rcParams['figure.dpi'] = 250 
    plt.rcParams['font.size'] = 10    
    fig, ax = plt.subplots(1,1, figsize = (8,3))
    
    # Get the default colormap name
    default_cmap = plt.get_cmap('plasma')
    
    values = np.linspace(0, 0.8, nStim)

    # Get the array of RGBA colors from the colormap
    colors_array = default_cmap(values)
    
    for i in range(nStim):
        z1_i = z1[i,:,:]
        if i in x1_idx_slc:
            plt.scatter(z1_i[:,0], z1_i[:,1],s = 10, alpha = 0.3, c = colors_array[i,:])
            plt.plot(ellipse_points[:,0] + x1[0,i], ellipse_points[:,1] + \
                     x1[1,i],c = colors_array[i,:])
        else:
            plt.plot(ellipse_points[:,0] + x1[0,i], ellipse_points[:,1] + \
                     x1[1,i],c = colors_array[i,:], alpha = 0.2)
        plt.scatter(x1[0,i], x1[1,i], marker = '+', c = colors_array[i,:])
    plt.axis('equal')
    plt.xticks(x1_slc[0,:])
    plt.yticks([0])
    plt.xlim([x1[0,0]-3*np.sqrt(covMat[0][0]), x1[0,-1]+3*np.sqrt(covMat[0][0])])
    plt.ylim([-3*np.sqrt(covMat[0][0]), 3*np.sqrt(covMat[0][0])])
    plt.xlabel('Horizontal value of x1')
    plt.ylabel('Vertical value of x1')
    plt.title('Eigenvalue = '+str(covMat[0,0]))
    plt.tight_layout()
    plt.show()
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
    
#%%
cov_scaler = 1e-5
nGrids     = 11
xref       = np.zeros((2,nGrids))
x0         = np.zeros((2,nGrids))
sig        = np.eye(2)*cov_scaler

#1e-5: [-0.019, 0.019]
#5e-5: [-0.042, 0.042]
#1e-4: [-0.06, 0.06]
#5e-4: [-0.134, 0.134]
#1e-3: [-0.19, 0.19]
nSDs       = 3.5
x1_ub      = np.round(np.sqrt(cov_scaler*4)*nSDs,3)
x1_lb      = -x1_ub
x1_vec     = np.linspace(x1_lb, x1_ub, nGrids)
x1         = np.stack((x1_vec, np.zeros(nGrids)), axis = 0)

nSims_gt = int(1e5)

#ground truth
ecdf_gt, diff_sorted_gt, pChoosingX0_gt, _, _, _ = sim_prob_choosing_x0(xref,\
    x0, x1, sig, nSims_gt)

figName_part1 = '_covMat_identity' +str(cov_scaler)
fig_outputDir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                'ELPS_analysis/SanityChecks_FigFiles'
flag_saveFig = True
                
#visualize the CDF
plot_ECDF_gt(diff_sorted_gt, ecdf_gt, x1_vec,saveFig = flag_saveFig, \
             figPath = fig_outputDir, figName = 'ECDF_gt' + figName_part1)
#visualize the PMF
plot_PMF_gt(x1_vec, pChoosingX0_gt,saveFig = flag_saveFig, figPath = fig_outputDir, \
            figName = 'PMF_gt'+figName_part1)

#%% 
nSims_vec = np.array([10, 50, 100])
bandwidth_vec = 10**np.arange(np.log(cov_scaler)/np.log(10)-3,\
                              np.log(cov_scaler)/np.log(10)+4,0.25) #28 levels
nBins     = 1000
nRepeats  = 10
#1e-5: [-0.002, 0.001] 
#5e-4: [-0.008, 0.003] 
#1e-4: [-0.012, 0.005] 
#5e-3: [-0.050, 0.025] 
#1e-3: [-0.100, 0.040] 
bins_lb   = -np.round(x1_ub**2 * nSDs,3)
bins_ub   = np.round(-bins_lb/3,3)
bins_range = np.full((len(nSims_vec), len(bandwidth_vec), nBins),np.nan)
logit_pdf, logit_cdf = [np.full((nRepeats,len(nSims_vec), len(bandwidth_vec),\
                                 nBins,nGrids),np.nan) for _ in range(2)]
pChoosingX0 = np.full((nRepeats,len(nSims_vec), len(bandwidth_vec), nGrids), np.nan)

for n in range(nRepeats):
    print(n)
    for i in range(len(nSims_vec)):
        for j in range(len(bandwidth_vec)):
            bins_range[i,j,:], logit_pdf[n,i,j,:,:], logit_cdf[n,i,j,:,:],\
                pChoosingX0[n,i,j,:], _, _, _ = sim_prob_choosing_x0(\
                xref, x0, x1, sig, nSims_vec[i],bandwidth = bandwidth_vec[j],\
                flag_smooth = True, bins_lb = bins_lb, bins_ub =bins_ub) 
                    
            if n == 0 and i == (len(nSims_vec)-1) and j == 0:
                _, _, _, _, zref, z0, z1 = sim_prob_choosing_x0(xref, x0, x1, sig,\
                    nSims_vec[i],bandwidth = bandwidth_vec[j],flag_smooth = True,\
                    bins_lb = bins_lb, bins_ub =bins_ub)         

idx_slc_x1 = np.array([0,3,5,7,10])
plot_z1(z1, x1, idx_slc_x1, sig, saveFig = flag_saveFig, figPath = fig_outputDir,\
        figName = 'SampledZ1_sim' + figName_part1)
            
#%%
bandwidth_slc = 10**np.arange(np.log(cov_scaler)/np.log(10)-2,\
                              np.log(cov_scaler)/np.log(10)+2,1) 
bandwidth_slc_idx = [np.where(bandwidth_vec == element)[0][0] for element in bandwidth_slc]
#plot one instance 
plot_ECDF_sims(bins_range[:,bandwidth_slc_idx,:], logit_cdf[:,:,bandwidth_slc_idx,:,:], \
               nSims_vec, bandwidth_vec[bandwidth_slc_idx],\
               slc_run = 0, saveFig = flag_saveFig, figPath = fig_outputDir, \
               figName = 'ECDF_sim' + figName_part1)
                   
#compute mean and 95% confidence interval 
mean_pChoosingX0, _, CI_lb_pChoosingX0, CI_ub_pChoosingX0 = \
    compute_95CI(nRepeats, pChoosingX0)

# visualize simulations given difference combinations of #sims and bandwidth
plot_PMF_sims(mean_pChoosingX0[:,bandwidth_slc_idx,:], pChoosingX0_gt,\
              CI_lb_pChoosingX0[:,bandwidth_slc_idx,:],\
              CI_ub_pChoosingX0[:,bandwidth_slc_idx,:],\
              x1, nSims_vec, bandwidth_vec[bandwidth_slc_idx],\
              saveFig = flag_saveFig, figPath = fig_outputDir, figName = \
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
                saveFig = flag_saveFig,figPath = fig_outputDir, figName = \
                    'BrierScore'+figName_part1)
    
#%% save file
#save to .pkl
file_name = 'SanityChecks_covScaler' +str(cov_scaler) + figName_part1 + '.pkl'
path_output = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
    'ELPS_analysis/SanityChecks_DataFiles/'
full_path = f"{path_output}{file_name}"

# Write the list of dictionaries to a file using pickle
with open(full_path, 'wb') as f:
    variable_names = ['nGrids', 'xref', 'x0', 'x1_lb', 'x1_ub',\
                      'x1_vec', 'x1', 'cov_scaler', 'sig', 'nSims_gt',\
                      'ecdf_gt', 'diff_sorted_gt', 'pChoosingX0_gt',\
                      'nSims_vec', 'bandwidth_vec', 'nBins', 'nRepeats',\
                      'bins_range', 'logit_pdf', 'logit_cdf', 'pChoosingX0',\
                    'bScore', 'optimal_bandwidth', 'min_bScore']
    vars_dict = {var_name: globals()[var_name] for var_name in variable_names}
    pickle.dump(vars_dict, f)
