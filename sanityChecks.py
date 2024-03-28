#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 17:20:55 2024

@author: fangfang
"""

import numpy as np
from scipy.stats import logistic
import matplotlib.pyplot as plt
import matplotlib.patches as patches

nGrids = 11
xref = np.zeros((2,nGrids))
x0   = np.zeros((2,nGrids))
x1_lb = -4
x1_ub = 4
x1_vec = np.linspace(x1_lb, x1_ub, nGrids)
x1   = np.stack((x1_vec, np.zeros(nGrids)), axis = 0)
sig  = np.eye(2)

nSims_gt = int(1e5)

def sim_prob_choosing_x0(xref, x0, x1, nS, flag_smooth = False, bandwidth = 0.1,
                         nbins = 1000):
    nGrids = xref.shape[1]
    ecdf, diff_sorted = np.full((nS,nGrids), np.nan), np.full((nS,nGrids), np.nan)
    pChoosingX0 = np.full((nGrids), np.nan)
    if flag_smooth:
        bins_range, logit_pdf, logit_cdf = np.full((nbins,nGrids), np.nan), np.full((nbins,nGrids), np.nan), np.full((nbins,nGrids), np.nan)
    
    for i in range(nGrids):
        zref = np.random.multivariate_normal(xref[:,i], sig, size = (1, nS))
        z0 = np.random.multivariate_normal(x0[:,i], sig, size = (1, nS))
        z1 = np.random.multivariate_normal(x1[:,i], sig, size = (1, nS))
        
        # Compute squared distance of each probe stimulus to reference
        z0_to_zref = np.sum((z0 - zref) ** 2, axis=2)
        z1_to_zref = np.sum((z1 - zref) ** 2, axis=2)
        
        #signed difference
        diff = z0_to_zref - z1_to_zref
        
        #Sort the data in ascending order
        diff_sorted[:,i] = np.sort(diff[0])
        
        ecdf[:,i] = np.arange(1, len(diff_sorted) + 1) / len(diff_sorted)
        
        if flag_smooth:
            bins_range[:,i] = np.linspace(-100, 50, nbins)
            logit_pdf_i = np.zeros((nbins))
            for n in range(nS):
                logit_pdf_i += logistic.pdf(bins_range[:,i], diff_sorted[n,i], bandwidth)
            logit_pdf[:,i] = logit_pdf_i
            logit_cdf[:,i] = np.cumsum(logit_pdf[:,i])
            logit_cdf[:,i] = logit_cdf[:,i]/logit_cdf[-1,i]
            min_idx = np.argmin(np.abs(bins_range[:,i]))
            pChoosingX0[i] = logit_cdf[min_idx,i]/logit_cdf[-1,i]
        else:
            #find the value that's closest to 0
            min_idx = np.argmin(np.abs(diff_sorted[:,i]))
        
            #probability of correct
            pChoosingX0[i] = ecdf[min_idx, i]
    if flag_smooth == False:
        return ecdf, diff_sorted, pChoosingX0
    else:
        return bins_range, logit_pdf, logit_cdf, pChoosingX0

#ground truth
ecdf_gt, diff_sorted_gt, pChoosingX0_gt = sim_prob_choosing_x0(xref, x0, x1, nSims_gt)

#%%
nSims_vec = np.array([10, 100,1000])
bandwidth_vec = np.array([1e-1,1,10])
nBins     = 1000
nRepeats  = 120
bins_range = np.full((nRepeats,len(nSims_vec), len(bandwidth_vec), nBins,nGrids),np.nan)
logit_pdf = np.full((nRepeats,len(nSims_vec), len(bandwidth_vec), nBins,nGrids),np.nan)
logit_cdf = np.full((nRepeats,len(nSims_vec), len(bandwidth_vec), nBins,nGrids),np.nan)
pChoosingX0 = np.full((nRepeats,len(nSims_vec), len(bandwidth_vec), nGrids), np.nan)

for n in range(nRepeats):
    print(n)
    for i in range(len(nSims_vec)):
        for j in range(len(bandwidth_vec)):
            bins_range[n,i,j,:,:], logit_pdf[n,i,j,:,:], logit_cdf[n,i,j,:,:],\
                pChoosingX0[n,i,j,:] = sim_prob_choosing_x0(xref, x0, x1,\
                nSims_vec[i],flag_smooth = True, bandwidth = bandwidth_vec[j], \
                nbins = 1000)
                    
#%% compute mean and 95% confidence interval 
CI_lb_idx = int(np.ceil(nRepeats*0.025))
CI_ub_idx = int(np.floor(nRepeats*0.975))
mean_pChoosingX0 = np.mean(pChoosingX0, axis = 0)
pChoosingX0_sort = np.sort(pChoosingX0, axis = 0)
CI_lb_pChoosingX0 = pChoosingX0_sort[CI_lb_idx,:,:,:]
CI_ub_pChoosingX0 = pChoosingX0_sort[CI_ub_idx,:,:,:]
        

#%% plotting the ground truths
plt.rcParams['figure.dpi'] = 250
fig, ax = plt.subplots(1, 1)
for i in range(nGrids):
    plt.plot(diff_sorted_gt[:,i], ecdf_gt[:,i], linestyle='-',\
             label = str(np.round(x1_vec[i],1)))
plt.legend(title = 'Horizontal value of x1')
plt.xlabel('||z0 - zref|| - ||z1 - zref||')
plt.ylabel('Empirical Cumulative Distribution Function')
plt.title('Ground truth (#Sims = '+str(nSims_gt)+', bandwidth = 0)')
plt.grid(True)
plt.savefig('SanityCheck_fig1.png')
plt.show()

#plot one instance 
slc_run = 0
fig, ax = plt.subplots(len(nSims_vec), len(bandwidth_vec), figsize = (8,8))
for i in range(len(nSims_vec)):
    for j in range(len(bandwidth_vec)):
        for k in range(nGrids):
            ax[i][j].plot(bins_range[slc_run,i,j,:,k], logit_cdf[slc_run,i,j,:,k],\
                          linestyle='-')        
        if j == 0:
            ax[i][j].set_ylabel('#Sims =' + str(nSims_vec[i]))
        if i == 0:
            ax[i][j].set_title('bandwidth =' + str(bandwidth_vec[j]))
        ax[i][j].grid(True)
plt.savefig('SanityCheck_fig2.png')

#%% ground truth
fig2, ax2 = plt.subplots(1,1)
plt.plot(x1[0,:], pChoosingX0_gt, c = [0.5,0.5,0.5])
for i in range(nGrids):
    plt.scatter(x1[0,i], pChoosingX0_gt[i], marker='o', s = 50,\
                label = str(np.round(x1_vec[i],1)))
plt.legend(ncol = 2)
plt.xlim([x1_lb, x1_ub])
plt.ylim([0,1])
plt.xlabel('Horizontal value of x1')
plt.ylabel('Probability of reporting x0 as more similar to xref')
plt.title('Ground truth (#Sims = '+str(nSims_gt)+', bandwidth = 0)')
plt.grid(True)
plt.savefig('SanityCheck_fig3.png')

#
fig3, ax3 = plt.subplots(len(nSims_vec),3, figsize = (8,8))
for i in range(len(nSims_vec)):
    for j in range(len(bandwidth_vec)):
        CI_lb_ij = CI_lb_pChoosingX0[i,j,:]
        CI_ub_ij = CI_ub_pChoosingX0[i,j,:]
        CI_bds = np.concatenate((CI_lb_ij,CI_ub_ij), axis = 0)
        CI_bds = np.append(CI_bds, CI_lb_ij[0])
        
        x1_temp = x1[0,:]
        x1_repeat = np.concatenate((x1_temp,x1_temp[::-1]), axis = 0)
        x1_repeat = np.append(x1_repeat,x1_temp[0])
        points = np.column_stack((x1_repeat, CI_bds))
        
        polygon = patches.Polygon(points, closed=True, linewidth=1, edgecolor='k',\
                        facecolor='lightblue',alpha = 0.3, label = '95% CI (for 120 \nrepetitions)')
        # Add the polygon patch to the Axes
        ax3[i,j].add_patch(polygon)
        ax3[i,j].plot(x1[0,:], mean_pChoosingX0[i,j,:], c ='k',\
                    label = 'Mean')
        for k in range(nGrids):
            ax3[i,j].scatter(x1[0,k], pChoosingX0_gt[k], marker='o', s = 50)
            
        ax3[i,j].set_xlim([x1_lb, x1_ub])
        ax3[i,j].set_ylim([0,1])
        if i == 0: ax3[i,j].set_title('Bandwidth = ' + str(bandwidth_vec[j]))
        if j == 0: ax3[i,j].set_ylabel('#Sims = '+str(nSims_vec[i]))
        if i == len(nSims_vec)-1 and j == 0:
            ax3[i,j].legend()
plt.savefig('SanityCheck_fig4.png')

   