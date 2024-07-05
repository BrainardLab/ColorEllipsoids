#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 13:15:58 2024

@author: fangfang
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
import sys
import os
import pickle
import numpy as np

def compute_Bures_Wasserstein_distance(M1, M2):
    # Compute the square root of M1
    sqrt_M1 = sqrtm(M1)
    # Compute the product sqrt(M1) * M2 * sqrt(M1)
    product = sqrt_M1 @ M2 @ sqrt_M1
    # Compute the square root of the product
    sqrt_product = sqrtm(product)
    # Calculate the Bures-Wasserstein distance
    BW_distance = np.sqrt(np.trace(M1) + np.trace(M2) - 2 * np.trace(sqrt_product))
    return BW_distance
        
def compute_normalized_Bures_similarity(M1, M2):
    # Compute the product inside the trace
    inner_product = sqrtm(sqrtm(M1) @ M2 @ sqrtm(M1))  
    # Calculate the trace of the product
    trace_value = np.trace(inner_product)    
    # Normalize by the geometric mean of the traces of M1 and M2
    normalization_factor = np.sqrt(np.trace(M1) * np.trace(M2))    
    # Calculate NBS
    NBS = trace_value / normalization_factor    
    return NBS
                    
def plot_similarity_metric_scores(similarity_score, bin_edges, **kwargs):
    nSets = similarity_score.shape[0]
    
    # Default parameters for ellipsoid fitting. Can be overridden by kwargs.
    pltP = {
        'y_ub':25,
        'legend_labels':[None for i in range(nSets)],
        'legend_title':'',
        'cmap':[],
        'xlabel':'Similarity metric',
        'ylabel':'Frequency',
        'xticks':[],
        'figDir':'',
        'saveFig': False,
        'figName':'ModelPerformance_metricScores_3Dellipsoids',
        'figName_ext':'',
            } 
    pltP.update(kwargs)
    
    fig, ax = plt.subplots(1, 1, figsize=(5,4.5))
    plt.rcParams['figure.dpi'] = 250
    for j in range(nSets):
        if len(pltP['cmap']) == 0: cmap_l = np.random.rand(1,3)
        else: cmap_l = pltP['cmap'][j];
        ax.hist(similarity_score[j].flatten(), bins = bin_edges, color = cmap_l,\
                alpha = 0.6, edgecolor = [1,1,1], label = pltP['legend_labels'][j])
        #plot the median
        median_j = np.median(similarity_score[j].flatten())
        ax.plot([median_j,median_j], [0,pltP['y_ub']],color = cmap_l, linestyle = '--')
    ax.grid(True, alpha=0.3)
    if pltP['legend_title'] != '':
        ax.legend(title = pltP['legend_title'])
    ax.set_xticks(pltP['xticks'])
    ax.set_ylim([0, pltP['y_ub']])
    ax.set_xlabel(pltP['xlabel'])
    ax.set_ylabel(pltP['ylabel'])
    if pltP['saveFig'] and pltP['figDir'] != '':
        full_path = os.path.join(pltP['figDir'], pltP['figName']+pltP['figName_ext']+'.png')
        fig.savefig(full_path)          

def plot_benchmark_similarity(similarity_score, bin_edges, **kwargs):
    nSets = similarity_score.shape[0]
    
    # Default parameters for ellipsoid fitting. Can be overridden by kwargs.
    pltP = {
        'y_ub':80,
        'cmap':[],
        'xticks':[],
        'xlabel':'Similarity metric',
        'ylabel':'Frequency',
        'linestyle':[],
        'figDir':'',
        'saveFig': False,
        'figName':'ModelPerformance_benchmark_3Dellipsoids',
        'figName_ext':'',
            } 
    pltP.update(kwargs)
    
    fig, ax = plt.subplots(1, 1, figsize=(5,4.5))
    plt.rcParams['figure.dpi'] = 250
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    for m in range(nSets):
        if len(pltP['cmap']) == 0: cmap_m = np.random.rand(1,3)
        else: cmap_m = pltP['cmap'][m];
        if len(pltP['linestyle']) == 0: ls_m = '-';
        else: ls_m = pltP['linestyle'][m]
        counts_m,_ = np.histogram(similarity_score[m].flatten(), bins=bin_edges)
        ax.plot(bin_centers, counts_m,  color = cmap_m, ls = ls_m)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(pltP['xticks'])
    ax.set_yticks(np.linspace(0,pltP['y_ub'],5))
    ax.set_ylim([0,pltP['y_ub']])
    ax.set_xlabel(pltP['xlabel'])
    ax.set_ylabel(pltP['ylabel'])
    if pltP['saveFig'] and pltP['figDir'] != '':
        full_path = os.path.join(pltP['figDir'], pltP['figName']+pltP['figName_ext']+'.png')
        fig.savefig(full_path)    