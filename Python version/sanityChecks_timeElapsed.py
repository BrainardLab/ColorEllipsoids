#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 11:20:24 2024

@author: fh862-adm
"""
import matplotlib.pyplot as plt
import dill as pickled
import sys
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
sys.path.append("/Users/fh862-adm/Documents/GitHub/ellipsoids/ellipsoids")
from core import optim, oddity_task
from core.model_predictions import wishart_model_pred
from core.wishart_process import WishartProcessModel
from plotting.adaptive_sampling_plotting import SamplingRefCompPairVisualization
from analysis.color_thres import color_thresholds
from analysis.config_generator import ConfigGenerator
from analysis.sim_trials import SimulateTrialGivenWishart
from plotting.AEPsych_predictions_plotting import AEPsych_predictions_visualization
from plotting.wishart_predictions_plotting import WishartPredictionsVisualization

#define output directory for output files and figures
COLOR_DIMENSION = 4
SUBJ = [0, 1]
baseDir = '/Users/fh862-adm/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'
output_figDir = baseDir + f'META_analysis/ModelFitting_FigFiles/{COLOR_DIMENSION}dTask/speed testing/'
output_fileDir = baseDir + f'META_analysis/ModelFitting_DataFiles/{COLOR_DIMENSION}dTask/speed testing/'
sys.path.append(output_fileDir)
nTrials, time_elapsed, mean_time_elapsed = [],[],[]
onset_eavc = 150
outlier_max = 100

for s in SUBJ:
    output_file = 'Fitted_byWishart_isothreshold_Isoluminant plane_4DExpt_sim00800total_'+\
        f'from00150_50_50_50_650_AEPsychSampling_EAVC_sub{s}.pkl'
    full_path = f"{output_fileDir}{output_file}"
    #load file
    with open(full_path, 'rb') as f:
        data_load = pickled.load(f)
    time_elapsed_s = np.array(data_load['AEPsych_trial_given_Wishart_gt'].time_elapsed)
    time_elapsed_s[time_elapsed_s > outlier_max] = np.nan
    mean_time_elapsed.append(np.nanmean(time_elapsed_s[onset_eavc:]))
    nTrials.append(len(time_elapsed_s))
    time_elapsed.append(time_elapsed_s)
    
    
#%% plot them together
plt.rcParams['font.sans-serif'] = ['Arial']
mc = np.array([[79, 121, 66],[124, 66, 168]])/255
lbl = ['GPU - no', 'GPU - yes']
fig, ax = plt.subplots(2,1, figsize = (5,7), dpi = 1024)
for s in range(len(SUBJ)):
    ax[0].plot(list(range(nTrials[s])), time_elapsed[s], c = mc[s], lw = 0.5, label = lbl[s])
    ax[0].plot([onset_eavc,nTrials[s]], [mean_time_elapsed[s], mean_time_elapsed[s]], 
               c = mc[s], ls = '--', label = f'{lbl[s]}: mean = {mean_time_elapsed[s]:.2f} s')
ax[0].set_ylim([0,60])
ax[0].set_ylabel('Time elapsed (s)')
ax[0].legend(loc='upper left', title='')

ax[1].plot(list(range(nTrials[s])), time_elapsed[1] - time_elapsed[0], c = 'k',
           lw = .5, label = 'Difference between GPU - yes and GPU - no')
ax[1].plot([onset_eavc,nTrials[0]], (mean_time_elapsed[1] - mean_time_elapsed[0])*np.array([1,1]),
           c = 'k', ls = '--', label = f'Mean = {(mean_time_elapsed[1] - mean_time_elapsed[0]):.2f} s')
ax[1].set_ylim([-40,60])
ax[1].legend(loc='upper left', title='')
ax[1].set_xlabel('Trial number')
ax[1].set_ylabel('Time elapsed (s)')
fig.savefig(f'{output_figDir}Time_elapsed_{output_file[7:-5]}{SUBJ}.pdf')



