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

#define output directory for output files and figures
COLOR_DIMENSION = 4
SUBJ = [5,6]
baseDir = '/Volumes/T9/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'
output_figDir = baseDir + f'META_analysis/Simulation_FigFiles/{COLOR_DIMENSION}dTask/speed testing/'
output_fileDir = baseDir + f'META_analysis/Simulation_DataFiles/{COLOR_DIMENSION}dTask/speed testing/'
sys.path.append(output_fileDir)
nTrials, time_elapsed, mean_time_elapsed = [],[],[]
onset_eavc = 100#750
outlier_max = 10

for s in SUBJ:
    output_file = f'Speed_test_sub{s}.pkl'
    full_path = f"{output_fileDir}sub{s}/{output_file}"
    #load file
    with open(full_path, 'rb') as f:
        data_load = pickled.load(f)
    time_elapsed_s = np.array(data_load['AEPsych_trial_given_Wishart_gt'].time_elapsed)
    time_elapsed_s[time_elapsed_s > outlier_max] = outlier_max#np.nan
    mean_time_elapsed.append(np.nanmean(time_elapsed_s[onset_eavc:]))
    nTrials.append(len(time_elapsed_s))
    time_elapsed.append(time_elapsed_s)
    
    
#%% plot them together
plt.rcParams['font.sans-serif'] = ['Arial']
mc = np.array([[79, 121, 66],[124, 66, 168]])/255
lbl = ['Generating: CPU; Fitting: CPU', 'Generating: GPU; Fitting: GPU']
fig, ax = plt.subplots(2,1, figsize = (5,7), dpi = 1024)
for s in range(len(SUBJ)):
    ax[0].plot(list(range(nTrials[s])), time_elapsed[s], c = mc[s], lw = 0.5, label = lbl[s])
    ax[0].plot([onset_eavc,nTrials[s]], [mean_time_elapsed[s], mean_time_elapsed[s]], 
               c = mc[s], ls = '--', label = f'{lbl[s]}. Mean = {mean_time_elapsed[s]:.2f} s')
ax[0].set_ylim([0,15])
ax[0].set_ylabel('Time elapsed (s)')
ax[0].legend(loc='upper left', title='')

ax[1].plot(list(range(nTrials[s])), time_elapsed[1] - time_elapsed[0], c = 'k',
           lw = .5, label = 'Difference between the two conditions')
ax[1].plot([onset_eavc,nTrials[0]], (mean_time_elapsed[1] - mean_time_elapsed[0])*np.array([1,1]),
           c = 'k', ls = '--', label = f'Mean = {(mean_time_elapsed[1] - mean_time_elapsed[0]):.2f} s')
ax[1].set_ylim([-10,10])
ax[1].legend(loc='upper left', title='')
ax[1].set_xlabel('Trial number')
ax[1].set_ylabel('Time elapsed (s)')
plt.tight_layout()
fig.savefig(f'{output_figDir}Time_elapsed_{output_file[7:-5]}{SUBJ}_max10.pdf')


