#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 20:10:32 2024

@author: fangfang
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import dill as pickled
import sys
import numpy as np
sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids")
from plotting.wishart_predictions_plotting import WishartPredictionsVisualization
sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/Python version/')
from analysis.utils_load import select_file_and_get_path, extract_sub_number

#%% load files
# Select a file and get its directory and file name
file_dir, file_name = select_file_and_get_path()

# Extract the subject number (subN) from the file directory
subN = extract_sub_number(file_dir)

# Ensure the subject number extracted from the directory matches the one from the file name
assert(subN == extract_sub_number(file_name)), "Mismatch in subject number between file directory and file name."

# Set the output figure directory to the selected file's directory
output_figDir_fits = file_dir

# Replace 'DataFiles' with 'FigFiles' in the directory path (useful for saving figures)
output_figDir_fits = output_figDir_fits.replace('DataFiles', 'FigFiles')

# Output the modified directory path
print(f"Figure directory: {output_figDir_fits}")

#%%
full_dir = f"{file_dir}/{file_name}"
with open(full_dir, 'rb') as f: data_load = pickled.load(f)
for key, value in data_load.items():
    globals()[key] = value  # This will create variables named after the keys in your dictionary

class expt_data:
    def __init__(self, xref_all, x1_all, y_all, pseudo_order):
        self.xref_all = xref_all
        self.x1_all = x1_all
        self.y_all = y_all
        self.pseudo_order = pseudo_order
        
expt_trial = expt_data(data[1], data[2], data[0], session_order[:,np.newaxis])

wishart_pred_vis = WishartPredictionsVisualization(expt_trial,
                                                   model_pred_Wishart.model, 
                                                   model_pred_Wishart, 
                                                   color_thres_data,
                                                   fig_dir = output_figDir_fits, 
                                                   save_fig = False)

grid_samples_temp = color_thres_data.N_unit_to_W_unit(np.linspace(0.2, 0.8,5))
grid_samples = jnp.stack(jnp.meshgrid(*[grid_samples_temp for _ in range(2)]), axis=-1)

fig2, ax2 = plt.subplots(1, 1, figsize = (3.81,4.2), dpi= 1024)
wishart_pred_vis.plot_2D(
    grid, 
    grid_samples,
    ax = ax2,
    visualize_model_estimatedCov = False,
    visualize_samples= True,
    visualize_gt = False,
    samples_alpha = 0.5,
    samples_s = 10,
    sigma_lw = 0.5,
    sigma_alpha = 1,
    modelpred_alpha = 1,
    modelpred_lw = 1,
    modelpred_lc = 'k',
    modelpred_ls = '-',
    samples_label = 'Empirical data',
    samples_colorcoded_resp = True)
ax2.plot([0, 2*(5/255)], [-0.065, -0.065], c = 'k')
ax2.text(2*(5.5/255), -0.0675, '5 / 255')
ax2.set_xlim(np.array([-1, 1]) * (20/255))
ax2.set_ylim(np.array([-1, 1]) * (20/255))
ax2.set_xlabel('Wishart space dimension 1');
ax2.set_ylabel('Wishart space dimension 2');
ax2.set_title('Isoluminant plane');
plt.show()
# Save the figure as a PDF
fig2.savefig(output_figDir_fits+f"/{file_name[:-4]}_wYesNo_ZoomedInGray.pdf",
             format='pdf', bbox_inches='tight')




