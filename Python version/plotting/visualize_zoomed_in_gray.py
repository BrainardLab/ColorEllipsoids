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

#%% visualization
# Construct the full path to the selected file
full_dir = f"{file_dir}/{file_name}"

# Load the data from the file using pickle
with open(full_dir, 'rb') as f:
    data_load = pickled.load(f)

# Create variables in the global namespace for each key in the loaded dictionary
for key, value in data_load.items():
    globals()[key] = value  # Warning: This can overwrite existing variables with the same name

# Define a class to organize experiment trial data
class expt_data:
    def __init__(self, xref_all, x1_all, y_all, pseudo_order):
        self.xref_all = xref_all  # Reference stimuli for all trials
        self.x1_all = x1_all      # Comparison stimuli for all trials
        self.y_all = y_all        # Binary response data for all trials
        self.pseudo_order = pseudo_order  # Trial pseudo order for visualization/analysis
        
# Initialize an instance of the expt_data class with the loaded data
expt_trial = expt_data(data[1], data[2], data[0], session_order[:, np.newaxis])

wishart_pred_vis = WishartPredictionsVisualization(expt_trial,
                                                   model_pred_Wishart.model, 
                                                   model_pred_Wishart, 
                                                   color_thres_data,
                                                   fig_dir = output_figDir_fits, 
                                                   save_fig = False)

# Define a grid of sample points in Wishart space for visualization
grid_samples_temp = color_thres_data.N_unit_to_W_unit(np.linspace(0.2, 0.8, 5))
grid_samples = jnp.stack(jnp.meshgrid(*[grid_samples_temp for _ in range(2)]), axis=-1)

# Create a figure and axis for the 2D plot
fig2, ax2 = plt.subplots(1, 1, figsize=(3.81, 4.2), dpi=1024)
wishart_pred_vis.plot_2D(
    grid, 
    grid_samples,
    ax=ax2,
    visualize_model_estimatedCov=False,  # Don't visualize the estimated covariance
    visualize_samples=True,              # Visualize sample points
    visualize_gt=False,                  # Don't visualize the ground truth
    samples_alpha=0.5,                   # Transparency for sample points
    samples_s=10,                        # Size of sample points
    sigma_lw=0.5,                        # Line width for sigma visualization
    sigma_alpha=1,                       # Transparency for sigma visualization
    modelpred_alpha=1,                   # Transparency for model predictions
    modelpred_lw=1,                      # Line width for model predictions
    modelpred_lc='k',                    # Line color for model predictions
    modelpred_ls='-',                    # Line style for model predictions
    samples_label='Empirical data',      # Label for sample points
    samples_colorcoded_resp=True         # Color code samples based on responses
)

# Plot a benchmark line representing 5 quantizations in RGB unit
# Multiply by 2 because the plot is in Wishart space
ax2.plot([0, 2 * (5 / 255)], [-0.065, -0.065], c='k')
ax2.text(2 * (5.5 / 255), -0.0675, '5 / 255')
ax2.set_xlim(np.array([-1, 1]) * (20 / 255))
ax2.set_ylim(np.array([-1, 1]) * (20 / 255))
# Set axis labels and plot title
ax2.set_xlabel('Wishart space dimension 1')
ax2.set_ylabel('Wishart space dimension 2')
ax2.set_title('Isoluminant plane')
# Display the plot
plt.show()
fig2.savefig(output_figDir_fits + f"/{file_name[:-4]}_wYesNo_ZoomedInGray.pdf",
             format='pdf', bbox_inches='tight')