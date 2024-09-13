#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 17:01:45 2024

@author: fangfang
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import dill as pickle
import sys
import numpy as np

sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids')
from core import oddity_task, model_predictions, optim
from core.wishart_process import WishartProcessModel
from core.model_predictions import wishart_model_pred
from analysis.color_thres import color_thresholds
from analysis.ellipses_tools import covMat3D_to_2DsurfaceSlice
from plotting.wishart_predictions_plotting import WishartPredictionsVisualization
sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/Python version')
from plotting.trial_placement_nonadaptive_plotting import TrialPlacementVisualization
from data_reorg import organize_data

#%% three variables we need to define for loading the data
rnd_seed  = 2
nSims     = 240
jitter    = 0.3

base_dir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'
output_figDir_fits = base_dir +'ELPS_analysis/ModelFitting_FigFiles/Python_version/3D_oddity_task/'
output_fileDir = base_dir + 'ELPS_analysis/ModelFitting_DataFiles/3D_oddity_task/'

#%% ------------------------------------------
# Load data simulated using CIELab
# ------------------------------------------
#file 1
path_str = base_dir + 'ELPS_analysis/Simulation_DataFiles/'
# Create an instance of the class
color_thres_data = color_thresholds(3, base_dir + 'ELPS_analysis/')
# Load Wishart model fits
color_thres_data.load_CIE_data()  
stim3D = color_thres_data.get_data('stim3D', dataset='CIE_data')
results3D = color_thres_data.get_data('results3D', dataset='CIE_data')

#simulation files
file_sim = f'Sims_isothreshold_ellipsoids_sim{nSims}perCond_samplingNearContour_'+\
            f'jitter{jitter}_seed{rnd_seed}.pkl'
full_path = f"{path_str}{file_sim}"
with open(full_path, 'rb') as f: data_load = pickle.load(f)
sim = data_load[0]

#we take 5 samples from each color dimension
#but sometimes we don't want to sample that finely. Instead, we might just pick
#2 or 3 samples from each color dimension, and see how well the model can 
#interpolate between samples

##################################
#idx_trim  = [0,2,4]
idx_trim = list(range(5))
##################################

"""
Fitting would be easier if we first scale things up, and then scale the model 
predictions back down
"""
scaler_x1  = 5
#x1_raw is unscaled
data, x1_raw, xref_raw = organize_data(3, sim,\
        scaler_x1, slc_idx = idx_trim, visualize_samples = False)
# unpackage data
ref_size_dim1, ref_size_dim2, ref_size_dim3 = x1_raw.shape[0:3]
# if we run oddity task with the reference stimulus fixed at the top
y_jnp, xref_jnp, x0_jnp, x1_jnp = data 
# if we run oddity task with all three stimuli shuffled
data_new = (y_jnp, xref_jnp, x1_jnp)

#%% Visualize the simulated data again
fixedRGB_val_full = np.linspace(0.2,0.8,5)
fixedRGB_val = fixedRGB_val_full[idx_trim]

for fixedPlane, varyingPlanes in zip(['R','G','B'], ['GB','RB','RG']):
    for val in fixedRGB_val:
        TrialPlacementVisualization.plot_3D_sampledComp(stim3D['grid_ref'][idx_trim]*2-1, 
            results3D['fitEllipsoid_unscaled'][idx_trim][:,idx_trim][:,:,idx_trim]*2-1,
            x1_raw, fixedPlane, val*2-1, slc_grid_ref_dim1 = [0,1,2], 
            slc_grid_ref_dim2 = [0,1,2], surf_alpha =  0.1, 
            samples_alpha = 0.1,scaled_neg12pos1 = True,
            bds = 0.05,title = varyingPlanes+' plane',
            saveFig = False, figDir = path_str[0:-10] + 'FigFiles/',\
            figName =f"{file_sim}_{varyingPlanes}plane_fixedVal{val}")

#%%
# -------------------------------
# Constants describing simulation
# -------------------------------
model = WishartProcessModel(
    5,     # Degree of the polynomial basis functions #5
    3,     # Number of stimulus dimensions
    1,     # Number of extra inner dimensions in `U`. #1
    3e-4,  # Scale parameter for prior on `W`.
    0.4,   # Geometric decay rate on `W`.  #0.4
    0,     # Diagonal term setting minimum variance for the ellipsoids.
)

NUM_GRID_PTS = 5           # Number of grid points over stimulus space.
MC_SAMPLES   = 2000        # Number of simulated trials to compute likelihood.
BANDWIDTH    = 5e-3        # Bandwidth for logistic density function. #5e-3

# Random number generator seeds
W_INIT_KEY   = jax.random.PRNGKey(322)  # Key to initialize `W_est`. 
DATA_KEY     = jax.random.PRNGKey(333)  # Key to generate datatset.
OPT_KEY      = jax.random.PRNGKey(444)  # Key passed to optimizer.

# -----------------------------
# Fit W by maximum a posteriori
# -----------------------------
# Fit model, initialized at random W
W_init = 1e-1*model.sample_W_prior(W_INIT_KEY)  

opt_params = {
    "learning_rate": 1e-3, #5e-2
    "momentum": 0.2,
    "mc_samples": MC_SAMPLES,
    "bandwidth": BANDWIDTH,
}
W_est, iters, objhist = optim.optimize_posterior(
    W_init, data_new, model, OPT_KEY,
    opt_params,
    oddity_task.simulate_oddity,
    total_steps=1200,
    save_every=1,
    show_progress=True
)

fig, ax = plt.subplots(1, 1)
ax.plot(iters, objhist)
fig.tight_layout()
plt.show()

#%% -----------------------------
# Rocover covariance matrices
# -----------------------------
# Specify grid over stimulus space
grid_1d = jnp.linspace(jnp.min(xref_jnp), jnp.max(xref_jnp), NUM_GRID_PTS)
grid = jnp.stack(jnp.meshgrid(*[grid_1d for _ in range(model.num_dims)]), axis=-1)
grid_trans = np.transpose(grid,(1,0,2,3))
Sigmas_est_grid = model.compute_Sigmas(model.compute_U(W_est, grid))

# -----------------------------
# Compute model predictions
# -----------------------------
model_pred_Wishart = wishart_model_pred(model, opt_params, NUM_GRID_PTS, 
                                        MC_SAMPLES, BANDWIDTH, W_INIT_KEY,
                                        DATA_KEY, OPT_KEY, W_init, 
                                        W_est, Sigmas_est_grid, 
                                        color_thres_data,
                                        target_pC= 2/3,
                                        scaler_x1 = scaler_x1,
                                        ngrid_bruteforce = 200,
                                        bds_bruteforce = [0.01, 0.25])

model_pred_Wishart.convert_Sig_Threshold_oddity_batch(grid_trans)

#%%derive 2D slices
# Initialize 3D covariance matrices for ground truth and predictions
gt_covMat_CIE   = np.full((NUM_GRID_PTS, NUM_GRID_PTS, NUM_GRID_PTS, 3, 3), np.nan)

# Loop through each reference color in the 3D space
for g1 in range(NUM_GRID_PTS):
    for g2 in range(NUM_GRID_PTS):
        for g3 in range(NUM_GRID_PTS):
            #Convert the ellipsoid parameters to covariance matrices for the 
            #ground truth
            gt_covMat_CIE[g1,g2,g3] = (scaler_x1*2)**2*model_predictions.ellParams_to_covMat(\
                            results3D['ellipsoidParams'][g1,g2,g3]['radii'],\
                            results3D['ellipsoidParams'][g1,g2,g3]['evecs'])
# Compute the 2D ellipse slices from the 3D covariance matrices for both ground 
#truth and predictions
gt_slice_2d_ellipse_CIE = model_predictions.covMat3D_to_2DsurfaceSlice(gt_covMat_CIE)
# compute the slice of model-estimated cov matrix
model_pred_slice_2d_ellipse = model_predictions.covMat3D_to_2DsurfaceSlice(\
                                        model_pred_Wishart.Sigmas_recover_grid)
model_pred_Wishart.Sigmas_recover_grid_slice_2d = np.transpose(model_pred_slice_2d_ellipse,(1,0,2,3,4,5))
    
#%% plot figures and save them as png and gif
fig_outputDir = base_dir+ 'ELPS_analysis/ModelFitting_FigFiles/Python_version/3D_oddity_task/'
name_ext = '_withInterpolations' if np.prod(xref_raw.shape[0:3]) < np.prod(grid.shape[0:3]) else ''
fig_name = 'Fitted' + file_sim[4:-4] + name_ext #+'_maxDeg' + str(model.degree)

class sim_data:
    def __init__(self, xref_all, x1_all):
        self.xref_all = xref_all
        self.x1_all = x1_all
sim_trial_by_CIE = sim_data(xref_jnp, x1_jnp)

wishart_pred_vis = WishartPredictionsVisualization(sim_trial_by_CIE,
                                                     model, 
                                                     model_pred_Wishart, 
                                                     color_thres_data,
                                                     fig_dir = output_figDir_fits, 
                                                     save_fig = False,
                                                     save_gif = False)
        
wishart_pred_vis.plot_3D(
    grid_trans, 
    grid_trans[np.array(idx_trim)][:,np.array(idx_trim)][:,:,np.array(idx_trim)],
    gt_covMat_CIE, 
    gt_slice_2d_ellipse_CIE,
    fontsize = 12,
    samples_alpha = 0.2,
    gt_ls = '--',
    gt_lw = 1,
    gt_alpha = 0.85,#0.85
    modelpred_alpha = 0.55,
    fig_name = fig_name) 

if wishart_pred_vis.save_gif:
    wishart_pred_vis._save_gif(fig_name, fig_name, fig_name_end = '.pdf')

#%% save data
output_file = f"Fitted{file_sim[4:-4]}_bandwidth{BANDWIDTH}{name_ext}_oddity.pkl"
#    '_maxDeg' + str(model.degree)+'.pkl'
full_path4 = f"{output_fileDir}{output_file}"

variable_names = ['data', 'x1_raw', 'xref_raw','sim_trial_by_CIE', 'grid_1d',
                  'grid','grid_trans','iters', 'objhist', 'model',
                  'model_pred_Wishart','gt_covMat_CIE','gt_slice_2d_ellipse_CIE']
vars_dict = {}
for i in variable_names: vars_dict[i] = eval(i)

# Write the list of dictionaries to a file using pickle
with open(full_path4, 'wb') as f:
    pickle.dump(vars_dict, f)

