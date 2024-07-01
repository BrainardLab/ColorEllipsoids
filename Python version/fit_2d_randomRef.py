#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 16:05:29 2024

@author: fangfang
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import pickle
import sys
import numpy as np
import os
from scipy.linalg import sqrtm
from matplotlib.patches import Rectangle
import imageio.v2 as imageio

#load functions from other modules
sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids")
from core import optim, model_predictions, oddity_task
sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/"+\
                "Python version")
from Simulate_probCorrectResp import sample_rgb_comp_2DNearContour

#%%
def simResp_given_pChoosingX1(p):
    """
    Simulates binary responses based on a specified probability of choosing the 
        correct option.
    
    Parameters
    ----------
    p : an array (size: (n, ))
        Probability of a correct response (must be between 0 and 1)

    Returns
    -------
    np.ndarray or int
        An array or a single binary response (1 for correct, 0 for incorrect), 
        depending on the `size` parameter.

    """

    # Generate random numbers and compare to probability to determine outcomes
    randNum = np.random.rand(*p.shape) 
    binaryResp = (randNum < p).astype(int)
    
    return binaryResp

def covMat_to_ellParamsQ(covM):
    """
    Convert a 2D or 3D covariance matrix to the parameters of the corresponding
    ellipse or ellipsoid.
    Returns different parameters based on the dimensionality of the input:
    - For 2D: Returns the lengths of axes, eigenvectors, and the rotation angle 
        in degrees.
    - For 3D: Returns only the lengths of axes and eigenvectors.

    Parameters:
    - covM (numpy.ndarray): A 2x2 or 3x3 covariance matrix representing the 
        variances and covariances of a two- or three-dimensional dataset. This 
        matrix should be symmetric and positive semi-definite.

    Returns:
    - For 2D: Tuple containing axis lengths, eigenvectors matrix, and rotation 
        angle in degrees.
    - For 3D: Tuple containing axis lengths and eigenvectors matrix.

    """

    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(covM)
    sorted_indices = np.argsort(eigenvalues)[::-1]  # Sort eigenvalues in descending order
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    # Semi-axes lengths (for 1 std deviation)
    axes_lengths = np.sqrt(eigenvalues)

    if covM.shape == (2, 2):  # Additional computation for 2D case
        # Rotation angle in degrees (only needed for 2D)
        theta = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        return eigenvalues, eigenvectors, axes_lengths, theta
    elif covM.shape == (3, 3):  # For 3D case, return less information
        return eigenvalues, eigenvectors, axes_lengths
    else:
        raise ValueError("Input must be a 2x2 or 3x3 covariance matrix")

def plot_randSamples(xref, xcomp, idx_fixedPlane, fixedVal, bounds, **kwargs):
    # Default parameters for ellipsoid fitting. Can be overridden by kwargs.
    pltP = {
        'visualize_bounds': True,
        'visualize_lines': False,
        'ref_marker': '+',
        'ref_markersize': 20,
        'ref_markeralpha': 0.8,
        'comp_marker': 'o',
        'comp_markersize': 4,
        'comp_markeralpha': 0.8,        
        'plane_2D':'',
        'fontsize':8,
        'saveFig':False,
        'figDir':'',
        'figName':'RandomSamples'} 
    pltP.update(kwargs)
    
    fig, ax = plt.subplots(1, 1, figsize = (3,3.5))
    plt.rcParams['figure.dpi'] = 250 
    cmap = (xref+1)/2
    cmap = np.insert(cmap, idx_fixedPlane, np.ones((xref.shape[0],))*fixedVal, axis=1)
    # Add grey patch
    if pltP['visualize_bounds']:
        rectangle = Rectangle((bounds[0], bounds[0]), bounds[1] - bounds[0],\
                              bounds[1] - bounds[0], facecolor='grey', alpha=0.1)  # Adjust alpha for transparency
        rectangle.set_label('Bounds for the reference')  # Set the label here
        ax.add_patch(rectangle)
    
    ax.scatter(xref[:,0],xref[:,1], c = cmap, marker = pltP['ref_marker'],\
               s = pltP['ref_markersize'], alpha = pltP['ref_markeralpha'],\
               label = 'Reference stimulus')
    ax.scatter(xcomp[:,0], xcomp[:,1], c = cmap, marker = pltP['comp_marker'],\
               s = pltP['comp_markersize'], alpha = pltP['comp_markeralpha'],\
               label = 'Comparison stimulus') 
    if pltP['visualize_lines']:
        for l in range(xref.shape[0]):
            ax.plot([xref[l,0],xcomp[l,0]], [xref[l,1],xcomp[l,1]], c = cmap[l], lw = 0.5)
    
    plt.grid(alpha = 0.2)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1])
    ticks = np.sort(np.concatenate((np.linspace(-0.5, 0.5, 3), np.array([-0.85, 0.85]))))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(axis='both', which='major', labelsize=pltP['fontsize'])
    if pltP['plane_2D'] != '':
        ax.set_xlabel(pltP['plane_2D'][0], fontsize=pltP['fontsize']);
        ax.set_ylabel(pltP['plane_2D'][1], fontsize=pltP['fontsize'])
        ax.set_title(pltP['plane_2D'], fontsize=pltP['fontsize'])
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.47),fontsize = pltP['fontsize'])
    fig.tight_layout(); plt.show()
    if pltP['saveFig'] and pltP['figDir'] != '':
        full_path = os.path.join(pltP['figDir'],pltP['figName']+'.png') 
        fig.savefig(full_path)   
    
#%% load files
#load a file
path_str1        = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                   'ELPS_analysis/Simulation_DataFiles/'
os.chdir(path_str1)
file_CIE         = 'Isothreshold_contour_CIELABderived_fixedVal0.5.pkl'
full_path        = f"{path_str1}{file_CIE}"
with open(full_path, 'rb') as f: 
    data_load    = pickle.load(f)
results          = data_load[2]

#set stimulus info
plane_2D_dict    = {'GB plane': 0, 'RB plane': 1, 'RG plane': 2}
plane_2D         = 'RG plane' 
plane_2D_idx     = plane_2D_dict[plane_2D]
varying_RGBplane = [0,1] #two indices that correspond to the varying planes
fixedPlane_val   = 0.5 #fixed value for the fixed plane
nSims            = 240 # Number of simulations or trials per reference stimulus.

#set path and load another file
path_str2     = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                         'ELPS_analysis/ModelFitting_DataFiles/2D_oddity_task/'  
file_fits     = 'Fitted_isothreshold_'+plane_2D+'_sim'+str(nSims)+'perCond_'+\
                'samplingNearContour_jitter0.1_bandwidth0.005' + '.pkl'
full_path     = f"{path_str2}{file_fits}"
with open(full_path, 'rb') as f:  
    data_load = pickle.load(f)
#load info we need for later
W_est         = data_load['W_est']
model         = data_load['model']
opt_params    = data_load['opt_params']

#%% simulate data
# Define the boundaries for the reference stimuli
xref_bds        = [-0.85, 0.85] # Square boundary limits for the stimuli
# Amount of jitter added to comparison stimulus sampling
jitter          = 0.5
# Total number of simulations to perform
nSims_total     = 6000 
# Draw random reference stimuli within the specified boundary
xrand           = np.array(np.random.rand(nSims_total,2)*(xref_bds[-1]-xref_bds[0]) + xref_bds[0])
# Compute the covariance matrices for each reference stimulus based on a model
Sigmas_est_grid = model.compute_Sigmas(model.compute_U(W_est, xrand))
# Scaler used to adjust covariance matrices to approximate 78% correct responses
scaler_sigma     = 10.6

# Initialize arrays to hold the comparison stimuli, predicted probabilities, and responses
x1rand           = np.full(xrand.shape, jnp.nan)
pX1              = np.full((nSims_total,), jnp.nan)
resp             = np.full((nSims_total,), jnp.nan)

# Process each randomly sampled reference stimulus
for i in range(nSims_total):
    # Scale the estimated covariance matrix for the current stimulus
    Sigmas_est_grid_i = scaler_sigma*Sigmas_est_grid[i]
    # Convert the covariance matrix to ellipse parameters (semi-major axis, 
    # semi-minor axis, rotation angle)
    _, _, ab_i, theta_i = covMat_to_ellParamsQ(Sigmas_est_grid_i)
    # Pack the center of the ellipse with its parameters
    paramEllipse_i = [*xrand[i], *ab_i, theta_i]
    # Sample a comparison stimulus near the contour of the reference stimulus, applying jitter
    x1rand_temp = sample_rgb_comp_2DNearContour(xrand[i], varying_RGBplane, 0,\
                                                1, paramEllipse_i, jitter)
    # Reshape and clip the sampled comparison stimulus to fit within the [-1, 1] bounds        
    x1rand_reshape = x1rand_temp[varying_RGBplane].T
    x1rand[i] = np.clip(x1rand_reshape,-1,1)
    
xrand = jnp.array(xrand)
x1rand = jnp.array(x1rand)
# compute weighted sum of basis function at rgb_ref 
Uref = model.compute_U(W_est, xrand)
# compute weighted sum of basis function at rgb_comp
U1   = model.compute_U(W_est, x1rand)
# Predict the probability of choosing the comparison stimulus over the reference
pX1 = oddity_task.oddity_prediction((xrand, x1rand, Uref, U1),\
                  jax.random.split(data_load['OPT_KEY'], num = nSims_total),\
                  opt_params['mc_samples'], opt_params['bandwidth'],\
                  model.diag_term, oddity_task.simulate_oddity)
# Simulate a response based on the predicted probability
resp = jnp.array(simResp_given_pChoosingX1(pX1))

# Package the processed data into a tuple for further use
data_rand = (resp, xrand, x1rand)

#%% visualize randomly sampled data
#specify where the figures need to be saved
output_figDir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                         'ELPS_analysis/Simulation_FigFiles/'
saveFig = False
#visualize the figure with different number of trials
slc_datapoints_to_show = [2**i for i in range(11)]
for n in slc_datapoints_to_show:
    plot_randSamples(xrand[:n], x1rand[:n], plane_2D_idx, fixedPlane_val,\
                     xref_bds, plane_2D = plane_2D, visualize_lines = True,\
                     saveFig = saveFig, figDir = output_figDir, \
                     figName = f"Sims_isothreshold_{plane_2D}_sim{n:04}total_"+\
                        "samplingRandom_wFittedW_"+f"jitter{jitter:.1f}")
        
# make a gif
gif_name = f'Sims_isothreshold_{plane_2D}_samplingRandom_wFittedW_jitter{jitter:.1f}.gif'
if saveFig:
    images = [img for img in os.listdir(output_figDir) \
              if img.endswith(f'total_samplingRandom_wFittedW_jitter{jitter:.1f}.png')]
    images.sort()  # Sort the images by name (optional)
    
    # Load images using imageio.v2 explicitly to avoid deprecation warnings
    image_list = [imageio.imread(f"{output_figDir}/{img}") for img in images]
    
    # Create a GIF
    output_path = f"{output_figDir}{gif_name}" 
    imageio.mimsave(output_path, image_list, fps=2)  

#%% Fit the WP model to the randomly selected data
# Random number generator seeds
W_INIT_KEY   = jax.random.PRNGKey(220)  # Key to initialize `W_est`. 
OPT_KEY      = jax.random.PRNGKey(444)  # Key passed to optimizer.

# Fit model, initialized at random W
W_init = model.sample_W_prior(W_INIT_KEY) 

opt_params['learning_rate'] =1e-4
W_recover, iters, objhist = optim.optimize_posterior(
    W_init, data_rand, model, OPT_KEY,
    opt_params,
    oddity_task.simulate_oddity,
    total_steps=1000,
    save_every=1,
    show_progress=True
)

fig, ax = plt.subplots(1, 1)
ax.plot(iters, objhist)
fig.tight_layout()

#%%
# -----------------------------
# Rocover covariance matrices
# -----------------------------
# Specify grid over stimulus space
num_grid_pts = 5
xgrid = jnp.stack(jnp.meshgrid(*[jnp.linspace(-0.6, 0.6, num_grid_pts) \
                                 for _ in range(model.num_dims)]), axis=-1)

Sigmas_recover_grid = model.compute_Sigmas(model.compute_U(W_recover, xgrid))

# -----------------------------
# Compute model predictions
# -----------------------------
target_pC               = 0.78
ngrid_search            = 1000
bds_scaler_gridsearch   = [0.25, 8]
nTheta                  = 200
scaler_x1               = 5
#sample total of 16 directions (0 to 360 deg) 
numDirPts     = 16
grid_theta    = np.linspace(0,2*np.pi-np.pi/8,numDirPts)
grid_theta_xy = np.stack((np.cos(grid_theta),np.sin(grid_theta)),axis = 0)

recover_fitEllipse_scaled, recover_fitEllipse_unscaled,\
    recover_rgb_comp_scaled, recover_rgb_contour_cov, params_ellipses =\
    model_predictions.convert_Sig_2DisothresholdContour_oddity_batch(np.transpose(\
    xgrid,(2,0,1)), varying_RGBplane, grid_theta_xy, target_pC,\
    W_recover, model,oddity_task.simulate_oddity,\
    results['opt_vecLen'], ngrid_bruteforce = ngrid_search,\
    scaler_bds_bruteforce = bds_scaler_gridsearch, scaler_x1 = scaler_x1,\
    nThetaEllipse = nTheta, mc_samples = opt_params['mc_samples'], \
    bandwidth = opt_params['bandwidth'], opt_key = OPT_KEY)

#%% visualize the model predictions and compare that with the ground truth
outputDir_fig = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                        'ELPS_analysis/ModelFitting_FigFiles/Python_version/2D_oddity_task/'
gt_sigma = results['fitEllipse_scaled'][plane_2D_idx]
gt_sigma_scaled = (gt_sigma * 2 - 1)
model_predictions.plot_2D_modelPredictions_byWishart(
    xgrid, xgrid, [], gt_sigma_scaled, Sigmas_recover_grid, 
    recover_fitEllipse_unscaled, plane_2D_idx,\
    visualize_samples= False, visualize_sigma = False,\
    visualize_groundTruth = True, visualize_modelPred = True,\
    gt_mc = 'r', gt_ls = '--', gt_lw = 1, gt_alpha = 0.5, modelpred_mc = 'g',\
    modelpred_ls = '-', modelpred_lw = 2, modelpred_alpha = 0.5,\
    plane_2D = plane_2D, saveFig = True, figDir = outputDir_fig,\
    figName = 'Fitted'+gif_name[4:-4]+ '_nSims'+str(nSims_total)+'total')
    
#%% save data
outputDir = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
                        'ELPS_analysis/ModelFitting_DataFiles/2D_oddity_task/'
output_file = 'Fitted'+gif_name[4:-4] + '_nSims'+str(nSims_total)+'total.pkl'
full_path = f"{outputDir}{output_file}"

variable_names = ['plane_2D', 'jitter','nSims_total', 'data_rand','pX1','model',\
                  'W_INIT_KEY', 'OPT_KEY', 'W_init','opt_params', 'W_est',\
                  'Sigmas_est_grid','W_recover','iters', 'objhist','xgrid',\
                  'Sigmas_recover_grid', 'recover_fitEllipse_scaled',\
                  'recover_fitEllipse_unscaled', 'recover_rgb_comp_scaled',\
                  'recover_rgb_contour_cov','params_ellipses','gt_sigma_scaled']
vars_dict = {}
for i in variable_names: vars_dict[i] = eval(i)

# Write the list of dictionaries to a file using pickle
with open(full_path, 'wb') as f:
    pickle.dump(vars_dict, f)

