#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 14:30:18 2024

@author: fangfang
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

#import functions from the other script
func_path = '/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/Python version/'
os.chdir(func_path)
from Simulate_probCorrectResp import query_simCondition,sample_rgb_comp_random, sample_rgb_comp_Gaussian
from simulations_CIELab import convert_rgb_lab

#%%
def sample_rgb_comp_3DNearContour(rgb_ref, nSims, radii, eigenVec, jitter):
    """
    Simulates RGB components near the surface of an ellipsoid contour. This can 
    be used for generating test points in color space around a reference color.
    
    Parameters:
    - rgb_ref: The reference RGB stimulus.
    - nSims: The number of simulated points to generate.
    - radii: The radii of the ellipsoid along its principal axes.
    - eigenVec: The eigenvectors defining the orientation of the ellipsoid.
    - jitter: The standard deviation of the Gaussian noise added to simulate 
        points near the surface.
    
    Returns:
    - rgb_comp_sim: A 3xN matrix containing the simulated RGB components.
    """
    #Uniformly distributed angles between 0 and 2*pi
    randtheta = np.random.rand(1,nSims) * 2 * np.pi
    
    #Uniformly distributed angles between 0 and pi
    randphi = np.random.rand(1,nSims) * np.pi
    
    #Generate random points on the surface of a unit sphere by converting
    # spherical coordinates to Cartesian coordinates, then add Gaussian noise
    # (jitter) to each coordinate to simulate points near the surface.
    randx = np.sin(randphi) * np.cos(randtheta) + np.random.randn(1, nSims) * jitter
    randy = np.sin(randphi) * np.sin(randtheta) + np.random.randn(1, nSims) * jitter
    randz = np.cos(randphi) + np.random.rand(1,nSims) * jitter
    
    #Stretch the random points by the ellipsoid's semi-axes lengths to fit
    # the ellipsoid's shape. This effectively scales the unit sphere points
    # to the size of the ellipsoid along each principal axis.
    randx_stretched = randx * radii[0]
    randy_stretched = randy * radii[1]
    randz_stretched = randz * radii[2]
    
    #Combine the stretched coordinates into a single matrix. Each column
    # represents the (x, y, z) coordinates of a point.
    xyz = np.vstack((randx_stretched, randy_stretched, randz_stretched))
    
    #Rotate and translate the simulated points to their correct positions
    #in RGB space. The rotation is defined by the ellipsoid's eigenvectors
    #(orientation), and the translation moves the ellipsoid to be centered
    #at the reference RGB value. This step aligns the ellipsoid with its
    #proper orientation and position as defined by the input parameters.
    rgb_comp_sim = eigenVec @ xyz + np.reshape(rgb_ref,(3,1))
    
    return rgb_comp_sim

def plot_3D_sampledComp(ref_points, fitEllipsoid_unscaled, sampledComp,
                        fixedPlane, fixedPlaneVal, nPhiEllipse, nThetaEllipse, 
                        **kwargs):
    # Default parameters for ellipsoid fitting. Can be overridden by kwargs.
    pltP = {
        'visualize_ellipsoid':True, 
        'visualize_samples':True, 
        'scaled_neg12pos1':False,
        'slc_grid_ref_dim1': np.array(list(range(len(ref_points)))),
        'slc_grid_ref_dim2': np.array(list(range(len(ref_points)))),
        'surf_alpha': 0.3,
        'samples_alpha': 0.2,
        'markerSize_samples':2,
        'default_viewing_angle':False,
        'x_bds_symmetrical': 0.025,
        'y_bds_symmetrical': 0.025,
        'z_bds_symmetrical': 0.025,
        'fontsize':15,
        'figsize':(8,8),
        'title':'',
        'saveFig':False,
        'figDir':'',
        'figName':'Sampled_comparison_stimuli_3D'
        }
    pltP.update(kwargs)
    
    #Determine the indices of the reference points based on the fixed 
    # plane specified ('R', 'G', or 'B' for different color channels)
    if fixedPlane =='R':
        idx_x = np.where(np.abs(ref_points - fixedPlaneVal)< 1e-6)[0][0]
    elif fixedPlane == 'G':
        idx_y = np.where(np.abs(ref_points - fixedPlaneVal)< 1e-6)[0][0]
    elif fixedPlane == 'B':
        idx_z = np.where(np.abs(ref_points - fixedPlaneVal)< 1e-6)[0][0]
    else:
        return
    
    nGridPts_dim1 = len(pltP['slc_grid_ref_dim1'])
    nGridPts_dim2 = len(pltP['slc_grid_ref_dim2'])
    ref_points_idx = np.array(list(range(len(ref_points))))
    
    fig, axs = plt.subplots(nGridPts_dim2, nGridPts_dim1, subplot_kw={'projection': '3d'}, \
                            figsize=pltP['figsize'])
    for j in range(nGridPts_dim2-1,-1,-1):
        jj = ref_points_idx[pltP['slc_grid_ref_dim2'][nGridPts_dim2-j-1]]
        for i in range(nGridPts_dim1):
            ii = ref_points_idx[pltP['slc_grid_ref_dim1'][i]]
            
            ax = axs[j, i]
            if fixedPlane == 'R':
                slc_ref = np.array([fixedPlaneVal, ref_points[ii], ref_points[jj]])
                slc_gt = fitEllipsoid_unscaled[idx_x, ii,jj,:,:]
                slc_gt_x = np.reshape(slc_gt[0,:], (nPhiEllipse, nThetaEllipse))
                slc_gt_y = np.reshape(slc_gt[1,:], (nPhiEllipse, nThetaEllipse))
                slc_gt_z = np.reshape(slc_gt[2,:], (nPhiEllipse, nThetaEllipse))
                slc_rgb_comp = sampledComp[idx_x, ii,jj,:,:]
            elif fixedPlane == 'G':
                slc_ref = np.array([ref_points[ii], fixedPlaneVal, ref_points[jj]])
                slc_gt = fitEllipsoid_unscaled[ii,idx_y,jj,:,:]
                slc_gt_x = np.reshape(slc_gt[0,:], (nPhiEllipse, nThetaEllipse))
                slc_gt_y = np.reshape(slc_gt[1,:], (nPhiEllipse, nThetaEllipse))
                slc_gt_z = np.reshape(slc_gt[2,:], (nPhiEllipse, nThetaEllipse))
                slc_rgb_comp = sampledComp[ii,idx_y,jj,:,:]
            elif fixedPlane == 'B':
                slc_ref = np.array([ref_points[ii], ref_points[jj], fixedPlaneVal])
                slc_gt = fitEllipsoid_unscaled[ii,jj,idx_z,:,:]
                slc_gt_x = np.reshape(slc_gt[0,:], (nPhiEllipse, nThetaEllipse))
                slc_gt_y = np.reshape(slc_gt[1,:], (nPhiEllipse, nThetaEllipse))
                slc_gt_z = np.reshape(slc_gt[2,:], (nPhiEllipse, nThetaEllipse))
                slc_rgb_comp = sampledComp[ii,jj,idx_z,:,:]
                   
            #subplot
            if pltP['visualize_ellipsoid']:
                if pltP['scaled_neg12pos1']: color_v = (slc_ref+1)/2
                else: color_v = slc_ref
                ax.plot_surface(slc_gt_x, slc_gt_y, slc_gt_z, \
                    color=color_v, edgecolor='none', alpha=0.5)
                
            if pltP['visualize_samples']:
                ax.scatter(slc_rgb_comp[0,:], slc_rgb_comp[1,:], slc_rgb_comp[2,:],\
                           s=pltP['markerSize_samples'], c= [0,0,0],
                           alpha=pltP['samples_alpha'])
                    
            ax.set_xlim(slc_ref[0]+np.array(pltP['x_bds_symmetrical']*np.array([-1,1]))); 
            ax.set_ylim(slc_ref[1]+np.array(pltP['y_bds_symmetrical']*np.array([-1,1])));  
            ax.set_zlim(slc_ref[2]+np.array(pltP['z_bds_symmetrical']*np.array([-1,1])));  
            ax.set_xlabel('');ax.set_ylabel('');ax.set_zlabel('');
            #set tick marks
            if fixedPlane == 'R':
                ax.set_xticks([]); 
            else:
                ax.set_xticks(np.round(slc_ref[0]+\
                    np.array(np.ceil(pltP['x_bds_symmetrical']*100)/100*\
                    np.array([-1,0,1])),2))
                
            if fixedPlane == 'G':
                ax.set_yticks([]); 
            else:
                ax.set_yticks(np.round(slc_ref[1]+\
                    np.array(np.ceil(pltP['y_bds_symmetrical']*100)/100*\
                    np.array([-1,0,1])),2))
                
            if fixedPlane == 'B':
                ax.set_zticks([]);
            else:
                ax.set_zticks(np.round(slc_ref[2]+\
                    np.array(np.ceil(pltP['z_bds_symmetrical']*100)/100*\
                    np.array([-1,0,1])),2))
            # Adjust viewing angle for better visualization
            if not pltP['default_viewing_angle']:
                if fixedPlane == 'R': ax.view_init(0,0)
                elif fixedPlane == 'G': ax.view_init(0,-90)
                elif fixedPlane == 'B': ax.view_init(90,-90)
            else:
                ax.view_init(30,-37.5)
            ax.grid(True)
            ax.set_aspect('equal')
    fig.suptitle(pltP['title'])
    plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05,\
                        wspace=-0.05, hspace=-0.05)
    plt.show()
    if pltP['saveFig'] and pltP['figDir'] != '':
        full_path2 = os.path.join(pltP['figDir'],pltP['figName']+'.png')
        fig.savefig(full_path2)            

#%%
def main():
    file_name = 'Isothreshold_ellipsoid_CIELABderived.pkl'
    path_str = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
            'ELPS_analysis/Simulation_DataFiles/'
    full_path = f"{path_str}{file_name}"
    os.chdir(path_str)
    
    #Here is what we do if we want to load the data
    with open(full_path, 'rb') as f:
        # Load the object from the file
        data_load = pickle.load(f)
    param, stim, results, plt_specifics = data_load[0], data_load[1], data_load[2], data_load[3]
    
    #%% Define dictionary sim
    sim = query_simCondition();
    
    # Configure the varying RGB planes based on the fixed plane selected by the user.
    sim['varying_RGBplane'] = list(range(3))
    
    # Load specific simulation data based on the selected RGB plane.
    sim['ref_points'] = stim['ref_points'][:,:,:,sim['slc_RGBplane']]
    sim['background_RGB'] = stim['background_RGB']
    
    # Define parameters for the psychometric function used in the simulation.
    sim['alpha'] = 1.1729
    sim['beta']  = 1.2286
    # Define the Weibull psychometric function.
    WeibullFunc = lambda x: (1 - 0.5*np.exp(- (x/sim['alpha'])** sim['beta']))
    # Calculate the probability of correct response given alpha and beta.
    sim['pC_given_alpha_beta'] = WeibullFunc(stim['deltaE_1JND'])
    
    #initialization
    sim['ref_Lab'] = np.full((stim['nGridPts_ref'],stim['nGridPts_ref'],\
                             stim['nGridPts_ref'],3), np.nan)
    sim['rgb_comp'] = np.full((stim['nGridPts_ref'],stim['nGridPts_ref'],\
                             stim['nGridPts_ref'],3, sim['nSims']), np.nan)
    sim['lab_comp'] = np.full(sim['rgb_comp'].shape, np.nan)
    sim['deltaE'] = np.full((stim['nGridPts_ref'],stim['nGridPts_ref'],\
                             stim['nGridPts_ref'], sim['nSims']), np.nan)
    sim['probC'] = np.full(sim['deltaE'].shape, np.nan)
    sim['resp_binary'] = np.full(sim['deltaE'].shape, np.nan)
    
    for i in range(stim['nGridPts_ref']):
        for j in range(stim['nGridPts_ref']):
            for k in range(stim['nGridPts_ref']):
                #grab the reference stimulus' RGB
                rgb_ref_ijk = sim['ref_points'][i,j,k,:]
                #convert it to Lab
                ref_Lab_ijk,_,_ = convert_rgb_lab(param['B_monitor'], \
                                                  sim['background_RGB'],\
                                                  rgb_ref_ijk)
                sim['ref_Lab'][i,j,k,:] = ref_Lab_ijk;
                
                #simulate comparison stimulus
                if sim['method_sampling'] == 'NearContour':
                    ellPara = results['ellipsoidParams'][i,j,k]
                    sim['rgb_comp'][i,j,k,:,:] = sample_rgb_comp_3DNearContour(\
                        rgb_ref_ijk, sim['nSims'], ellPara['radii'],\
                        ellPara['evecs'], sim['random_jitter'])
                    
                elif sim['method_sampling'] == 'Random':
                    sim['rgb_comp'][i,j,k,:,:] = sample_rgb_comp_random(\
                        rgb_ref_ijk, sim['varying_RGBplane'], np.nan,\
                        sim['range_randomSampling'], sim['nSims'])
                elif sim['method_sampling'] == 'Gaussian':
                    sim['rgb_comp'][i,j,k,:,:] = sample_rgb_comp_Gaussian(\
                        rgb_ref_ijk[sim['varying_RGBplane']],sim['varying_RGBplane'],\
                        np.nan, results['rgb_surface_cov'][i,j,k,:,:],\
                        sim['nSims'], sim['covMatrix_scaler'])         
                
                #simulate binary responses
                for n in range(sim['nSims']):
                    sim['lab_comp'][i,j,k,:,n], _,_ = convert_rgb_lab(param['B_monitor'], \
                        sim['background_RGB'], sim['rgb_comp'][i,j,k,:,n])
                    sim['deltaE'][i,j,k,n] = np.linalg.norm(sim['lab_comp'][i,j,k,:,n] - \
                                                          ref_Lab_ijk)
                    sim['probC'][i,j,k,n] = WeibullFunc(sim['deltaE'][i,j,k,n])
                sim['resp_binary'][i,j,k,:] = np.random.binomial(1, \
                    sim['probC'][i,j,k,:], (sim['nSims'],))
    
    #%% plotting and saving data
    for test in 'RGB':
        ttl = 'RGB plane'
        ttl_new = ttl.replace(test,'')
        plot_3D_sampledComp(stim['grid_ref'], results['fitEllipsoid_unscaled'],\
            sim['rgb_comp'], test, 0.5, plt_specifics['nPhiEllipsoid'],\
            plt_specifics['nThetaEllipsoid'], slc_grid_ref_dim1 = [0,2,4],\
            slc_grid_ref_dim2 = [0,2,4], title = ttl_new)
            
    #save to pkl
    file_name_firsthalf = 'Sims_isothreshold_ellipsoids_sim' + str(sim['nSims']) +\
        'perCond_sampling'+sim['method_sampling']
    if sim['method_sampling'] == 'NearContour':
        file_name = file_name_firsthalf+'_jitter'+str(sim['random_jitter'])+'.pkl'
    elif sim['method_sampling'] == 'Random':
        file_name = file_name_firsthalf+'_range'+str(sim['range_randomSampling'])+'.pkl'        
    elif sim['method_sampling'] == 'Gaussian':
        file_name = file_name_firsthalf+'_covMatrixScaler'+str(sim['covMatrix_scaler'])+'.pkl' 
    path_output = '/Users/fangfang/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'+\
        'ELPS_analysis/Simulation_DataFiles/'
    full_path = f"{path_output}{file_name}"
            
    # Write the list of dictionaries to a file using pickle
    with open(full_path, 'wb') as f:
        pickle.dump([sim], f)
                    
if __name__ == "__main__":
    main()    
    
                        
                    
                    
                    
                    
                    
                    
                
    