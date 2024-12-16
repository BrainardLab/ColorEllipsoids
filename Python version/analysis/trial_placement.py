#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 15:09:10 2024

@author: fangfang
"""
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

class TrialPlacementWithoutAdaptiveSampling:
    def __init__(self, gt_CIE = None, skip_sim = False):
        if skip_sim is False:
            self.sim = self._query_simCondition()
        if gt_CIE is not None:
            self.gt_CIE_param = gt_CIE[0]
            self.gt_CIE_stim = gt_CIE[1]
            self.gt_CIE_results = gt_CIE[2]
            self._extract_ref_points()
            
    def _query_simCondition(self, default_fixed_plane = 'R', 
                            default_method = 'NearContour',
                            default_jitter = 0.1, 
                            default_ub = 0.025, 
                            default_trialNum = 80):
        """
        Queries and sets up simulation conditions based on user input, with default values provided.
        
        Parameters:
        - default_fixed_plane (str): Default RGB plane to fix during the simulation.
        - default_method (str): Default sampling method.
        - default_jitter (float): Default jitter variability.
        - default_ub (float): Default upper bound for random sampling.
        - default_trialNum (int): Default number of simulation trials.
        
        Returns:
        - sim (dict): A dictionary containing the simulation parameters.
        """

        #initialize
        random_jitter = None; range_randomSampling = None;
        
        # QUESTION 1: Ask the user which RGB plane to fix during the simulation
        slc_RGBplane_input = self.get_input(
            'Which plane would you like to fix (R/G/B/[]); default: R. '+\
                'If you are simulating ellipsoids, please enter []: ', default_fixed_plane)

        # Validate the user input for the RGB plane. If valid, store the index; 
        #otherwise, use the default value.
        RGBplane = 'RGB'
        if len(slc_RGBplane_input) == 1 and slc_RGBplane_input in RGBplane:
            slc_RGBplane = RGBplane.find(slc_RGBplane_input)
        elif len(slc_RGBplane_input) == 2:
            slc_RGBplane = None
        else:
            slc_RGBplane = RGBplane.find(default_fixed_plane)
        
        # QUESTION 2: Ask the user to choose the sampling method
        method_sampling = self.get_input(
            'Which sampling method (NearContour/Random; default: NearContour): ',
            default_method)
        
        # Depending on the chosen sampling method, ask for relevant parameters.
        if method_sampling == 'NearContour':
            # QUESTION 3: For 'NearContour', ask for the jitter variability
            random_jitter = self.get_input(
                'Enter variability of random jitter (default: 0.1): ',
                default_jitter, float)
        elif method_sampling == 'Random':
            # QUESTION 3: For 'Random', ask for the upper bound of the square range
            square_ub = self.get_input(
                'Enter the upper bound of the square (default: 0.025): ',
                default_ub, float)
            range_randomSampling = [-square_ub, square_ub]
        else:
            # Fallback to default sampling method if invalid input is provided
            method_sampling = default_method
            random_jitter = self.get_input(
                'Enter variability of random jitter (default: 0.1): ',
                default_jitter, float)
            
        # QUESTION 4: Ask how many simulation trials
        nSims = self.get_input(
            'How many simulation trials per condition (default: 80): ',
            default_trialNum, int)
        
        #Identify the fixed RGB dimension by excluding the varying dimensions.
        varying_RGBplane = list(range(3))
        if slc_RGBplane is not None:
            varying_RGBplane.remove(slc_RGBplane)
            plane_2D_dict = {0: 'GB plane', 1: 'RB plane', 2: 'RG plane'}
            plane_2D = plane_2D_dict[slc_RGBplane]
            self.ndims = 2
        else:
            self.ndims = 3
        
        # Create a dictionary of all simulation parameters
        sim = {
            'slc_RGBplane': slc_RGBplane,
            'varying_RGBplane': varying_RGBplane,
            'method_sampling': method_sampling,
            'random_jitter': random_jitter,
            'range_randomSampling': range_randomSampling,
            'nSims': nSims,
        }
        
        if slc_RGBplane is not None: sim['plane_2D'] = plane_2D
                
        return sim
    
    def _extract_ref_points(self):
        """
        Extracts reference points and related data based on the selected RGB plane.
        """
        try: 
            stim = self.gt_CIE_stim
            if self.ndims == 2: #2D
                idx = self.sim['slc_RGBplane']
        
                # Use dynamic indexing to extract data based on the selected plane
                self.sim.update({
                    'plane_points': stim['plane_points'][idx],
                    'ref_points': stim['ref_points'][idx],
                    'background_RGB': stim['background_RGB'],
                    'slc_fixedVal': stim['fixed_RGBvec'],
                    'deltaE_1JND': stim['deltaE_1JND'],
                    'grid_ref': stim['grid_ref']
                })
            else: #3D
                self.sim.update({
                    'nGridPts_ref': stim['nGridPts_ref'],
                    'ref_points': stim['ref_points'],
                    'background_RGB': stim['background_RGB'],
                    'deltaE_1JND': stim['deltaE_1JND'],
                    'grid_ref': stim['grid_ref']
                })
        except KeyError as e:
            print(f"Error: Missing expected data in gt_CIE_stim - {e}")
        except IndexError as e:
            print(f"Error: Indexing issue with RGB plane - {e}")
                
    def _initialize(self):
        """
        Initializes simulation arrays to hold computed data such as RGB comparisons, 
        Lab values, deltaE values, probability of correct response, and binary responses.
        """
        try:
            nGridPts_ref = self.gt_CIE_stim['nGridPts_ref']
            nSims = self.sim['nSims']
            if self.ndims == 2:
                base_shape = (nGridPts_ref, nGridPts_ref)
        
            elif self.ndims == 3:
                base_shape = (nGridPts_ref, nGridPts_ref, nGridPts_ref)
            # Allocate memory for the simulation arrays
            self.sim.update({
                'rgb_comp': np.full(base_shape + (3, nSims), np.nan),
                'lab_comp': np.full(base_shape + (3, nSims), np.nan),
                'deltaE': np.full(base_shape + (nSims,), np.nan),
                'probC': np.full(base_shape + (nSims,), np.nan),
                'resp_binary': np.full(base_shape + (nSims,), np.nan)
            })
        except KeyError as e:
            print(f"Error: Missing expected data in gt_CIE_stim or sim - {e}")
        except Exception as e:
            print(f"Unexpected error during initialization - {e}")
    
    def _generate_comparison_stimuli(self, rgb_ref, ref_idx):
        """
        Generate comparison stimuli based on the selected sampling method.
        
        Parameters:
        - rgb_ref: Reference RGB values for the current grid point.
        - i, j: Grid indices.
        
        Returns:
        - rgb_comp_temp: The generated comparison stimuli.
        """
        
        if self.ndims == 2:
            varying_plane_idx = self.sim['varying_RGBplane']
            if self.sim['method_sampling'] == 'NearContour':
                # Use ellipsoidal parameters to generate comparison stimuli
                ellPara = self.gt_CIE_results['ellParams'][self.sim['slc_RGBplane']][*ref_idx]
                rgb_comp_temp, _, _, _ = self.sample_rgb_comp_2DNearContour(rgb_ref[varying_plane_idx],
                                                                            ellPara)                
            elif self.sim['method_sampling'] == 'Random':
                # Generate comparison stimuli within a specified square range
                rgb_comp_temp = self.sample_rgb_comp_random(rgb_ref[varying_plane_idx])
        else:
            if self.sim['method_sampling'] == 'NearContour':
                # Use ellipsoidal parameters to generate comparison stimuli
                ellPara = self.gt_CIE_results['ellipsoidParams'][*ref_idx]
                rgb_comp_temp = self.sample_rgb_comp_3DNearContour(rgb_ref, ellPara)      
            elif self.sim['method_sampling'] == 'Random':
                print('Some day, remember to update sample_rgb_comp_random so that it generalizes to 3D.')
                # Generate comparison stimuli within a specified square range
                rgb_comp_temp = self.sample_rgb_comp_random(rgb_ref)
        return rgb_comp_temp
    
    def setup_WeibullFunc(self, alpha=1.1729, beta=1.2286, guessing_rate=1/3):
        """
        Sets up the parameters for the Weibull psychometric function and calculates
        the probability of correct response for a given deltaE value.
    
        Parameters:
        - alpha (float): Scale parameter of the Weibull function, controlling the threshold.
        - beta (float): Shape parameter, controlling the slope.
        - guessing_rate (float): The probability of a correct guess by chance.
        """
        # Validate input parameters
        if alpha <= 0:
            raise ValueError("Alpha must be positive.")
        if beta <= 0:
            raise ValueError("Beta must be positive.")
        if not (0 <= guessing_rate <= 1):
            raise ValueError("Guessing rate must be between 0 and 1.")
    
        # Define parameters for the psychometric function used in the simulation.
        self.sim['alpha'] = alpha
        self.sim['beta'] = beta
        self.sim['guessing_rate'] = guessing_rate
    
        # Calculate the probability of correct response given alpha and beta.
        self.sim['pC_given_alpha_beta'] = self.WeibullFunc(self.sim['deltaE_1JND'],
                                                           alpha, beta, guessing_rate)
    
    def run_sim_1ref(self, sim_CIELab, rgb_ref, rgb_comp):
        #initialize
        lab_comp = np.full((3,self.sim['nSims']), np.nan)
        deltaE = np.full((self.sim['nSims']), np.nan)
        probC = np.full((self.sim['nSims']), np.nan)
        resp_binary = np.full((self.sim['nSims']), np.nan)
        
        # Convert the reference RGB values to Lab color space.
        ref_Lab, _, _ = sim_CIELab.convert_rgb_lab(rgb_ref)
            
        # For each simulation, calculate color difference, probability of 
        #correct identification, and simulate binary responses based on the 
        #probability.
        for n in range(self.sim['nSims']):
            # Convert the comparison RGB values to Lab color space.
            lab_comp[:,n], _, _ =  sim_CIELab.convert_rgb_lab(rgb_comp[:,n])
            # Calculate the color difference (deltaE) between the 
            #comparison and reference stimuli.
            deltaE[n] =  np.linalg.norm(lab_comp[:,n] - ref_Lab)
            # Calculate the probability of correct identification using the 
            #Weibull function.
            probC[n] = self.WeibullFunc(deltaE[n],
                                        self.sim['alpha'], 
                                        self.sim['beta'], 
                                        self.sim['guessing_rate'])
            # Simulate binary responses (0 or 1) based on the calculated probabilities.
            resp_binary[n] = np.random.binomial(1, probC[n])
            
        return lab_comp, deltaE, probC, resp_binary

    def run_sim(self, sim_CIELab, random_seed = None):
        """
        Runs the simulation to generate comparison stimuli, calculate color differences,
        determine the probability of correct identification, and simulate binary responses.
        This function can be used for both 2D and 3D cases.
        
        Parameters:
        ----------
        sim_CIELab : object
            An object that handles conversions between RGB and Lab color spaces.
        random_seed : int, optional
            Seed for the random number generator to ensure reproducibility. 
            If None, a random seed will be generated.
        """

        self._initialize()
        # Set the random seed if provided, otherwise generate a random seed
        if random_seed is not None:
            np.random.seed(random_seed)
        else:
            random_seed = np.random.randint(0, 2**32 - 1)
            np.random.seed(random_seed)
            
        # Store the random seed in the sim dictionary for reproducibility
        self.sim['random_seed'] = random_seed
    
        if self.ndims == 2:
            # Iterate over the grid points of the reference stimulus.
            for i in range(self.gt_CIE_stim['nGridPts_ref']):
                for j in range(self.gt_CIE_stim['nGridPts_ref']):
                    # Extract the reference stimulus' RGB values for the current grid point.
                    rgb_ref_ij = self.sim['ref_points'][:,i,j]
                    # Convert the reference RGB values to Lab color space.
                    ref_Lab_ij, _, _ = sim_CIELab.convert_rgb_lab(rgb_ref_ij)
                    
                    # Generate the comparison stimulus based on the sampling method.
                    rgb_comp_temp = self._generate_comparison_stimuli(rgb_ref_ij, [i,j])
                            
                    #RGB values can't exceed 1 and go below 0
                    self.sim['rgb_comp'][i,j] = np.clip(rgb_comp_temp, 0, 1)
                        
                    # For each simulation, calculate color difference, probability of 
                    #correct identification, and simulate binary responses based on the 
                    #probability.
                    self.sim['lab_comp'][i,j], self.sim['deltaE'][i,j],\
                    self.sim['probC'][i,j], self.sim['resp_binary'][i,j] = \
                        self.run_sim_1ref(sim_CIELab, 
                                          rgb_ref_ij, 
                                          self.sim['rgb_comp'][i,j])
        else:
            # Generalize the code above for 3d case
            for i in range(self.gt_CIE_stim['nGridPts_ref']):
                for j in range(self.gt_CIE_stim['nGridPts_ref']):
                    for k in range(self.gt_CIE_stim['nGridPts_ref']):
                        rgb_ref_ijk = self.sim['ref_points'][i,j,k]
                        ref_Lab_ijk, _, _ = sim_CIELab.convert_rgb_lab(rgb_ref_ijk)
                        rgb_comp_temp = self._generate_comparison_stimuli(rgb_ref_ijk, [i, j, k])
                        self.sim['rgb_comp'][i,j,k] = np.clip(rgb_comp_temp, 0, 1)
                        self.sim['lab_comp'][i,j,k], self.sim['deltaE'][i,j,k],\
                        self.sim['probC'][i,j,k], self.sim['resp_binary'][i,j,k] = \
                            self.run_sim_1ref(sim_CIELab, 
                                              rgb_ref_ijk, 
                                              self.sim['rgb_comp'][i,j,k])
                    
    #%%
    def sample_rgb_comp_2DNearContour(self, rgb_ref, paramEllipse, random_seed = None):
        """
        Samples RGB compositions near an isothreshold ellipsoidal contour.
        This function generates simulated RGB compositions based on a reference
        RGB value and parameters defining an ellipsoidal contour. The function
        is designed to vary two of the RGB dimensions while keeping the third fixed,
        simulating points near the contour of an ellipsoid in RGB color space.
    
        Parameters
        ----------
        - rgb_ref (array): The reference RGB value around which to simulate new 
            points. This RGB value defines the center of the ellipsoidal contour 
            in RGB space. This array only includes RGB of varying dimensions
        - paramEllipse (array): Parameters defining the ellipsoid contour. 
            Includes the center coordinates in the varying dimensions, the lengths 
            of the semi-axes of the ellipsoid in the plane of variation, and the 
            rotation angle of the ellipsoid in degrees.
            Expected format: [xc, yc, semi_axis1_length, semi_axis2_length, rotation_angle].
    
        Returns
        -------
        - rgb_comp_sim (array): A 3xN array of simulated RGB compositions, 
            where N is the number of simulations (`nSims`). Each column represents 
            an RGB composition near the specified ellipsoidal contour in RGB color 
            space. The row order corresponds to R, G, and B dimensions, respectively.
    
    
        """
    
        # Set the random seed if provided, otherwise generate a random seed
        if random_seed is not None:
            np.random.seed(random_seed)
            
        #Initialize the output matrix with nans
        rgb_comp_sim = np.full((3, self.sim['nSims']), np.nan)
        
        #Generate random angles to simulate points around the ellipse.
        randTheta = np.random.rand(1, self.sim['nSims']) * 2 * np.pi
        
        #calculate x and y coordinates with added jitter
        randx_noNoise = np.cos(randTheta)
        randy_noNoise = np.sin(randTheta)
        randx = randx_noNoise + np.random.randn(1, self.sim['nSims']) *\
            self.sim['random_jitter']
        randy = randy_noNoise + np.random.randn(1, self.sim['nSims']) *\
            self.sim['random_jitter']
        
        #adjust coordinates based on the ellipsoid's semi-axis lengths
        randx_stretched, randy_stretched = self.stretch_unit_circle(\
            randx, randy, paramEllipse[2], paramEllipse[3])
        
        #calculate the varying RGB dimensions, applying rotation and translation 
        #based on the reference RGB values and ellipsoid parameters
        rgb_comp_sim[self.sim['varying_RGBplane'][0],:] = \
            randx_stretched * np.cos(np.deg2rad(paramEllipse[-1])) - \
            randy_stretched * np.sin(np.deg2rad(paramEllipse[-1])) + rgb_ref[0]
        rgb_comp_sim[self.sim['varying_RGBplane'][1],:] = \
            randx_stretched * np.sin(np.deg2rad(paramEllipse[-1])) + \
            randy_stretched * np.cos(np.deg2rad(paramEllipse[-1])) + rgb_ref[1]
            
        #set the fixed RGB dimension to the specificed fixed value for all simulations
        rgb_comp_sim[self.sim['slc_RGBplane'],:] = self.sim['slc_fixedVal'];
        
        #stuff other than rgb_comp_sim are returned because we may want to plot
        #the transformation from unit circle to simulated data
        return rgb_comp_sim, np.vstack((randx_stretched, randy_stretched)),\
            np.vstack((randx, randy)), np.vstack((randx_noNoise, randy_noNoise))
    
    def sample_rgb_comp_random(self, rgb_ref):
        """
        Generates random RGB compositions within a specified square range in the
        RGB color space. Two of the RGB dimensions are allowed to vary within the 
        specified range, while the third dimension is fixed at a specified value.
    
        Parameters
        ----------
        - rgb_ref (array): The reference RGB value which serves as the starting 
            point for the simulation. The varying components will be adjusted 
            relative to this reference.
            
        Returns
        ----------
        - rgb_comp_sim (array): A 3xN array of simulated RGB compositions, 
             where N is the number of simulations (`nSims`). Each column represents 
             an RGB composition near the specified ellipsoidal contour in RGB color 
             space. The row order corresponds to R, G, and B dimensions, respectively.
    
        """
        
        box_range = self.sim['range_randomSampling']
        rgb_comp_sim = np.random.rand(3, self.sim['nSims']) *\
            (box_range[1] - box_range[0]) + box_range[0]
            
        if self.sim['slc_RGBplane'] in list(range(3)):
            rgb_comp_sim[self.sim['slc_RGBplane'],:] = self.sim['slc_fixedVal']
        
        rgb_comp_sim[self.sim['varying_RGBplane'],:] = \
            rgb_comp_sim[self.sim['varying_RGBplane'],:] + \
            rgb_ref.reshape((len(self.sim['varying_RGBplane']),1))
            
        return rgb_comp_sim

    def sample_rgb_comp_3DNearContour(self, rgb_ref, paramEllipsoid, 
                                      random_seed = None, uniform_inv_phi = True):
        """
        Simulates RGB components near the surface of an ellipsoid contour. This can 
        be used for generating test points in color space around a reference color.
        
        Parameters:
        - rgb_ref: The reference RGB stimulus.
        - radii: The radii of the ellipsoid along its principal axes.
        - eigenVec: The eigenvectors defining the orientation of the ellipsoid.
        - jitter: The standard deviation of the Gaussian noise added to simulate 
            points near the surface.
        
        Returns:
        - rgb_comp_sim: A 3xN matrix containing the simulated RGB components.
        """
        radii, eigenVec = paramEllipsoid['radii'], paramEllipsoid['evecs']
        
        # Set the random seed if provided, otherwise generate a random seed
        if random_seed is not None:
            np.random.seed(random_seed)
        
        #Uniformly distributed angles between 0 and 2*pi
        randtheta = np.random.rand(1, self.sim['nSims']) * 2 * np.pi
        
        #If you were to sample theta uniformly, you'd place too many points near 
        #the poles and too few points near the equator, because the surface area 
        #decreases near the poles. To correct for this, we uniformly sample 
        #cos(theta). This ensures that points are spaced evenly across the sphere's
        #surface because the cosine of theta (which ranges from -1 to 1) accounts 
        #for the different sizes of latitude bands as you move from pole to pole.
        if uniform_inv_phi:
            #Uniformly sampled from [-1, 1] ensures uniform distribution along
            #the z-axis
            randphi_temp = np.random.uniform(-1, 1, self.sim['nSims'])    # cos(theta) for polar angle
    
            # Converted from costheta using the inverse cosine function to get the angle.
            randphi = np.arccos(randphi_temp)
        else:
            #Uniformly distributed angles between 0 and pi
            randphi = np.random.rand(1, self.sim['nSims']) * np.pi
        
        #Generate random points on the surface of a unit sphere by converting
        # spherical coordinates to Cartesian coordinates, then add Gaussian noise
        # (jitter) to each coordinate to simulate points near the surface.
        randx = np.sin(randphi) * np.cos(randtheta) + \
            np.random.randn(1, self.sim['nSims']) * self.sim['random_jitter']
        randy = np.sin(randphi) * np.sin(randtheta) + \
            np.random.randn(1, self.sim['nSims']) * self.sim['random_jitter']
        randz = np.cos(randphi) + np.random.rand(1, self.sim['nSims']) * self.sim['random_jitter']
        
        #Stretch the random points by the ellipsoid's semi-axes lengths to fit
        # the ellipsoid's shape. This effectively scales the unit sphere points
        # to the size of the ellipsoid along each principal axis.
        randx_stretched, randy_stretched, randz_stretched =\
            self.stretch_unit_circle(randx, randy, radii[0], radii[1], 
                                     z = randz, ax_length_z = radii[2])
        
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

    @staticmethod
    def stretch_unit_circle(x, y, ax_length_x, ax_length_y, z = None, ax_length_z = None):
        #adjust coordinates based on the ellipsoid's semi-axis lengths
        x_stretched = x * ax_length_x
        y_stretched = y * ax_length_y
        if z is None or ax_length_z is None:
            return x_stretched, y_stretched
        else:
            z_stretched = z * ax_length_z
            return  x_stretched, y_stretched, z_stretched
    
    @staticmethod
    # Helper function to get user input with a default value and type conversion
    def get_input(prompt, default, input_type=str):
        user_input = input(prompt).strip()
        return input_type(user_input) if user_input else default

    @staticmethod
    def WeibullFunc(x, alpha, beta, guessing_rate):
        """
        Computes the Weibull psychometric function, giving the probability of a 
        correct response.
        
        Parameters:
        - x (float or array-like): The stimulus intensity (e.g., deltaE).
        - alpha (float): Scale parameter of the Weibull function.
        - beta (float): Shape parameter of the Weibull function.
        - guessing_rate (float): The probability of a correct guess by chance.
        
        Returns:
        - pCorrect (float or array-like): The probability of a correct response.
        
        """
        pCorrect = (1 - (1-guessing_rate)*np.exp(- (x/alpha)** beta))
        return pCorrect
    
    @staticmethod
    def compute_radii_scaler_to_reach_targetPC(pC_target, lb = 2, ub = 3,
                                               nsteps = 100, nz = int(1e6),
                                               visualize = False):
        """
        Computes the optimal scaling factor (radii) to reach a target probability of 
        correct classification (pC_target) between three points in a 2D bivariate 
        Gaussian distribution. 
        
        Parameters:
        - pC_target: Target probability of correct classification.
        - lb: Lower bound of the scaler (radius) range to search over.
        - ub: Upper bound of the scaler (radius) range to search over.
        - nsteps: Number of steps for the scaler search range.
        - nz: Number of samples to generate for the distribution.
        
        Returns:
        - opt_scaler: The scaler that yields the probability closest to the target pC_target.
        - probC[min_idx]: The probability of correct classification at the optimal scaler.
        """

        # Define the mean vector and covariance matrix for the bivariate Gaussian distribution
        mean = [0, 0]   
        cov = np.eye(2)  
    
        # Generate the scaling factors (radii) to be tested between lb and ub
        z2_scaler = np.linspace(lb, ub, nsteps)
        
        # Initialize an array to store the probability of correct classification for each scaler
        probC = np.full((nsteps), np.nan)
    
        # Loop through each scaling factor to compute the probability of correct classification
        for idx, scaler in enumerate(z2_scaler):            
            # Draw nz samples from the bivariate Gaussian distribution for z0 and z1 
            #(two independent points)
            z0 = np.random.multivariate_normal(mean, cov, nz)
            z1 = np.random.multivariate_normal(mean, cov, nz)
    
            # For z2, apply a center offset based on the current scaler value
            z2_center = np.array([0, scaler])
            z2 = np.random.multivariate_normal(mean, cov, nz) + z2_center[np.newaxis, :]
    
            # Compute pairwise differences between points z0, z1, and z2
            r01 = z0 - z1   
            r02 = z0 - z2   
            r12 = z1 - z2   
    
            # Compute squared Mahalanobis distances (a measure of the distance between points 
            # in Gaussian space) Mahalanobis distance accounts for the covariance of the distribution.
            z0_to_z1 = jnp.sum(r01 * jnp.linalg.solve(cov, r01.T).T, axis=1)
            z0_to_z2 = jnp.sum(r02 * jnp.linalg.solve(cov, r02.T).T, axis=1)
            z1_to_z2 = jnp.sum(r12 * jnp.linalg.solve(cov, r12.T).T, axis=1)
    
            # Compute the difference between z0-to-z1 distance and the minimum of z0-to-z2 and 
            # z1-to-z2 distances
            zdiff = z0_to_z1 - jnp.minimum(z0_to_z2, z1_to_z2)
    
            # Calculate the probability of correct classification as the fraction where zdiff < 0
            probC[idx] = np.sum(zdiff < 0) / nz
    
        # Plot the computed probabilities as a function of the scaling factors
        if visualize: plt.plot(z2_scaler, probC)
        
        # Find the index of the scaling factor closest to the target probability (pC_target)
        min_idx = np.argmin(np.abs(pC_target - probC))
        
        # Retrieve the optimal scaling factor and the corresponding probability
        opt_scaler = z2_scaler[min_idx]
        return opt_scaler, probC[min_idx], probC
    
    
    
    
    
    
    
    