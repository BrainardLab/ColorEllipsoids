#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 15:09:10 2024

@author: fangfang
"""
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import sys
sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids")
from analysis.color_thres import color_thresholds
sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/Python version")
from analysis.MOCS_thresholds import sim_MOCS_trials
from analysis.ellipses_tools import stretch_unit_circle, rotate_relocate_stretched_ellipse

#%%
class NonAdaptiveTrialPlacement(ABC):
    """ 
    This class has common functionalities that can be shared by subclasses.
    The goal is to place trials near the thresholds given that we know what 
    the thresholds are. The ground truth thresholds can vary: 
        (1) CIELAB derived
        (2) Wishart fits to CIELAB derived thresholds
    The sampling method can vary:
        (a) fixed reference stimuli
        (b) Sobol-generated reference stimuli
    The stimulus space can vary:
        (i) GB, RG, RB planar slices or RGB cube
        (ii) Isoluminant plane
        
    Depending on the combinations, we choose different subclasses:
        (1-a-i/ii) TrialPlacement_RGB_gridRef_gtCIE
        (1-b-ii) TrialPlacement_Isoluminant_sobolRef_gtCIE
        ...
    """
    def __init__(self):
        #initialize the dictionary
        self.sim = {}
        
    # Helper function to get user input with a default value and type conversion
    @staticmethod
    def get_input(prompt, default, input_type=str):
        """
        Prompts the user for input, with an option to specify a default value and 
        automatically convert the input type.
        """
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
     
    #%%
    def _parse_plane_selection(self, input_plane):
        """
        Identifies the selected plane, its varying dimensions, and its corresponding name 
        based on the input. Also sets relevant flags for later transformations.

        Args:
            input_plane (str): The selected plane ('R', 'G', 'B', 'lum', or '[]' for 3D case).

        Returns:
            tuple: (selected plane index, list of varying dimensions, name of the 2D plane)

        Notes on `plane_mappings`:
            - The isoluminant plane ('lum') is treated as the RG plane for convenience, 
              with the selected index set to 2 and the varying indices as [0, 1].
            - A third row filled with ones is later inserted to facilitate transformation 
              between W space and RGB space.
            - When working with the isoluminant plane, ground-truth thresholds derived 
              from CIELAB are already in Wishart space (bounded between -1 and 1). 
              A flag is set to prevent unnecessary double conversion in subsequent processing.
        """
        plane_mappings = {
            'R': (0, [1, 2], 'GB plane', False, 2),  # Varying G and B
            'G': (1, [0, 2], 'RB plane', False, 2),  # Varying R and B
            'B': (2, [0, 1], 'RG plane', False, 2),  # Varying R and G
            'lum': (2, [0, 1], 'Isoluminant plane', True, 2)  # Treated as RG plane but with a Wishart space flag
        }
    
        if input_plane == '[]':  # Case for 3D ellipsoid
            self.ndims = 3
            self.flag_W_space = False
            return None, None, None  
        
        if input_plane in plane_mappings:
            slc_plane, varying_RGBplane, plane_2D, self.flag_W_space, self.ndims = \
                plane_mappings[input_plane]
            return slc_plane, varying_RGBplane, plane_2D
    
    def setup_WeibullFunc(self, alpha, beta, guessing_rate, deltaE_1JND):
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
        self.sim['deltaE_1JND'] = deltaE_1JND
    
        # Calculate the probability of correct response given alpha and beta.
        self.sim['pC_given_alpha_beta'] = self.WeibullFunc(deltaE_1JND,
                                                           alpha, beta, guessing_rate)   
     
    #%% mandatory changes
    @abstractmethod
    def _initialize(self):
        """
        Initializes necessary simulation fields within `self.sim`.
    
        This method should:
        - Pre-allocate arrays for storing simulated comparison stimuli, 
          probability values, and response data.
        - Set up any additional parameters required for the simulation.
    
        Notes:
        ------
        - This method must be implemented in subclasses.
        """
        pass
    
    @abstractmethod    
    def _validate_sampled_comp(self):
        """
        Ensures that the sampled comparison stimuli remain within the valid boundary.
    
        This method should:
        - Clip values to prevent them from exceeding predefined limits.
        - Add extra rows (filler 1's) to allow transformation across spaces.
    
        Notes:
        ------
        - This method must be implemented in subclasses.
        """
        pass
    
    @abstractmethod
    def _query(self):
        """
        Prompts the user (or another interface) for specific simulation settings.
    
        This method should:
        - Gather user input regarding key simulation parameters.
        - Validate and store the inputs in `self.sim`.
    
        Notes:
        ------
        - This method must be implemented in subclasses.
        """
        pass
    
    @abstractmethod
    def run_sim_1ref(self):
        """
        Simulates comparison stimuli for a single reference stimulus.
    
        This method should:
        - Generate a set of test stimuli based on a given reference.
        - Compute color differences and probabilities of correct responses.
        - Store the simulation results for the reference stimulus.
    
        Notes:
        ------
        - This method must be implemented in subclasses.
        """
        pass
    
    @abstractmethod
    def run_sim(self):
        """
        Runs the full simulation by generating comparison stimuli for multiple reference points.
    
        This method should:
        - Iterate over all reference stimuli.
        - Call `run_sim_1ref()` for each reference to generate comparison trials.
        - Aggregate the results across all references.
    
        Notes:
        ------
        - This method must be implemented in subclasses.
        """
        pass
    
#%%
class TrialPlacement_Isoluminant_sobolRef_gtCIE(NonAdaptiveTrialPlacement):
    def __init__(self, M_2DWToRGB, M_RGBTo2DW, colordiff_alg = 'CIE2000',
                 random_seed=None):
        super().__init__()
        
        if colordiff_alg not in ['CIE1976', 'CIE1994', 'CIE2000']:
            raise ValueError("The method can only be 'CIE1976' or 'CIE1994' or 'CIE2000'.")
        self.M_2DWToRGB = M_2DWToRGB
        self.M_RGBTo2DW = M_RGBTo2DW
        self.colordiff_alg = colordiff_alg
        self.random_seed = random_seed
        self._query()
        self._initialize()
        
    def _initialize(self):
        """
        Initializes simulation arrays to store computed data, including:
        - RGB comparison values
        - Probability of correct response
        - Binary response (correct/incorrect)
        - (Optional) Lab values and deltaE differences
            - They are only applicable when the ground truth is derived by CIELAB
    
        The arrays are preallocated with NaNs for efficient storage and processing.
        """
                
        # Common arrays for all cases
        self.sim.update({
            'rgb_comp': np.full((3, self.sim['nSims']), np.nan),
            'probC': np.full((self.sim['nSims'],), np.nan),
            'resp_binary': np.full((self.sim['nSims'],), np.nan),
            'deltaE': np.full((self.sim['nSims'],), np.nan)
        })
         
    def _validate_sampled_comp(self, rgb_comp):
        return np.clip(rgb_comp, -1, 1)
    
    def _query(self, default_jitter = 0.1, default_trialNum = 6000):
        """
        Queries and sets up simulation conditions based on user input, with default values provided.
        
        Parameters:
        - default_fixed_plane (str): Default RGB plane to fix during the simulation.
        - default_jitter (float): Default jitter variability.
        - default_trialNum (int): Default number of simulation trials.
        
        Returns:
        - sim (dict): A dictionary containing the simulation parameters.
        """
        
        slc_RGBplane, varying_RGBplane, plane_2D = self._parse_plane_selection('lum')
        
        # QUESTION 2: Ask the user to choose the sampling method
        # Ask for jitter variability
        random_jitter = self.get_input(
            f'Enter variability of random jitter (default: {default_jitter}): ',
            default_jitter, float
        )
            
        # QUESTION 3: Ask how many simulation trials
        nSims = self.get_input(
            f'How many simulation trials in total (default: {default_trialNum}): ',
            default_trialNum, int)
        
        # Create a dictionary of all simulation parameters
        self.sim = {
            'plane_2D': plane_2D,
            'slc_RGBplane': slc_RGBplane,
            'varying_RGBplane': varying_RGBplane,
            'method_sampling': 'NearContour',  # Currently redundant, but kept for consistency 
                                               # since previous versions included other methods 
                                               # (e.g., random sampling)
            'random_jitter': random_jitter,
            'nSims': nSims,
        }
                        
    def _sobol_generate_ref(self, lb, ub):
        sobolRef_cat = sim_MOCS_trials.sample_sobol(self.sim['nSims'], lb, ub,
                                                    force_center=False, 
                                                    seed= self.random_seed)
        
        # in W space (append a row of 1's to make the following transformation easier)
        # shape: 3 x self.sim['nSims']
        self.sim['ref_points'] = np.vstack((sobolRef_cat[:,:2].T, np.ones((1,self.sim['nSims']))))
        
        sobolAngle = sobolRef_cat[:,-1]
        
        self.sim['vecDir'] = np.column_stack((np.cos(np.radians(sobolAngle)), 
                                    np.sin(np.radians(sobolAngle))))
            
    def _add_jitter(self, ref, comp):
        # in W space, apply independent Gaussian noise to each coordinate
        opt_vecLen_W = np.linalg.norm(comp - ref)
        jitter = np.random.randn(1,2)*opt_vecLen_W*self.sim['random_jitter']
        
        return jitter
        
    def run_sim_1ref(self, sim_CIELab, rgb_ref, rgb_comp):        
        # Convert the reference RGB values to Lab color space.
        actual_rgb_ref = self.M_2DWToRGB @ rgb_ref
        actual_rgb_comp = self.M_2DWToRGB @ rgb_comp
            
        deltaE = sim_CIELab.compute_deltaE(actual_rgb_ref, None,None,
                                    comp_RGB=actual_rgb_comp, 
                                    method=self.colordiff_alg)
        
        # Calculate the probability of correct identification using the 
        #Weibull function.
        probC = self.WeibullFunc(deltaE,
                                 self.sim['alpha'], 
                                 self.sim['beta'], 
                                 self.sim['guessing_rate'])
        # Simulate binary responses (0 or 1) based on the calculated probabilities.
        randNum = np.random.rand() 
        resp_binary = (randNum < probC).astype(int)
            
        return deltaE, probC, resp_binary

    def run_sim(self, sim_CIELab, sobol_bounds_lb, sobol_bounds_ub):
        """
        Runs the simulation to generate comparison stimuli, calculate color differences,
        determine the probability of correct identification, and simulate binary responses.
        This function can be used for both 2D and 3D cases.
        
        Parameters:
        ----------
        sim_CIELab : object, optional
            An object that handles conversions between RGB and Lab color spaces.
            Required if `self.flag_Wishart` is False. Ignored if `self.flag_Wishart` is True.
        random_seed : int, optional
            Seed for the random number generator to ensure reproducibility. 
            If None, a random seed will be generated.
            
        colordiff_alg (str): The method for calculating deltaE. Options are:
            - 'CIE1976': DeltaE using the CIE1976 method (Euclidean distance in CIELab).
            - 'CIE1994': DeltaE using the CIE1994 method (accounts for perceptual non-uniformity).
            - 'CIE2000': DeltaE using the CIE2000 method (more advanced perceptual uniformity).
    
        """
        # Set the random seed if provided, otherwise generate a random seed
        np.random.seed(self.random_seed)
        
        self._sobol_generate_ref(sobol_bounds_lb, sobol_bounds_ub)
    
        # Iterate over the grid points of the reference stimulus.
        for i in range(self.sim['nSims']):           
            ref_i = self.sim['ref_points'][:,i]
            # extract the ground truth
            _, _, _, threshold_point_W = sim_CIELab.find_threshold_point_on_isoluminant_plane(\
                                        ref_i, 
                                        self.sim['vecDir'][i], 
                                        self.M_RGBTo2DW,
                                        self.M_2DWToRGB,
                                        self.sim['deltaE_1JND'],
                                        color_diff_algorithm = self.colordiff_alg)
            
            jitter_i = self._add_jitter(ref_i[:2], threshold_point_W)
            self.sim['rgb_comp'][:,i] = np.append(threshold_point_W + jitter_i, 1)
            
            # For each simulation, calculate color difference, probability of 
            #correct identification, and simulate binary responses based on the 
            #probability.
            self.sim['deltaE'][i], self.sim['probC'][i], self.sim['resp_binary'][i] = \
                self.run_sim_1ref(sim_CIELab, 
                                  self.sim['ref_points'][:,i], 
                                  self.sim['rgb_comp'][:,i])
                
#%%
class TrialPlacement_RGB_gridRef_gtCIE(NonAdaptiveTrialPlacement):
    def __init__(self, gt_CIE,  colordiff_alg = 'CIE2000', random_seed=None, 
                 M_RGBTo2DW = None, M_2DWToRGB = None):
        """
        
        colordiff_alg (str): The method for calculating deltaE. Options are:
            - 'CIE1976': DeltaE using the CIE1976 method (Euclidean distance in CIELab).
            - 'CIE1994': DeltaE using the CIE1994 method (accounts for perceptual non-uniformity).
            - 'CIE2000': DeltaE using the CIE2000 method (more advanced perceptual uniformity).

        """
        super().__init__()
        
        if colordiff_alg not in ['CIE1976', 'CIE1994', 'CIE2000']:
            raise ValueError("The method can only be 'CIE1976' or 'CIE1994' or 'CIE2000'.")

        self.gt_CIE_param = gt_CIE[0]    
        self.gt_CIE_stim = gt_CIE[1]
        self.gt_CIE_results = gt_CIE[2]
        self.colordiff_alg = colordiff_alg
        self.random_seed = random_seed
        self.M_RGBTo2DW = M_RGBTo2DW
        self.M_2DWToRGB = M_2DWToRGB

        self._query()
        self._extract_ref_points()
        self._initialize()
        
    def _initialize(self):
        """
        Initializes simulation arrays to store computed data, including:
        - RGB comparison values
        - Probability of correct response
        - Binary response (correct/incorrect)
        - (Optional) Lab values and deltaE differences
            - They are only applicable when the ground truth is derived by CIELAB
    
        The arrays are preallocated with NaNs for efficient storage and processing.
        """
        # Determine the base shape based on nGridPts_ref
        nGridPts_ref = self.sim['nGridPts_ref']
        base_shape = (nGridPts_ref,) * self.ndims
    
        # Common arrays for all cases
        self.sim.update({
            'rgb_comp': np.full(base_shape + (3, self.sim['nSims']), np.nan),
            'probC': np.full(base_shape + (self.sim['nSims'],), np.nan),
            'resp_binary': np.full(base_shape + (self.sim['nSims'],), np.nan),
            'deltaE': np.full(base_shape + (self.sim['nSims'],), np.nan)
        })
        
    def _validate_sampled_comp(self, rgb_comp):
        #rgb_comp[self.sim['slc_RGBplane'],:] = self.sim['slc_fixedVal'];
        if self.flag_W_space:
            return np.clip(rgb_comp, -1, 1)
        else:
            return np.clip(rgb_comp, 0, 1)
    
    def _query(self, default_fixed_plane = 'R', default_jitter = 0.3, default_trialNum = 240):
        """
        Queries and sets up simulation conditions based on user input, with default values provided.
        
        Parameters:
        - default_fixed_plane (str): Default RGB plane to fix during the simulation.
        - default_jitter (float): Default jitter variability.
        - default_trialNum (int): Default number of simulation trials.
        
        Returns:
        - sim (dict): A dictionary containing the simulation parameters.
        """
        
        # QUESTION 1: Ask which RGB plane to fix during the simulation
        slc_RGBplane_input = self.get_input(
            f'Which plane would you like to fix (lum/R/G/B/[]); default: {default_fixed_plane}. '
            'If you are simulating 3D ellipsoids, please enter []: ',
            default_fixed_plane
        )
        slc_RGBplane, varying_RGBplane, plane_2D = self._parse_plane_selection(slc_RGBplane_input)
        
        if (plane_2D == 'Isoluminant plane') and ((self.M_2DWToRGB is None) or (self.M_RGBTo2DW is None)):
            raise ValueError('Since the isoluminant plane is chosen, transformation matrices must be defined!')
        
        # QUESTION 2: Ask the user to choose the sampling method
        # Ask for jitter variability
        random_jitter = self.get_input(
            f'Enter variability of random jitter (default: {default_jitter}): ',
            default_jitter, float
        )
            
        # QUESTION 3: Ask how many simulation trials
        nSims = self.get_input(
            f'How many simulation trials per condition (default: {default_trialNum}): ',
            default_trialNum, int)
        
        # Create a dictionary of all simulation parameters
        self.sim = {
            'plane_2D': plane_2D,
            'slc_RGBplane': slc_RGBplane,
            'varying_RGBplane': varying_RGBplane,
            'method_sampling': 'NearContour',  # Currently redundant, but kept for consistency 
                                               # since previous versions included other methods 
                                               # (e.g., random sampling)
            'random_jitter': random_jitter,
            'nSims': nSims
        }

    def _extract_ground_truth(self, ref_idx):
        """
        Extract the parameters that define the elliptical / ellipsoidal threshold contours
        
        """
        if self.ndims == 2:
            ellPara = self.gt_CIE_results['ellParams'][self.sim['slc_RGBplane']][*ref_idx]      
        else:
            # Use ellipsoidal parameters to generate comparison stimuli
            i,j,k = ref_idx
            ellPara = self.gt_CIE_results['ellipsoidParams'][i][j][k]
        return ellPara
    
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
                    'nGridPts_ref': len(stim['grid_ref']),
                    'ref_points': np.transpose(stim['ref_points'][idx], (1,2,0)), #shape: 3 x 5 x 5 -> 5 x 5 x 3
                    'background_RGB': stim['background_RGB'],
                    'slc_fixedVal': stim['fixed_RGBvec'],
                    'deltaE_1JND': stim['deltaE_1JND'],
                    'grid_ref': stim['grid_ref'] #1d grid (5 points)
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
            
    def _random_points_on_unit_circle(self):
        """
        Generates random points near the unit circle with independent jitter for x and y.
        """
        N = self.sim['nSims']
        jitter = self.sim['random_jitter']
    
        # Generate random angles and compute unit circle coordinates
        randTheta = np.random.rand(1, N) * 2 * np.pi
        randx_noNoise, randy_noNoise = np.cos(randTheta), np.sin(randTheta)
    
        # Apply independent Gaussian noise to x and y coordinates
        noise_x = np.random.randn(1, N) * jitter
        noise_y = np.random.randn(1, N) * jitter
        randx, randy = randx_noNoise + noise_x, randy_noNoise + noise_y
    
        return randx, randy, randx_noNoise, randy_noNoise
    
    def _random_points_on_unit_sphere(self):
        """
        Generates random points near the surface of a unit sphere with independent jitter 
        applied to each coordinate.
        
        """
        N = self.sim['nSims']
        jitter = self.sim['random_jitter']

        #Uniformly distributed angles between 0 and 2*pi
        randtheta = np.random.rand(1, N) * 2 * np.pi
        
        #If you were to sample phi uniformly, you'd place too many points near 
        #the poles and too few points near the equator, because the surface area 
        #decreases near the poles. To correct for this, we uniformly sample 
        #cos(phi). This ensures that points are spaced evenly across the sphere's
        #surface because the cosine of phi (which ranges from -1 to 1) accounts 
        #for the different sizes of latitude bands as you move from pole to pole.
        #Uniformly sampled from [-1, 1] ensures uniform distribution along
        #the z-axis
        randphi_temp = np.random.uniform(-1, 1, N)    # cos(theta) for polar angle
        # Converted from costheta using the inverse cosine function to get the angle.
        randphi = np.arccos(randphi_temp)
        
        # Convert spherical coordinates to Cartesian (unit sphere)
        randx_noNoise = np.sin(randphi) * np.cos(randtheta)
        randy_noNoise = np.sin(randphi) * np.sin(randtheta)
        randz_noNoise = np.cos(randphi)

        # Apply independent Gaussian noise to each coordinate
        noise_x = np.random.randn(1, N) * jitter
        noise_y = np.random.randn(1, N) * jitter
        noise_z = np.random.randn(1, N) * jitter
        randx, randy, randz = randx_noNoise + noise_x, randy_noNoise + noise_y, randz_noNoise + noise_z

        return randx, randy, randz, randx_noNoise, randy_noNoise, randz_noNoise
 
    def run_sim_1ref(self, sim_CIELab, rgb_ref, rgb_comp):
        """
        Simulates comparison stimuli near a fixed reference color.
    
        Since the reference points are fixed on a grid, this function generates 
        multiple comparison stimuli around each reference. The loop iterates 
        through `nSims` trials, computing color differences, probabilities of 
        correct identification, and binary responses.

        """
        nSims = self.sim['nSims']
        
        #initialize
        deltaE = np.full((nSims,), np.nan)
        probC = np.full((nSims,), np.nan)
        resp_binary = np.full((nSims,), np.nan)
        
        if self.flag_W_space:
            actual_rgb_ref = self.M_2DWToRGB @ rgb_ref
            actual_rgb_comp = self.M_2DWToRGB @ rgb_comp
        else:
            actual_rgb_ref = rgb_ref
            actual_rgb_comp = rgb_comp
            
        # For each simulation, calculate color difference, probability of 
        #correct identification, and simulate binary responses based on the 
        #probability.
        for n in range(nSims):
            deltaE[n] = sim_CIELab.compute_deltaE(actual_rgb_ref, None,None,
                                        comp_RGB=actual_rgb_comp[:,n], 
                                        method=self.colordiff_alg)
            
            # Calculate the probability of correct identification using the 
            #Weibull function.
            probC[n] = self.WeibullFunc(deltaE[n],
                                        self.sim['alpha'], 
                                        self.sim['beta'], 
                                        self.sim['guessing_rate'])
        # Simulate binary responses (0 or 1) based on the calculated probabilities.
        randNum = np.random.rand(nSims) 
        resp_binary = (randNum < probC).astype(int)
            
        return deltaE, probC, resp_binary

    def sample_rgb_comp_2DNearContour(self, rgb_ref, paramEllipse):
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
            
        #Initialize the output matrix with nans
        rgb_comp_sim = np.full((3, self.sim['nSims']), np.nan)
        
        #Generate random angles to simulate points around the ellipse.
        randx, randy, randx_noNoise, randy_noNoise = self._random_points_on_unit_circle()
        
        #adjust coordinates based on the ellipsoid's semi-axis lengths
        randx_stretched, randy_stretched =stretch_unit_circle(\
            randx, randy, paramEllipse[2], paramEllipse[3])
        
        #calculate the varying RGB dimensions, applying rotation and translation 
        #based on the reference RGB values and ellipsoid parameters            
        rgb_comp_sim[self.sim['varying_RGBplane'],:] = \
            rotate_relocate_stretched_ellipse(randx_stretched,
                                              randy_stretched, 
                                              paramEllipse[-1], 
                                              *rgb_ref[self.sim['varying_RGBplane']])
            
        #make sure the sampled comparison stimuli are within the desired boundaries
        rgb_comp_sim = self._validate_sampled_comp(rgb_comp_sim)
        
        #stuff other than rgb_comp_sim are returned because we may want to plot
        #the transformation from unit circle to simulated data
        return rgb_comp_sim, np.vstack((randx_stretched, randy_stretched)),\
            np.vstack((randx, randy)), np.vstack((randx_noNoise, randy_noNoise))   

    def sample_rgb_comp_3DNearContour(self, rgb_ref, paramEllipsoid):
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
        
        randx, randy, randz, randx_noNoise, randy_noNoise, randz_noNoise = \
            self._random_points_on_unit_sphere()
        
        #Stretch the random points by the ellipsoid's semi-axes lengths to fit
        # the ellipsoid's shape. This effectively scales the unit sphere points
        # to the size of the ellipsoid along each principal axis.
        randx_stretched, randy_stretched, randz_stretched =\
            stretch_unit_circle(randx, randy, radii[0], radii[1], z = randz, ax_length_z = radii[2])
        
        #Combine the stretched coordinates into a single matrix. Each column
        # represents the (x, y, z) coordinates of a point.
        xyz = np.vstack((randx_stretched, randy_stretched, randz_stretched))
        
        #Rotate and translate the simulated points to their correct positions
        #in RGB space. The rotation is defined by the ellipsoid's eigenvectors
        #(orientation), and the translation moves the ellipsoid to be centered
        #at the reference RGB value. This step aligns the ellipsoid with its
        #proper orientation and position as defined by the input parameters.
        rgb_comp_sim = eigenVec @ xyz + np.reshape(rgb_ref,(-1,1))
        
        #make sure the sampled comparison stimuli are within the desired boundaries
        rgb_comp_sim = self._validate_sampled_comp(rgb_comp_sim)
        
        #stuff other than rgb_comp_sim are returned because we may want to plot
        #the transformation from unit circle to simulated data
        return rgb_comp_sim, np.vstack((randx_stretched, randy_stretched, randz_stretched)),\
            np.vstack((randx, randy, randz)), \
            np.vstack((randx_noNoise, randy_noNoise, randz_noNoise))

    def run_sim(self, sim_CIELab):
        """
        Runs the simulation to generate comparison stimuli, calculate color differences,
        determine the probability of correct identification, and simulate binary responses.
        This function can be used for both 2D and 3D cases.
        
        Parameters:
        ----------
        sim_CIELab : object, optional
            An object that handles conversions between RGB and Lab color spaces.
            Required if `self.flag_Wishart` is False. Ignored if `self.flag_Wishart` is True.
        """
        # Set the random seed if provided, otherwise generate a random seed
        np.random.seed(self.random_seed)
    
        N = self.sim['nGridPts_ref']
        
        shape = (N,) * self.ndims
        
        for idx in np.ndindex(shape):
            #extract the reference RGB value
            rgb_ref = self.sim['ref_points'][idx]
            
            #extract ground truth parameters (ellipse or ellipsoid)
            ellPara = self._extract_ground_truth(idx)
            
            #Generate comparison stimuli based on 2D or 3D sampling
            if self.ndims == 2:
                rgb_comp, _, _, _ = self.sample_rgb_comp_2DNearContour(rgb_ref, ellPara)
            else:
                rgb_comp, _, _, _ = self.sample_rgb_comp_3DNearContour(rgb_ref, ellPara)
                
            #store comparison stimuli
            self.sim['rgb_comp'][idx] = rgb_comp
            
            #run the actual simulation
            self.sim['deltaE'][idx], self.sim['probC'][idx], self.sim['resp_binary'][idx] =\
                self.run_sim_1ref(sim_CIELab, rgb_ref, rgb_comp)
        
        # if self.ndims == 2:
        #     # Iterate over the grid points of the reference stimulus.
        #     for i in range(N):
        #         for j in range(N):
        #             # Extract the reference stimulus' RGB values for the current grid point.
        #             rgb_ref_ij = self.sim['ref_points'][:,i,j]
                    
        #             # extract the ground truth
        #             ellPara = self._extract_ground_truth([i,j])
                    
        #             #Generate the comparison stimulus based on the sampling method
        #             self.sim['rgb_comp'][i,j], _, _, _ = self.sample_rgb_comp_2DNearContour(\
        #                                                 rgb_ref_ij, ellPara)      
                        
        #             # For each simulation, calculate color difference, probability of 
        #             #correct identification, and simulate binary responses based on the 
        #             #probability.
        #             self.sim['deltaE'][i,j], self.sim['probC'][i,j], self.sim['resp_binary'][i,j] = \
        #                 self.run_sim_1ref(sim_CIELab, rgb_ref_ij,  self.sim['rgb_comp'][i,j])
                        
        # else:
        #     # Generalize the code above for 3d case
        #     # we do not use 3D wishart fits as ground truth yet!
        #     for i in range(N):
        #         for j in range(N):
        #             for k in range(N):
        #                 rgb_ref_ijk = self.sim['ref_points'][i,j,k]
        #                 ellPara = self._extract_ground_truth([i, j, k])
        #                 self.sim['rgb_comp'][i,j,k], _, _, _ = \
        #                     self.sample_rgb_comp_3DNearContour(rgb_ref_ijk, ellPara)      
                        
        #                 self.sim['deltaE'][i,j,k], self.sim['probC'][i,j,k], \
        #                     self.sim['resp_binary'][i,j,k] = \
        #                     self.run_sim_1ref(sim_CIELab, rgb_ref_ijk, 
        #                                       self.sim['rgb_comp'][i,j,k])
    
#%%
class TrialPlacementWithoutAdaptiveSampling:
    def __init__(self, gt_CIE = None, gt_Wishart = None, skip_sim = False, 
                 M_2DWToRGB = np.eye(3), M_RGBTo2DW = np.eye(3)):
        """
        This class simulates trials near the threshold contours. 
        The thresholds can be based on either CIE-derived values or Wishart model fits.
        
        Note: Initially, we sliced the GB, RG, and RB planes, but later decided 
        that it makes more sense to use the isoluminant plane. As a result, this 
        code is structured more generally than necessary, since we no longer consider 
        the GB, RG, or RB planes. However, we are keeping the generalized structure 
        for potential future flexibility.

        """
        if skip_sim is False:
            self.sim = self._query_simCondition()
        if gt_CIE is not None:
            self.flag_Wishart = False
            self.gt_CIE_param = gt_CIE[0]    
            self.gt_CIE_stim = gt_CIE[1]
            self.gt_CIE_results = gt_CIE[2]
            self._extract_ref_points()
        else:
            if gt_Wishart is not None:
                self.flag_Wishart = True
                self.gt_model_pred = gt_Wishart['model_pred_Wishart']
                self.gt_Wishart_ellParams = gt_Wishart['model_pred_Wishart'].params_ell
                self.sim['ref_points'] = color_thresholds.W_unit_to_N_unit(np.transpose(gt_Wishart['grid'], (2,0,1)))
                self.sim['nGridPts_ref'] = self.gt_model_pred.num_grid_pts
                self.sim['grid_ref'] = color_thresholds.W_unit_to_N_unit(np.unique(gt_Wishart['grid']))
        self.M_2DWToRGB = M_2DWToRGB
        self.M_RGBTo2DW = M_RGBTo2DW
        
            
    def _initialize(self):    
        nSims = self.sim['nSims']
        
        # Determine the base shape based on nGridPts_ref
        nGridPts_ref = self.sim['nGridPts_ref']
        base_shape = (nGridPts_ref,) * self.ndims
    
        # Common arrays for all cases
        self.sim.update({
            'rgb_comp': np.full(base_shape + (3, nSims), np.nan),
            'probC': np.full(base_shape + (nSims,), np.nan),
            'resp_binary': np.full(base_shape + (nSims,), np.nan)
        })

    def _extract_ground_truth(self, ref_idx):
        """
        Extract the parameters that define the elliptical / ellipsoidal threshold contours
        
        """
        ellPara = self.gt_Wishart_ellParams[ref_idx[0]][ref_idx[1]]
        #the function self.sample_rgb_comp_2DNearContour assumes the values are in the N unit
        #however, the estimated ell parameters are in the W unit, so we need to divide 
        #the axis lengths by 2
        ellPara[2] = ellPara[2]/2 #major semi-axis length
        ellPara[3] = ellPara[3]/2 #minor semi-axis length    

        return ellPara
    
    def _validate_sampled_comp(self, rgb_comp):
        if self.ndims == 2:
            if self.flag_Wishart:
                rgb_comp[-1,:] = 0 #fill the last dimension to be 0, indicating it's an isoluminant plane
            else:
                rgb_comp[self.sim['slc_RGBplane'],:] = self.sim['slc_fixedVal'];
            
        #RGB values can't exceed 1 and go below 0
        if self.flag_W_space:
            return np.clip(rgb_comp, -1, 1)
        else:
            return np.clip(rgb_comp, 0, 1)

    #%%
    
    def run_sim_1ref(self, rgb_ref, rgb_comp):
        """
        Simulates a single trial of an oddity task using the Wishart model to predict 
        the probability of correctly identifying the odd stimulus. The method involves 
        transforming the input stimuli, computing model predictions, and generating 
        a response based on the predicted probability.
    
        Args:
            rgb_ref (array-like): RGB values of the reference stimulus, normalized between 0 and 1.
            rgb_comp (array-like): RGB values of the comparison stimulus, normalized between 0 and 1.

        """
        # Convert RGB values from normalized space to Wishart model's W space
        xref = color_thresholds.N_unit_to_W_unit(rgb_ref)
        x1 = color_thresholds.N_unit_to_W_unit(rgb_comp)

        # Compute the probability of correctly identifying the odd stimulus using the signed difference.
        probC = self.gt_model_pred._compute_pChoosingX1(xref, x1)
        
        # Generate a random response based on the predicted probability and send feedback to the client
        randNum = np.random.rand(probC.shape[0]) 
        resp_binary = (randNum < probC).astype(int)
        
        return probC, resp_binary

    def run_sim(self, sim_CIELab = None, random_seed = None, colordiff_alg = 'CIE2000'):
        """
        Runs the simulation to generate comparison stimuli, calculate color differences,
        determine the probability of correct identification, and simulate binary responses.
        This function can be used for both 2D and 3D cases.
        
        Parameters:
        ----------
        sim_CIELab : object, optional
            An object that handles conversions between RGB and Lab color spaces.
            Required if `self.flag_Wishart` is False. Ignored if `self.flag_Wishart` is True.
        random_seed : int, optional
            Seed for the random number generator to ensure reproducibility. 
            If None, a random seed will be generated.
            
        colordiff_alg (str): The method for calculating deltaE. Options are:
            - 'CIE1976': DeltaE using the CIE1976 method (Euclidean distance in CIELab).
            - 'CIE1994': DeltaE using the CIE1994 method (accounts for perceptual non-uniformity).
            - 'CIE2000': DeltaE using the CIE2000 method (more advanced perceptual uniformity).
    
        """
        if colordiff_alg not in ['CIE1976', 'CIE1994', 'CIE2000']:
            raise ValueError("The method can only be 'CIE1976' or 'CIE1994' or 'CIE2000'.")

        self._initialize()
        # Set the random seed if provided, otherwise generate a random seed
        np.random.seed(random_seed)
            
        # Store the random seed in the sim dictionary for reproducibility
        self.sim['random_seed'] = random_seed
        
        # Check if sim_CIELab is required but not provided
        if not self.flag_Wishart and sim_CIELab is None:
            raise ValueError(
                "The `sim_CIELab` argument is required when `flag_Wishart` is False."
            )
    
        N = self.sim['nGridPts_ref']
        # Iterate over the grid points of the reference stimulus.
        for i in range(N):
            for j in range(N):
                # Extract the reference stimulus' RGB values for the current grid point.
                rgb_ref_ij = self.sim['ref_points'][:,i,j]
                
                # extract the ground truth
                ellPara = self._extract_ground_truth([i,j])
                
                #Generate the comparison stimulus based on the sampling method
                self.sim['rgb_comp'][i,j], _, _, _ = self.sample_rgb_comp_2DNearContour(rgb_ref_ij, ellPara)      
                    
                #if the ground truth is the Wishart fit, then we
                #repeat the ref to match the size of the sampled comparison stimuli
                rgb_ref_ij_rep = np.repeat(rgb_ref_ij[np.newaxis,:],
                                                    self.sim['nSims'], axis = 0)
                rgb_comp_ij_rep = np.transpose(self.sim['rgb_comp'][i,j,self.sim['varying_RGBplane']],(1,0))
                self.sim['probC'][i,j], self.sim['resp_binary'][i,j] = \
                    self.run_sim_1ref_Wishart(rgb_ref_ij_rep,
                                              rgb_comp_ij_rep)
                    
    #%%
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
    
    
    
    
                    
                    
                        
                    
    
    
    
    
    
    