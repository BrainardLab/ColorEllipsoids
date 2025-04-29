#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 10:48:25 2024

@author: fangfang
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol
import sys
from tqdm import trange
sys.path.append('/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids')
from core.model_predictions import wishart_model_pred

#%%
class fit_PMF_MOCS_trials():
    def __init__(self, nDim: int, stim: np.ndarray, resp: np.ndarray, nLevels: int,
                 flag_btst: bool = False, nBtst: int = 1000, 
                 guess_rate: float = 0.333, target_pC: float = 0.667, **kwargs: dict):
        """
        Fit a psychometric function to MOCS (Method of Constant Stimuli) trials
        and optionally perform bootstrapping to estimate confidence intervals.
        
        Parameters
        ----------
        nDim : int
            Number of dimensions in the stimulus space (e.g., 2 for 2D, 3 for 3D).
        stim : np.ndarray
            Array of stimulus values (N x nDim), where N is the number of trials.
            The stimulus should be centered at the origin.
        resp : np.ndarray
            Array of binary responses (0 or 1) corresponding to the trials.
        flag_btst : bool, optional
            Whether to perform bootstrapping (default is False).
        nBtst : int, optional
            Number of bootstrap iterations if bootstrapping is enabled (default is 1000).
        target_pC : float, optional
            Target proportion correct for threshold estimation (default is 0.667).
        **kwargs : dict
            Additional optional arguments:
            - nInitializations : int
                Number of initializations for parameter fitting.
            - bounds : list of tuples
                Bounds for the parameters of the psychometric function.
            - nGridPts : int
                Number of points for the reconstructed psychometric function.
        """

        self.nDim       = nDim #whether we are in 2D W space or 3D
        self.stim       = stim #stimulus has to be centered to the origin
        self.resp       = resp
        self.flag_btst  = flag_btst
        self.nBtst      = nBtst
        self.target_pC  = target_pC
        self.nLevels    = nLevels
        self.guess_rate = guess_rate
        #Validate inputs to ensure correctness
        self._validate_inputs()
        self.unique_stim, self.nTrials_perLevel, self.pC_perLevel, self.stim_org, self.resp_org = self._get_unique_stim()
        if self.nLevels != self.unique_stim.shape[0]:
            raise ValueError("The number of unique stimuli does not match the number of input levels!")
        
        # Set number of initializations from kwargs, or use default values if not provided
        self.nInitializations = kwargs.get('nInitializations', 20)  
        # Set bounds from kwargs, or use default bounds if not provided
        self.bounds   = kwargs.get('bounds', [(1e-4, 0.5), (1e-1, 5)]) 
        self.nGridPts = kwargs.get('nGridPts', 1200)
        
        
    def _validate_inputs(self):
        """
        Validate inputs to ensure correctness of dimensions and consistency between stim and resp.
        """
        # Check if stim.shape[1] matches nDim, raise an error if not
        if self.stim.shape[1] != self.nDim:
            raise ValueError(f"Stimulus dimensionality mismatch: "
                             f"Expected {self.nDim}, but got {self.stim.shape[1]}")
        # Check if the number of trial matches
        if self.stim.shape[0] != self.resp.shape[0]:
            raise ValueError(f"The number of responses N = {self.resp.shape[0]} does"
                             f" not match the number of trials N = {self.stim.shape[0]}!")  
        # Check if any element in the array is close to 0
        if ~np.any(np.abs(self.stim[:,0]) <= 1e-6) or ~np.any(np.abs(self.stim[:,1]) <= 1e-6):
            raise ValueError("The stimuli should be centerred to origin for this computation!")
            
    
    def _get_unique_stim(self, tol = 1e-8):
        """
        Generalized method to retrieve unique stimulus groups and compute statistics at each unique level.
        
        This method groups stimulus values in the first column within the specified tolerance
        and aggregates associated values in the other columns. Additionally, it calculates
        the number of trials (`nTrials_perLevel`) and the proportion correct (`pC_perLevel`) for 
        each unique stimulus group.
        
        Parameters:
        - tol (float): The tolerance for grouping stimulus values in the first column. 
                       Values within this tolerance are considered identical.
        
        Returns:
        - unique_stim (np.ndarray): An array of shape (M, K), where M is the number of unique groups 
                                     and K is the number of dimension (self.nDims). Each row 
                                     contains a unique stimulus group.
        - nTrials_perLevel (np.ndarray): A 1D array of shape (M,) representing the number of trials for 
                                         each unique stimulus group.
        - pC_perLevel (np.ndarray): A 1D array of shape (M,) representing the proportion correct for 
                                     each unique stimulus group.
        """
        # Extract the first column for grouping (dim1)
        stim_dim1 = self.stim[:, 0]
        
        # Scale and round values in the first column for grouping
        rounded_stim_dim1 = np.round(stim_dim1 / tol) * tol
        
        # Identify unique values in the rounded first column
        unique_stim_dim1 = np.sort(np.unique(rounded_stim_dim1))
        num_unique = len(unique_stim_dim1)  # Number of unique groups
        
        # Initialize arrays for results
        unique_stim      = np.full((num_unique, self.stim.shape[1]), np.nan)  # Unique rows for all columns
        nTrials_perLevel = np.full(num_unique, np.nan)                        # Number of trials per level
        pC_perLevel      = np.full(num_unique, np.nan)                        # Proportion correct per level
        stim_org         = np.full(self.stim.shape, np.nan)                   # undo shuffling
        resp_org         = np.full(self.resp.shape, np.nan)
        
        # Loop through each unique value in the first column
        idx_counter = 0
        for n, value in enumerate(unique_stim_dim1):
            # Find indices where the current unique value matches in the rounded array
            idx_n = np.where(value == rounded_stim_dim1)[0]
            
            # Store the aggregated values for each column
            unique_stim[n, 0] = value  # First column uses the unique rounded value
            for col in range(1, self.stim.shape[1]):  # For other columns, take the first occurrence
                unique_stim[n, col] = self.stim[idx_n[0], col]
            
            # Count the number of trials corresponding to the current unique stimulus group
            nTrials_perLevel[n] = len(idx_n)
            
            # organize the trials
            stim_org[int(idx_counter): int(idx_counter + nTrials_perLevel[n])] = unique_stim[n]
            resp_org[int(idx_counter): int(idx_counter + nTrials_perLevel[n])] = self.resp[idx_n]
            idx_counter += nTrials_perLevel[n]
            
            # Compute the proportion correct for the current unique stimulus group
            pC_perLevel[n] = np.sum(self.resp[idx_n]) / nTrials_perLevel[n]
        
        # Return the unique stimulus groups, number of trials, and proportion correct
        return unique_stim, nTrials_perLevel, pC_perLevel, stim_org, resp_org
    
    def _fit_PsychometricFunc(self, stim, resp):
        """
        Fit the Weibull psychometric function to the data using multiple random initializations.
        
        This method performs maximum likelihood estimation (MLE) to find the best-fitting parameters 
        for the Weibull psychometric function by minimizing the negative log-likelihood (nLL). 
        To avoid local minima, the optimization is repeated with multiple random initializations 
        using different starting points within the parameter bounds.
        
        Parameters
        ----------
        stim : np.ndarray
            A 2D or 3D array containing the stimulus coordinates for each trial.
        resp : np.ndarray
            A 1D array of binary responses (0 or 1) corresponding to each stimulus.
        
        Returns
        -------
        bestfit_result : OptimizeResult
            The result of the optimization for the best-fitting parameters. Contains:
            - x: Best-fitting parameters [threshold (a), steepness (b)].
            - fun: The minimized negative log-likelihood value (bestfit_nLL).
            - success: Whether the optimization was successful.
            - message: Description of the exit status.

        Raises
        ------
        ValueError
            If no optimization is successful, raises an error with the optimizer's message.
       
        """
        # Set an extremely high nLL that can be easily defeated
        bestfit_nLL = 1000
        bestfit_result = None
    
        # Perform multiple random initializations
        for n in range(self.nInitializations):       
            # Draw random initialization within the specified bounds
            initial_params_n = [np.random.uniform(*self.bounds[0]), 
                                np.random.uniform(*self.bounds[1])]
            
            # Perform optimization using `minimize`
            result = minimize(
                self.nLL_Weibull,                # Objective function
                initial_params_n,                # Initial parameters
                args=(stim, resp),               # Extra arguments passed to nLL_Weibull
                method='L-BFGS-B',               # Optimization method
                bounds=self.bounds               # Parameter bounds
            )
            
            # Update best fit if this result is successful and has a lower nLL
            if result.success and result.fun < bestfit_nLL:
                bestfit_nLL = result.fun
                bestfit_result = result
            
        # Check if the optimization was successful
        if bestfit_result is not None:
            return bestfit_result
        else:
            # Raise an error if all attempts fail
            raise ValueError("Optimization failed: " + result.message)
    
    def _find_stim_at_targetPC(self, predPC):
        """
        Find the stimulus value corresponding to the target proportion correct (target_pC).
        
        This method identifies the stimulus value from the fine grid of predicted probabilities
        (`predPC`) that is closest to the target proportion correct (`self.target_pC`).
        
        Parameters
        ----------
        predPC : np.ndarray
            A 1D array of predicted probabilities corresponding to the finely sampled stimulus values (`self.fineVal`).
        """
        idx = np.argmin(np.abs(predPC - self.target_pC))
        return self.fineVal[idx]
    
    def _reconstruct_PsychometricFunc(self, bestfit_params):
        """
        Reconstruct the psychometric function at finely sampled grid points.
        
        This method uses the best-fitting parameters from the optimization (`self.bestfit_result.x`) 
        to reconstruct the psychometric function. It computes the predicted probability of a correct 
        response (`pC`) for a set of finely sampled stimulus values, allowing visualization of the 
        psychometric function.
        
        """
        # Compute the L2 norm of the unique stimulus pairs
        # This reduces the stimulus space to a single magnitude dimension
        self.unique_stim_L2norm = np.linalg.norm(self.unique_stim, axis=1)
        
        # Create a finely spaced grid of stimulus magnitudes
        self.fineVal = np.linspace(np.min(self.unique_stim_L2norm),
                                   np.max(self.unique_stim_L2norm), 
                                   self.nGridPts)
        
        # Stack a zero column as a filler for compatibility with L2 norm computation
        fineVal_w0 = np.vstack((self.fineVal, np.full(self.fineVal.shape, 0)))
        
        # Compute predicted pC values at the finely sampled stimulus magnitudes
        fine_pC = self.pC_Weibull_many_trial(bestfit_params, 
                                             fineVal_w0.T, 
                                             guess_rate = self.guess_rate)
        return fine_pC
    
    @staticmethod
    def pC_Weibull_many_trial(weibull_params, xy_coords, guess_rate=1/3, eps=1e-6):
        """
        Compute the probability of a correct response (pC) for multiple trials 
        using the Weibull psychometric function.
    
        This function models the probability of a correct response as a function of 
        the Euclidean distance (L2 norm) of the stimulus coordinates from a reference, 
        based on the Weibull psychometric function.
    
        Parameters
        ----------
        weibull_params : np.ndarray (2,)
            Parameters of the Weibull function:
            - a : float, controls the threshold (the distance at which the response probability 
                  reaches a certain level, e.g., 82% for a 2AFC task).
            - b : float, controls the steepness (how quickly the probability changes 
                  around the threshold).
    
        xy_coords : np.ndarray
            A 2D array of shape (N, M), where N is the number of trials, and M is the 
            dimensionality of the stimulus coordinates (e.g., 2 for 2D, 3 for 3D). 
            Each row represents the coordinates of a trial.
    
        eps : float, optional
            A small value to clip the output probabilities and prevent extreme values 
            (e.g., exactly 0 or 1) that could cause numerical issues. Defaults to 1e-4.
    
        Returns
        -------
        pC_weibull : np.ndarray
            A 1D array of shape (N,) containing the probability of a correct response for 
            each trial, based on the Weibull psychometric function.
    
        """
        # Compute the L2 norm (Euclidean distance) of the stimulus coordinates
        l2_norm = np.linalg.norm(xy_coords, axis=1)
        
        # Unpack Weibull parameters: 'a' controls threshold, 'b' controls steepness
        a, b = weibull_params
        
        # Compute the probability of a correct response (pC) using the Weibull function
        pC_weibull = 1 - (1 - guess_rate) * np.exp(- (l2_norm / a)**b)
        
        # Clip probabilities to avoid numerical instability
        return np.clip(pC_weibull, eps, 1 - eps)
        
    def nLL_Weibull(self, params, stim, resp):
        """
        Compute the negative log-likelihood (nLL) for a Weibull psychometric function.
        
        This method calculates the negative log-likelihood of a Weibull psychometric 
        function given the observed stimulus-response data. It uses the predicted probabilities
        of correct responses (`pC_hyp`) and compares them to the observed responses (`resp`) 
        to evaluate the model's fit.
        
        Parameters
        ----------
        params : np.ndarray
            A 1D array containing the parameters of the Weibull function:
            - params[0]: Threshold (a), controls the point where the function transitions.
            - params[1]: Steepness (b), controls the slope of the curve.
        stim : np.ndarray
            A 2D or 3D array containing the stimulus coordinates for each trial.
        resp : np.ndarray
            A 1D array of binary responses (0 or 1) corresponding to each stimulus.
        
        Returns
        -------
        nLL : float
            The negative log-likelihood value. Lower values indicate a better fit 
            between the model predictions and the observed data.
        """
        # Compute predicted probabilities of correct responses (pC_hyp)
        pC_hyp = self.pC_Weibull_many_trial(params, stim, guess_rate= self.guess_rate)
        
        # Compute the negative log-likelihood
        nLL = -np.sum(resp * np.log(pC_hyp) + (1 - resp) * np.log(1 - pC_hyp))
        
        return nLL

            
    def fit_PsychometricFunc_toData(self):
        """
        Fit a psychometric function (PMF) to the original dataset.
    
        This method serves as a shortcut for fitting a PMF to the original stimulus-response 
        data. It uses `self._fit_PsychometricFunc`, which is also utilized for bootstrapped 
        datasets, to ensure consistency in the fitting process.
        """
        self.bestfit_result = self._fit_PsychometricFunc(self.stim, self.resp) 
    
    def find_stim_at_targetPC_givenData(self):
        self.stim_at_targetPC = self._find_stim_at_targetPC(self.fine_pC)
        
    def reconstruct_PsychometricFunc_givenData(self):
        self.fine_pC = self._reconstruct_PsychometricFunc(self.bestfit_result.x)
        
    #%% Bootstrap related methods 
    def bootstrap_and_refit(self, flag_groupwise_btst = False, seed = None):
        """
        Perform bootstrap resampling and refit the psychometric function.
    
        This method generates bootstrapped datasets by resampling the observed responses 
        with replacement for each stimulus level. For each bootstrapped dataset, it fits 
        a psychometric function, extracts parameter estimates, and reconstructs the 
        psychometric function with finer grid points.
        
        Attributes
        ----------
        - self.nBtst : int
            Number of bootstrap iterations.
        - self.nLevels : int
            Number of stimulus levels (excluding the filler trial).
        - self.nGridPts : int
            Number of finely spaced grid points for reconstructing the psychometric function.
        - self.stim_org : np.ndarray
            Original stimulus values, including the filler trial at the beginning.
        - self.resp_org : np.ndarray
            Original response values, including the filler trial at the beginning.
    
        Returns
        -------
        Updates the following attributes:
        - self.resp_btst : np.ndarray
            Bootstrapped response datasets.
        - self.bestfit_result_btst : np.ndarray
            Fitted parameters for each bootstrap iteration.
        - self.bestfit_result_nLL : np.ndarray
            Negative log-likelihood for each bootstrap iteration.
        - self.fine_pC_btst : np.ndarray
            Reconstructed psychometric function for each bootstrap iteration.
        - self.stim_at_targetPC_btst : np.ndarray
            Stimulus value corresponding to the target performance level for each iteration.
        """
        if seed is not None:
            np.random.seed(seed)
        
        if flag_groupwise_btst:
            # Draw random integers to generate bootstrap samples (trial indices)
            # `random_int` contains indices sampled with replacement for each level 
            #(excluding the filler trial)
            random_int = np.random.randint(
                low=0,
                high=np.max(self.nTrials_perLevel),
                size=(self.nBtst, self.resp.shape[0] - 1)
            )
        
            # Adjust indices to account for each level (trial multiples)
            # Add multiples of trials for correct indexing
            add_trialMultiples = np.repeat(np.arange(self.nLevels - 1), 
                                           np.max(self.nTrials_perLevel)).astype(int)
            shuffled_idx = random_int + (add_trialMultiples * np.max(self.nTrials_perLevel)).astype(int)
        else:
            # Combine all trials into a single pool (excluding the filler trial)
            all_indices = np.arange(self.resp.shape[0]-1)

            # Generate bootstrap samples from the combined pool
            shuffled_idx = np.random.choice(
                all_indices, 
                size=(self.nBtst, self.resp.shape[0]-1), 
                replace=True
            )

        # Exclude the filler trial (x=0, y=1/3) but retain it for reconstruction later
        resp_org_no0 = self.resp_org[1:]
        stim_org_no0 = self.stim_org[1:]
    
        # Initialize arrays to store bootstrap results
        self.stim_btst = np.full((self.nBtst,) + self.stim.shape, np.nan)  # Bootstrapped stimuli
        self.resp_btst = np.full((self.nBtst,) + self.resp.shape, np.nan)  # Bootstrapped responses
        self.bestfit_result_btst = np.full((self.nBtst, 2), np.nan)        # Fitted parameters
        self.bestfit_result_nLL = np.full((self.nBtst,), np.nan)           # Negative log-likelihood
        self.fine_pC_btst = np.full((self.nBtst, self.nGridPts), np.nan)   # Reconstructed psychometric function
        self.stim_at_targetPC_btst = np.full((self.nBtst,), np.nan)        # Stimuli at target performance level
    
        # Perform bootstrap iterations
        for n in range(self.nBtst):            
            # Resample responses (add filler trial back to the dataset)
            self.resp_btst[n] = np.append(self.resp_org[0], resp_org_no0[shuffled_idx[n]])
            self.stim_btst[n] = np.vstack((self.stim_org[0], stim_org_no0[shuffled_idx[n]]))
    
            # Fit a psychometric function to the bootstrapped dataset
            fit_btst_n = self._fit_PsychometricFunc(self.stim_btst[n], self.resp_btst[n])
    
            # Extract fitted parameters and negative log-likelihood
            self.bestfit_result_btst[n] = fit_btst_n.x  # Parameter estimates
            self.bestfit_result_nLL[n] = fit_btst_n.fun  # Negative log-likelihood
    
            # Reconstruct the psychometric function on a finer stimulus grid
            self.fine_pC_btst[n] = self._reconstruct_PsychometricFunc(self.bestfit_result_btst[n])
    
            # Identify the stimulus corresponding to the target performance level (e.g., 0.667)
            self.stim_at_targetPC_btst[n] = self._find_stim_at_targetPC(self.fine_pC_btst[n])
            
    def compute_95btstCI(self):
        """
        Compute the 95% bootstrap confidence intervals for the psychometric function predictions.
    
        This method calculates confidence intervals based on the bootstrapped estimates of the 
        stimulus corresponding to the target probability of correct responses (e.g., 66.7%). 
        It also computes the confidence intervals for the model-predicted probability of correct 
        responses at finer grid points.
    
        Attributes Updated
        ------------------
        - self.stim_at_targetPC_95btstCI : np.ndarray
            A 1D array containing the lower and upper bounds of the 95% confidence interval 
            for the stimulus corresponding to the target performance level.
        - self.stim_at_targetPC_95btstErr : np.ndarray
            A 1D array containing the lower and upper error bounds relative to the central 
            stimulus estimate for the target performance level.
        - self.fine_pC_95btstCI : np.ndarray
            A 2D array of shape (2, nGridPts), where the first row contains the lower bound 
            and the second row contains the upper bound of the 95% confidence intervals 
            for the probability correct at each grid point.
        """
        # Step 1: Sort the bootstrapped stimulus values in ascending order
        val_sorted = np.sort(self.stim_at_targetPC_btst)
        
        # Step 2: Compute the indices corresponding to the 2.5% and 97.5% percentiles
        idx_lb = int(np.floor(self.nBtst * 0.025))  # Lower bound index
        idx_ub = int(np.ceil(self.nBtst * 0.975))  # Upper bound index
        
        # Step 3: Extract the 95% confidence interval for the stimulus at the target performance level
        self.stim_at_targetPC_95btstCI = val_sorted[[idx_lb, idx_ub]]
        
        # Step 4: Compute the error bounds relative to the central stimulus estimate
        self.stim_at_targetPC_95btstErr = np.array([
            self.stim_at_targetPC - self.stim_at_targetPC_95btstCI[0],  # Lower error
            self.stim_at_targetPC_95btstCI[1] - self.stim_at_targetPC   # Upper error
        ])
        
        # Step 5: Sort the bootstrapped probability correct predictions (finer grid)
        # The array has shape (nBtst, nGridPts); sort along the bootstrap axis (axis=0)
        arr_sorted = np.sort(self.fine_pC_btst, axis=0)
        
        # Step 6: Extract the 95% confidence interval for the probability correct at each grid point
        self.fine_pC_95btstCI = arr_sorted[[idx_lb, idx_ub]]

def compute_Wishart_based_pCorrect_atMOCS(numBtst, nLevels, fit_PMF_MOCS, 
                                          xref_unique, model_pred_existing,
                                          color_thres_data, ndims = 2, return_dict=False):
    """
    Computes Wishart model-based predictions of proportion correct along MOCS directions,
    and extracts threshold estimates and corresponding stimulus locations.

    Returns:
        If return_dict is False:
            pChoosingX1_Wishart: Probability of choosing X1 as odd, predicted by the Wishart model.
            vecLen_at_targetPC_Wishart: Vector length at target performance (e.g., 66.7%) predicted by the Wishart model.
            stim_at_targetPC_Wishart: Stimulus locations corresponding to Wishart model thresholds.
    
        If return_dict is True:
            Dictionary with variable names as keys and corresponding arrays/lists as values.

    """
    
    model_pred = wishart_model_pred(model_pred_existing.model, model_pred_existing.opt_params, 
                                    model_pred_existing.w_init_key,
                                    model_pred_existing.opt_key, 
                                    model_pred_existing.W_init,
                                    model_pred_existing.W_est, 
                                    model_pred_existing.Sigmas_recover_grid,
                                    color_thres_data, 
                                    target_pC= model_pred_existing.target_pC,
                                    ngrid_bruteforce = 1000,
                                    bds_bruteforce = [0.0005, 0.25])
    
    # Initialize arrays to store results
    nRefs = len(fit_PMF_MOCS)
    pChoosingX1_Wishart          = np.full((nRefs, fit_PMF_MOCS[0].nGridPts), np.nan)
    vecLen_at_targetPC_Wishart   = np.full((nRefs,), np.nan)
    stim_at_targetPC_Wishart     = np.full((nRefs, ndims), np.nan)

    for n in trange(nRefs):
        # Sort stimulus vectors by descending distance from the origin
        sorted_indices = np.argsort(-np.linalg.norm(fit_PMF_MOCS[n].unique_stim, axis=1))
        sorted_array = fit_PMF_MOCS[n].unique_stim[sorted_indices]

        # Generate a finer grid of stimuli along the most distant chromatic direction
        finer_stim = sim_MOCS_trials.create_discrete_stim(
            sorted_array[0], 
            fit_PMF_MOCS[n].nGridPts,
            ndims= ndims
        )

        # Predict proportion correct (pChoosingX1) using the Wishart model
        pChoosingX1_Wishart[n] = model_pred._compute_pChoosingX1(
            np.full(finer_stim.shape, 0) + xref_unique[n], 
            finer_stim + xref_unique[n]
        )

        # Find the vector length corresponding to target performance (e.g., 66.7%) from Wishart predictions
        vecLen_at_targetPC_Wishart[n] = fit_PMF_MOCS[n]._find_stim_at_targetPC(pChoosingX1_Wishart[n])

        # Compute stimulus coordinates at Wishart threshold
        stim_at_targetPC_Wishart[n] = vecLen_at_targetPC_Wishart[n] * (
            fit_PMF_MOCS[n].unique_stim[nLevels // 2] /
            np.linalg.norm(fit_PMF_MOCS[n].unique_stim[nLevels // 2])
        ) + xref_unique[n]

    if return_dict:
        Wishart_based_thres_atMOCS = {
            "pChoosingX1_Wishart": pChoosingX1_Wishart,
            "vecLen_at_targetPC_Wishart": vecLen_at_targetPC_Wishart,
            "stim_at_targetPC_Wishart": stim_at_targetPC_Wishart,
        }
        return Wishart_based_thres_atMOCS
    else:
        return pChoosingX1_Wishart, vecLen_at_targetPC_Wishart, stim_at_targetPC_Wishart

    
    
#%%            
class sim_MOCS_trials:
    @staticmethod
    def generate_vectors_min_angle(min_angle_degrees=60, max_angle_degrees=160,
                                   ndims=2, seed=None):
        """        
        Generate two random vectors in a given dimension (2D or 3D) such that their angle is 
        at least `min_angle_degrees` apart and at most `max_angle_degrees` apart.

        Args:
            min_angle_degrees (float): The minimum angle (in degrees) between the two vectors.
            max_angle_degrees (float): The maximum angle (in degrees) between the two vectors.
            ndims (int): Dimension of the vectors (2 for plane, 3 for RGB cube).
            seed (int, optional): Seed for the random number generator for reproducibility.

        Returns:
            tuple: Two numpy arrays representing the two vectors.
            
        """
        if seed is not None:
            np.random.seed(seed)

        # Convert angles to radians
        min_angle_radians = np.radians(min_angle_degrees)
        max_angle_radians = np.radians(max_angle_degrees)

        while True:
            # Generate two random vectors in the specified dimension
            vector1 = np.random.randn(ndims)
            vector2 = np.random.randn(ndims)

            # Normalize the vectors to make them unit vectors
            vector1 /= np.linalg.norm(vector1)
            vector2 /= np.linalg.norm(vector2)

            # Compute the cosine of the angle between the two vectors
            cos_theta = np.dot(vector1, vector2)
            cos_theta = np.clip(cos_theta, -1, 1)  # Ensure numerical stability
            angle = np.arccos(cos_theta)  # Get the angle in radians

            # Check if the vectors satisfy the angle constraints
            if min_angle_radians <= angle <= max_angle_radians:
                return vector1, vector2

        
    @staticmethod
    def sim_binary_trials(p, N, seed=None):
        """
        Simulate binary responses based on a probability `p` for `N` trials.

        Args:
            p (float): Probability of success (1) for each trial (0 ≤ p ≤ 1).
            N (int): Number of trials to simulate.
            seed (int, optional): Seed for reproducibility.

        Returns:
            numpy.ndarray: A 1D array of binary responses (0 or 1).
        """
        # if seed is not None:
        #     np.random.seed(seed)

        # # Generate binary responses directly using a binomial distribution
        # resp = np.random.binomial(1, p, N)
        
        rng = np.random.default_rng(seed)  # Ensures rng is always defined
        random_values = rng.random(N)  # Generate N random numbers between 0 and 1
        resp = (random_values < p).astype(int)  # Convert to binary responses
        return resp, np.mean(resp)
        
    @staticmethod
    def create_discrete_stim(endpoint, num_pts, startpoint = None, ndims = 2):
        startpoint = np.array([0]*ndims)
        if endpoint.shape[0] != startpoint.shape[0] != ndims:
            raise ValueError('The dimensions of points do not match!')
         
        for n in range(ndims):
            discrete_dim_n = np.linspace(startpoint[n], endpoint[n], num_pts)
            if n == 0:
                discrete_stim = discrete_dim_n[:,np.newaxis]
            else:
                discrete_stim = np.hstack((discrete_stim, discrete_dim_n[:,np.newaxis]))
        return discrete_stim
    
    @staticmethod
    def generate_stacked_grids(bds, num_grid_pts, ndims = 2):
        """
        Generate and stack multiple grids based on given boundaries, number of grid points, and dimensionality.
    
        Args:
            bds (array-like): List or array of boundary values for each grid.
            num_grid_pts (array-like): List or array specifying the number of points per dimension.
            ndims (int): Number of dimensions for the grid.
    
        Returns:
            numpy.ndarray: Stacked grids, with shape (total_points, dim).
        """
        # Ensure inputs are numpy arrays
        bds = np.array(bds)
        num_grid_pts = np.array(num_grid_pts)
        
        # Check if bds and num_grid_pts have the same shape
        if bds.shape != num_grid_pts.shape:
            raise ValueError("bds and num_grid_pts must have the same shape.")
    
        stacked_grids = []
    
        # Generate grids for each boundary and corresponding number of points
        for bd, num_pts in zip(bds, num_grid_pts):
            # Generate a list of linspace arrays for each dimension
            linspaces = [np.linspace(-bd, bd, num_pts) for _ in range(ndims)]
            grid = np.stack(np.meshgrid(*linspaces, indexing='ij'), axis=-1)  # 'ij' ensures correct order in any dim
            stacked_grids.append(grid.reshape(-1, ndims))  # Flatten the grid
        
        # Stack all grids together
        return np.vstack(stacked_grids)
        
    @staticmethod
    def sample_sobol(N, lb, ub, force_center=False, seed=None):
        """
        Generate N Sobol-sequenced points within a bounded space in arbitrary dimensions,
        optionally forcing the first point to be at the center.
    
        Args:
            N (int): Number of points to sample.
            lb (array-like): Lower bounds for each dimension.
            ub (array-like): Upper bounds for each dimension.
            force_center (bool): If True, the first point is set at the center.
            seed (int, optional): Random seed for reproducibility.
    
        Returns:
            np.ndarray: (N, len(lb)) array of Sobol samples.
        """
        lb = np.array(lb)
        ub = np.array(ub)
        ndims = len(lb)  # Determine number of dimensions from bounds
    
        if N < 1:
            raise ValueError("N must be at least 1.")
        if len(lb) != len(ub):
            raise ValueError("Lower and upper bounds must have the same length.")
        
        # Initialize Sobol sequence generator
        sobol_sampler = Sobol(d=ndims, scramble=True, seed=seed)
    
        # Generate N Sobol points in [0,1]^ndims
        samples = sobol_sampler.random(N)
    
        # Scale to [lb, ub] for each dimension
        samples = lb + (ub - lb) * samples  
    
        if force_center:
            samples[0] = (lb + ub) / 2  # Force first point to be at the center
    
        return samples
        
        
        
    
    
    
    
    
    
    
        
        
        