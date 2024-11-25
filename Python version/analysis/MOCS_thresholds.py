#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 10:48:25 2024

@author: fangfang
"""

import numpy as np
import sys
from scipy.optimize import minimize
sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids")
from core.probability_surface import IndividualProbSurfaceModel
from analysis.ellipses_tools import fit_2d_isothreshold_contour
from analysis.ellipsoids_tools import fit_3d_isothreshold_ellipsoid
from analysis.ellipses_tools import ellParams_to_covMat, covMat3D_to_2DsurfaceSlice

#%%
class fit_PMF_MOCS_trials():
    def __init__(self, nDim, stim, resp, flag_btst = False,
                 nBtst = 1000, target_pC = 0.667, **kwargs):
        self.nDim = nDim
        self.stim = stim #stimulus has to be centered to the origin
        #raise an error is stim.shape[1] is not equal to nDim
        
        self.resp = resp
        self.flag_btst = flag_btst
        self.nBtst = nBtst
        self.target_pC = target_pC
        self.unique_stim, self.nTrials_perLevel, self.pC_perLevel, self.stim_org, self.resp_org = self._get_unique_stim()
        self.nLevels = self.unique_stim.shape[0]
        # Set number of initializations from kwargs, or use default values if not provided
        self.nInitializations = kwargs.get('nInitializations', 20)  

        # Set bounds from kwargs, or use default bounds if not provided
        # Default: non-negative bounds
        self.bounds = kwargs.get('bounds', [(1e-4, 0.5), (1e-1, 5)]) 
        self.nGridPts = kwargs.get('nGridPts', 1200)
        
    def _get_unique_stim(self, tol = 1e-10):
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
                                     and K is the number of columns in the original array. Each row 
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
        unique_stim = np.full((num_unique, self.stim.shape[1]), np.nan)  # Unique rows for all columns
        nTrials_perLevel = np.full(num_unique, np.nan)                   # Number of trials per level
        pC_perLevel = np.full(num_unique, np.nan)                        # Proportion correct per level
        stim_org = np.full(self.stim.shape, np.nan)                      # undo shuffling
        resp_org = np.full(self.resp.shape, np.nan)
        
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
        fine_pC = self.pC_Weibull_many_trial(bestfit_params, fineVal_w0.T)
        return fine_pC
    
    @staticmethod
    def pC_Weibull_many_trial(weibull_params, xy_coords, guess_rate=1/3, eps=1e-4):
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
    
        guess_rate : float, optional
            The guessing rate, representing the probability of a correct response when 
            no information is available. Defaults to 1/3 (typical for a 3-alternative forced choice task).
    
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
        pC_hyp = self.pC_Weibull_many_trial(params, stim)
        
        # Compute the negative log-likelihood
        nLL = -np.sum(resp * np.log(pC_hyp) + (1 - resp) * np.log(1 - pC_hyp))
        
        return nLL

            
    def fit_PsychometricFunc_toData(self):
        self.bestfit_result = self._fit_PsychometricFunc(self.stim, self.resp);      
    
    def find_stim_at_targetPC_givenData(self):
        self.stim_at_targetPC = self._find_stim_at_targetPC(self.fine_pC)
        
    def reconstruct_PsychometricFunc_givenData(self):
        self.fine_pC = self._reconstruct_PsychometricFunc(self.bestfit_result.x)
        
    def bootstrap_and_refit(self):
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
    
        # Exclude the filler trial (x=0, y=1/3) but retain it for reconstruction later
        resp_org_no0 = self.resp_org[1:]
    
        # Initialize arrays to store bootstrap results
        self.resp_btst = np.full((self.nBtst,) + self.resp.shape, np.nan)  # Bootstrapped responses
        self.bestfit_result_btst = np.full((self.nBtst, 2), np.nan)        # Fitted parameters
        self.bestfit_result_nLL = np.full((self.nBtst,), np.nan)           # Negative log-likelihood
        self.fine_pC_btst = np.full((self.nBtst, self.nGridPts), np.nan)   # Reconstructed psychometric function
        self.stim_at_targetPC_btst = np.full((self.nBtst,), np.nan)        # Stimuli at target performance level
    
        # Perform bootstrap iterations
        for n in range(self.nBtst):
            print(n)  # Optional: Print iteration number for debugging/progress tracking
            
            # Resample responses (add filler trial back to the dataset)
            self.resp_btst[n] = np.append(self.resp_org[0], resp_org_no0[shuffled_idx[n]])
    
            # Fit a psychometric function to the bootstrapped dataset
            fit_btst_n = self._fit_PsychometricFunc(self.stim_org, self.resp_btst[n])
    
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
                

        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
        