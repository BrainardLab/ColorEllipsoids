#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 12:56:01 2025

@author: fangfang
"""
import jax
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(None)

#%%
class expt_data:
    def __init__(self, xref_all, x1_all, y_all, pseudo_order):
        self.xref_all = xref_all
        self.x1_all = x1_all
        self.y_all = y_all
        self.pseudo_order = pseudo_order
        
class CrossValidation:
    @staticmethod
    def shuffle_data(data, xref_unique, tol=5e-2, seed=None, debug_plot = False):
        """
        Shuffle the data separately for each unique reference location.

        Parameters
        ----------
        data : tuple of np.ndarray
            - Contains three arrays: `(y_all, xref_all, x1_all)`, where:
              - `y_all`: Measured responses (dependent variable), shape `(N,)`
              - `xref_all`: Reference locations tested in the experiment (independent variable), shape `(N, M)`
              - `x1_all`: Comparison stimuli (independent variable), shape `(N, M)`

        xref_unique : np.ndarray, shape `(K, M)`
            - `K`: Number of unique reference locations.
            - Stores the unique reference locations used in the experiment.
        
        tol : float, optional (default=5e-2)
            - Tolerance for matching reference locations.
            - A trial is considered to match a reference if the absolute difference is below `tol` in all dimensions.

        seed : int, optional
            - Seed for the random number generator to ensure reproducibility.

        Returns
        -------
        data_shuffled : tuple of np.ndarray
            - Contains three arrays: `(y_shuffled, xref_shuffled, x1_shuffled)`, each shuffled within reference locations.
        """

        # Unpack input data
        y_all, xref_all, x1_all = data    

        # Initialize shuffled arrays with the same shape
        y_shuffled = np.empty_like(y_all)
        xref_shuffled = np.empty_like(xref_all)
        x1_shuffled = np.empty_like(x1_all)

        # Set random seed for reproducibility if provided
        if seed is not None:
            np.random.seed(seed)

        # Shuffle data within each unique reference location
        for ref_n in xref_unique:
            # Find the indices of trials matching the current reference location
            idx_match_original = np.where(np.all(np.abs(xref_all - ref_n) < tol, axis=1))[0]

            # Shuffle indices in place
            idx_match = np.array(idx_match_original)
            np.random.shuffle(idx_match)

            lb, ub = np.min(idx_match), np.max(idx_match)+1
            #put them in the appropirate array
            y_shuffled[lb:ub] = y_all[idx_match]
            xref_shuffled[lb:ub] = xref_all[idx_match]
            x1_shuffled[lb:ub] = x1_all[idx_match]
            
            if debug_plot:
                fig, ax = plt.subplots(1, 2)
                y_slc = y_all[idx_match_original]
                x1_slc = x1_all[idx_match_original]
                ax[0].scatter(x1_slc[y_slc == 1, 0], x1_slc[y_slc == 1, 1], color = 'g', s=1)
                ax[0].scatter(x1_slc[y_slc == 0, 0], x1_slc[y_slc == 0, 1], color = 'r', marker = 'x',s=1)
                ax[0].set_title('Before shuffling')
                
                yy_slc = y_shuffled[lb:ub]
                xx1_slc = x1_shuffled[lb:ub]
                ax[1].scatter(xx1_slc[yy_slc == 1, 0], xx1_slc[yy_slc == 1, 1], color = 'g')
                ax[1].scatter(xx1_slc[yy_slc == 0, 0], xx1_slc[yy_slc == 0, 1], color = 'r', marker = 'x')
                ax[1].set_title('After shuffling')
                

        return (y_shuffled, xref_shuffled, x1_shuffled)

    @staticmethod
    def select_NFold_data(data, total_folds, nRefs, nTrials_perRef, ndims = 2, debug_plot = False):
        """
        Splits the dataset into training and validation (held-out) sets using N-Fold cross-validation.
        
        Parameters
        ----------
        data : tuple of np.ndarray
            - Contains three arrays: `(y_all, xref_all, x1_all)`, where:
              - `y_all`: Measured responses (dependent variable), shape `(N,)`
              - `xref_all`: Reference locations tested in the experiment (independent variable), shape `(N, M)`
              - `x1_all`: Comparison stimuli (independent variable), shape `(N, M)`
              
        total_folds : int, optional (default=10)
                  - Total number of folds for cross-validation.
            
        nRefs : int
            - Number of unique reference locations.
        
        nTrials_perRef : int
            - Number of trials per reference location.
            
        ndims : int, optional (default=2)
            - Number of dimensions in the stimulus space (e.g., 2D color space).
        
        debug_plot : bool, optional (default=False)
            - If True, generates a scatter plot visualizing the split data.
        
        Returns
        -------
        data_keep : tuple of np.ndarray
            - The subset of data used for training.
            - Contains three arrays: `(y_keep, xref_keep, x1_keep)`, each of shape `(remaining_trials,)`
        
        data_heldout : tuple of np.ndarray
            - The subset of data held out for validation.
            - Contains three arrays: `(y_heldout, xref_heldout, x1_heldout)`, each of shape `(held_out_trials,)`
        
        Raises
        ------
        ValueError
            - If the total number of trials does not match the expected shape.
            - If `Nth_fold` exceeds `total_folds`.
        
        """
        # Unpack input data
        y_all, xref_all, x1_all = data  

        # Validate that the total number of trials matches expectations
        if nTrials_perRef * nRefs != y_all.shape[0] or \
           y_all.shape[0] != xref_all.shape[0] or \
           y_all.shape[0] != x1_all.shape[0]:
            raise ValueError('Size mismatch: The number of trials does not match expectations!')
            
        # Ensure the fold index does not exceed the total number of folds
        if nTrials_perRef * nRefs % total_folds != 0:
            raise ValueError('The number of data is not divisible by {total_folds} folds!')
            
        data_org = {key: None for key in range(1, total_folds+1)}
        
        # Reshape data to group trials by reference locations
        base_shape = (nRefs, nTrials_perRef)
        y_reshape = np.reshape(y_all, base_shape)
        xref_reshape = np.reshape(xref_all, base_shape + (ndims,))
        x1_reshape = np.reshape(x1_all, base_shape + (ndims,))
        
        for n in range(total_folds):
            # Determine column indices for the held-out fold
            col_lb = int(nTrials_perRef / total_folds * n)  # Lower bound
            col_ub = int(nTrials_perRef / total_folds * (n + 1))  # Upper bound
            nTrials_heldout = (col_ub - col_lb) * nRefs  # Number of held-out trials
            
            # Compute row indices
            idx_all = np.arange(y_all.shape[0]).reshape(nRefs, nTrials_perRef)
            idx_heldout = idx_all[:, col_lb:col_ub].flatten()
            idx_keep = np.delete(idx_all, np.s_[col_lb:col_ub], axis=1).flatten()
            
            # Extract held-out (validation) data
            y_heldout = np.reshape(y_reshape[:, col_lb:col_ub], (nTrials_heldout,))
            xref_heldout = np.reshape(xref_reshape[:, col_lb:col_ub], (nTrials_heldout, ndims))
            x1_heldout = np.reshape(x1_reshape[:, col_lb:col_ub], (nTrials_heldout, ndims))
            data_heldout = (y_heldout, xref_heldout, x1_heldout)
            
            # Identify indices of trials to keep (training data)
            col_idx = list(range(nTrials_perRef))
            del col_idx[col_lb:col_ub]  # Remove indices belonging to the held-out set
            nTrials_keep = len(col_idx) * nRefs  # Number of retained trials
    
            # Extract training data
            y_keep = np.reshape(y_reshape[:,col_idx],(nTrials_keep))
            xref_keep = np.reshape(xref_reshape[:,col_idx], (nTrials_keep, ndims))
            x1_keep = np.reshape(x1_reshape[:,col_idx], (nTrials_keep, ndims))
            data_keep = (y_keep, xref_keep, x1_keep)
            
            # Optional Debug Plot: Visualize the split between training and validation sets
            if debug_plot:
                fig, ax = plt.subplots(1,1)
                ax.scatter(x1_keep[y_keep==1,0], x1_keep[y_keep==1,1], color = 'g',s=3)
                ax.scatter(x1_keep[y_keep==0,0], x1_keep[y_keep==0,1], color = 'r',s= 3)
                ax.scatter(x1_heldout[y_heldout==1,0], x1_heldout[y_heldout==1,1],color = 'b',s=3)
                ax.scatter(x1_heldout[y_heldout==0,0], x1_heldout[y_heldout==0,1],color = 'y',s=3)
            
            data_org[n+1] = (data_keep, data_heldout, idx_keep, idx_heldout)
        
        return data_org
    
    @staticmethod
    def select_LOO_data(data, xref_heldout, nTrials, nRefs, tol=5e-2):
        """
        Selects data for Leave-One-Out (LOO) analysis by excluding trials where 
        the reference location matches the held-out location within a given tolerance.

        Parameters
        ----------
        data : tuple of np.ndarray
            - Contains three arrays: `(y_all, xref_all, x1_all)`
            - `y_all`: Measured responses or dependent variable. `shape (N,)`
            - `xref_all`: Reference locations tested in the experiment (independent variable). `shape (N, M)`
            - `x1_all`: Comparison stimuli or another independent variable. `shape (N, M)`
        
        xref_heldout : np.ndarray, shape (1, M)
            - The reference location that should be excluded from the analysis.
            - The function removes all trials where the reference matches `xref_heldout` within `tol`.

        nTrials : int
            - Number of trials per reference location.

        nRefs : int
            - Total number of unique reference locations in the dataset.

        tol : float, optional (default=5e-2)
            - Tolerance threshold for comparing reference locations.
            - A reference location is considered a match to `xref_heldout` if 
              the absolute difference is less than `tol` in all dimensions.

        Returns
        -------
        data_keep : tuple of np.ndarray
            - Subset of the original data excluding the trials with `xref_heldout`.
            - Contains three arrays `(y_keep, xref_keep, x1_keep)`, each filtered to exclude held-out trials.

        Notes
        -----
        - The function assumes that `xref_all` contains `nTrials * nRefs` total entries.
        - The expected number of retained trials should be `nTrials * (nRefs - 1)`, 
          since one reference location is removed.
        - If the retained trial count does not match the expected number, a warning is displayed.
        """

        # Unpack input data
        y_all, xref_all, x1_all = data

        # Identify indices of trials where xref matches xref_heldout within tolerance
        idx_exclude = np.where(np.all(np.abs(xref_all - xref_heldout) < tol, axis=1))[0]
        
        # Identify indices of trials to keep (excluding held-out trials)
        idx_keep = np.setdiff1d(np.arange(nTrials * nRefs), idx_exclude)
        
        # Validate that the number of retained trials matches the expected count
        expected_trials = nTrials * (nRefs - 1)
        if len(idx_keep) != expected_trials:
            print(f"Warning: Expected {expected_trials} trials, but retained {len(idx_keep)} trials!")

        # Extract and return the filtered subset of data
        data_keep = (y_all[idx_keep], xref_all[idx_keep], x1_all[idx_keep])
        
        return data_keep