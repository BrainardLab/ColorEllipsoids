#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 21:10:09 2024

@author: fangfang
"""
import tkinter as tk
from tkinter import filedialog
import os
import re
import dill as pickled
import jax.numpy as jnp
import numpy as np

def select_file_and_get_path():
    # Create a hidden Tkinter root window
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Open the file dialog
    file_path = filedialog.askopenfilename(
        title="Select a File",
        filetypes=[("CSV Files", "*.csv"), ("Pickle Files", "*.pkl")]
    )
    
    # If a file is selected, split its path into directory and file name
    if file_path:
        directory, file_name = os.path.split(file_path)
        return directory, file_name
    else:
        return None, None
    
def extract_sub_number(input_string):
    """
    Extracts the integer following 'sub' in the input string.
    
    Parameters:
        input_string (str): The string containing 'sub' followed by an integer.
    
    Returns:
        int: The integer following 'sub' in the input string.
        None: If no match is found.
    """
    match = re.search(r'sub(\d+)', input_string)
    if match:
        return int(match.group(1))
    return None

#%%
class load_4D_expt_data:
    def get_all_sessions_file_names(subN, nSessions, path_str):
        """
        Generate a list of file paths for all session data files of a given subject.

        Parameters:
        subN (int or str): Subject number or identifier.
        nSessions (int): Total number of sessions.
        path_str (str): Base directory path where session files are stored.

        Returns:
        tuple: A list of file paths (session_files) and the common filename part (session_file_name_part1).
        """

        # Construct the common part of the session file names
        session_file_name_part1 = f'ColorDiscrimination_4dExpt_Isoluminant plane_sub{subN}'

        # Generate full file paths for all session files
        session_files = [
            os.path.join(path_str, f'{session_file_name_part1}_session{n+1}_copy.pkl')
            for n in range(nSessions)
        ]
        
        return session_files, session_file_name_part1
    
    def load_data_all_sessions(session_files):
        """
        Load data from all session files.

        Parameters:
        session_files (list of str): List of file paths to session data files.

        Returns:
        list: A list containing loaded data for each session.
        """

        # Load and deserialize data from all session files
        data_all_sessions = []   
        for file in session_files:
            try:
                with open(file, 'rb') as f:
                    data_all_sessions.append(pickled.load(f))
            except (FileNotFoundError, EOFError) as e:
                print(f"Warning: Skipping {file} due to error: {e}")

        return data_all_sessions
    
    def load_MOCS_data(data_allSessions):
        """
        Extract and preprocess Method of Constant Stimuli (MOCS) trial data from all sessions.

        Parameters:
        data_allSessions (list of dict): List containing data from multiple sessions.

        Returns:
        tuple: 
            - xref_MOCS_list (list of jnp.ndarray): List of reference stimuli arrays from each session.
            - x1_MOCS_list (list of jnp.ndarray): List of comparison stimuli arrays from each session.
            - y_MOCS_list (list of jnp.ndarray): List of participant response arrays from each session.
            - xref_MOCS (np.ndarray): Concatenated reference stimuli across all sessions.
            - x1_MOCS (np.ndarray): Concatenated comparison stimuli across all sessions.
            - y_MOCS (np.ndarray): Concatenated participant responses across all sessions.
        """

        # Extract MOCS trial data from each session and store in separate lists
        xref_MOCS_list = [jnp.array(d['data_vis_MOCS'].xref_all) for d in data_allSessions]
        x1_MOCS_list   = [jnp.array(d['data_vis_MOCS'].x1_all) for d in data_allSessions]
        y_MOCS_list    = [jnp.array(d['data_vis_MOCS'].y_all) for d in data_allSessions]

        # Concatenate data across all sessions along axis 0
        xref_MOCS, x1_MOCS, y_MOCS = map(lambda lst: np.concatenate(lst, axis=0), 
                                         [xref_MOCS_list, x1_MOCS_list, y_MOCS_list])
        
        return xref_MOCS_list, x1_MOCS_list, y_MOCS_list, xref_MOCS, x1_MOCS, y_MOCS

        
    def org_MOCS_by_condition(xref_MOCS, x1_MOCS, y_MOCS):
        """
        Organize MOCS data by unique reference stimulus conditions.

        Parameters:
        xref_MOCS (np.ndarray): Concatenated reference stimuli across all sessions.
        x1_MOCS (np.ndarray): Concatenated comparison stimuli across all sessions.
        y_MOCS (np.ndarray): Concatenated participant responses across all sessions.

        Returns:
        tuple:
            - xref_unique_MOCS (np.ndarray): Unique reference stimuli.
            - nRefs_MOCS (int): Number of unique reference stimulus conditions.
            - refStimulus_MOCS (list of np.ndarray): Grouped reference stimuli for each unique condition.
            - compStimulus_MOCS (list of np.ndarray): Grouped comparison stimuli for each unique condition.
            - responses_MOCS (list of np.ndarray): Grouped participant responses for each unique condition.
            - nLevels_MOCS (int): Number of unique levels per condition.
            - nTrials_MOCS (int): Total number of trials.
        """

        # Identify unique reference stimulus conditions in MOCS trials
        xref_unique_MOCS = np.unique(xref_MOCS, axis=0)

        # Count the number of unique reference stimulus conditions
        nRefs_MOCS = xref_unique_MOCS.shape[0]

        # Initialize lists to store grouped data by reference stimulus condition
        refStimulus_MOCS, compStimulus_MOCS, responses_MOCS = [], [], []

        # Iterate through each unique reference stimulus and collect matching trials
        for i in range(nRefs_MOCS):
            matching_indices = np.where((xref_MOCS == xref_unique_MOCS[i]).all(axis=1))[0]
            refStimulus_MOCS.append(xref_MOCS[matching_indices])
            compStimulus_MOCS.append(x1_MOCS[matching_indices])
            responses_MOCS.append(y_MOCS[matching_indices])

        # Determine the number of unique comparison stimulus levels per condition
        nLevels_MOCS = np.unique(x1_MOCS[matching_indices], axis=0).shape[0]

        # Total number of trials in the dataset
        nTrials_MOCS = y_MOCS.shape[0]
        
        return xref_unique_MOCS, nRefs_MOCS, refStimulus_MOCS, compStimulus_MOCS, \
            responses_MOCS, nLevels_MOCS, nTrials_MOCS            
    
    def load_AEPsych_data(data_allSessions):
        """
        Extract and preprocess AEPsych trial data from all sessions.

        Parameters:
        data_allSessions (list of dict): List containing data from multiple experimental sessions.

        Returns:
        tuple: 
            - xref_AEPsych_list (list of jnp.ndarray): List of reference stimuli arrays from each session.
            - x1_AEPsych_list (list of jnp.ndarray): List of comparison stimuli arrays from each session.
            - y_AEPsych_list (list of jnp.ndarray): List of binary response arrays from each session.
            - time_elapsed_list (list of np.ndarray): List of time elapsed arrays from each session.
            - xref_AEPsych (np.ndarray): Concatenated reference stimuli across all sessions.
            - x1_AEPsych (np.ndarray): Concatenated comparison stimuli across all sessions.
            - y_AEPsych (np.ndarray): Concatenated binary responses across all sessions.
            - time_elapsed (np.ndarray): Concatenated time elapsed across all sessions.
        """

        # Extract relevant data for AEPsych trials from each session
        time_elapsed_list = [d['expt_trials'].time_elapsed for d in data_allSessions]
        xref_AEPsych_list = [jnp.array(d['expt_trials'].xref_all) for d in data_allSessions]
        x1_AEPsych_list   = [jnp.array(d['expt_trials'].x1_all) for d in data_allSessions]
        y_AEPsych_list    = [jnp.array(d['expt_trials'].binaryResp_all) for d in data_allSessions]

        # Concatenate data across all sessions along axis 0
        xref_AEPsych, x1_AEPsych, y_AEPsych, time_elapsed = \
            map(lambda lst: np.concatenate(lst, axis=0), 
                [xref_AEPsych_list, x1_AEPsych_list, y_AEPsych_list, time_elapsed_list])

        return xref_AEPsych_list, x1_AEPsych_list, y_AEPsych_list, time_elapsed_list, \
               xref_AEPsych, x1_AEPsych, y_AEPsych, time_elapsed
    
    def combine_AEPsych_MOCS(data_AEPsych, data_MOCS, flag_combine=True):
        """
        Combine AEPsych and MOCS datasets if the flag is set to True. 
        Otherwise, return only AEPsych data.

        Parameters:
        data_AEPsych (tuple of jnp.ndarray): AEPsych data containing (xref, x1, y).
        data_MOCS (tuple of jnp.ndarray): MOCS data containing (xref, x1, y).
        flag_combine (bool, optional): Whether to combine AEPsych and MOCS data. Defaults to True.

        Returns:
        tuple:
            - xref_jnp (jnp.ndarray): Combined or AEPsych-only reference stimuli.
            - x1_jnp (jnp.ndarray): Combined or AEPsych-only comparison stimuli.
            - y_jnp (jnp.ndarray): Combined or AEPsych-only response data.
        """

        if flag_combine:
            # Stack reference stimuli (xref), comparison stimuli (x1), and responses (y) from both datasets
            xref_jnp = jnp.vstack((data_AEPsych[0], data_MOCS[0]))  # Vertical stack for 2D data
            x1_jnp   = jnp.vstack((data_AEPsych[1], data_MOCS[1]))  # Vertical stack for 2D data
            y_jnp    = jnp.hstack((data_AEPsych[2], data_MOCS[2]))  # Horizontal stack for 1D response data
        else:
            # Use only AEPsych data without modification
            xref_jnp, x1_jnp, y_jnp = data_AEPsych

        return xref_jnp, x1_jnp, y_jnp
    
    def split_AEPsych_data(xref_AEPsych, x1_AEPsych, y_AEPsych, flag_first_half = True):
        if flag_first_half:
            return xref_AEPsych[::2], x1_AEPsych[::2], y_AEPsych[::2]
        else:
            return xref_AEPsych[1::2], x1_AEPsych[1::2], y_AEPsych[1::2]

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
