# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 21:57:13 2025

@author: brainardlab-adm
"""

import sys
import os
import numpy as np
sys.path.append('c:\\users\\brainardlab\\documents\\github\\colorellipsoids\\python version')
from analysis.utils_communication import CommunicateViaTextFile, ExperimentFileManager, get_experiment_info_custom

#%% Prompt the user for experiment information
# Use a custom Tkinter-based popup to collect subject ID, initials, and session number
subject_id, subject_init, session_today = get_experiment_info_custom()

# Define the main shared network disk path where files will be stored
networkDisk_path = 'c:\Shares\BrainardLab'
# Create the path for the subject's directory
path_sub = os.path.join(networkDisk_path, f'sub{subject_id}')

# Attempt to load the experiment manager state from a pickle file
try:
    # Define the metadata file name and full path
    expt_info = f'sub{subject_id}_expt_record.pkl'
    path_metadata = os.path.join(path_sub, expt_info)
    # Load the existing state of the experiment file manager
    expt_file_manager = ExperimentFileManager.load_state(path_metadata)
except:
    # If loading fails (e.g., file not found), initialize a new ExperimentFileManager
    expt_file_manager = ExperimentFileManager(subject_id, 
                                              subject_init,
                                              networkDisk_path)
# Create a new session file for the current session
file_path, file_name = expt_file_manager.create_session_file(session_today)
# List all files created for this subject
expt_file_manager.list_files()

#%% Initialize communication class
communicator = CommunicateViaTextFile(expt_file_manager.path_sub)
communicator.check_and_handle_file(file_name)

# Step 1: Initialize
print("Initializing communication...")
communicator.initialize_communication()
print("Initialization complete.")
#update the communication status
expt_file_manager.status_updates('Confirmed')

# Step 2: Send 10 sets of RGB values
rgb_values = np.random.rand(10, 3)  # Generate 10 random RGB values
for i, rgb in enumerate(rgb_values, start=1):
    print(f"Sending RGB values {i}: {rgb}")
    communicator.send_RGBvals(rgb.tolist())
    print(f"RGB values {i} confirmed.")

# Step 3: Finalize
print("Finalizing communication...")
communicator.finalize()
print("Communication finalized.")
#update the communication status
expt_file_manager.status_updates('Done')



