# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 23:45:21 2025

@author: brainardlab-adm
"""

import sys
import os
sys.path.append('c:\\users\\brainardlab\\documents\\github\\colorellipsoids\\python version')
from analysis.utils_communication import CommunicateViaTextFile, ExperimentFileManager, get_experiment_info_custom

#%% Prompt the user to enter experiment information using a custom Tkinter popup
# Collects subject ID, initials, and today's session number
subject_id, subject_init, session_today = get_experiment_info_custom()

# Define the shared network path specific to the subject
networkDisk_path = f'b:\\sub{subject_id}'

# Define the name of the metadata file (pickle file) that tracks the subject's experiment data
expt_info = f'sub{subject_id}_expt_record.pkl'

# Construct the full path to the pickle file
file_path = os.path.join(networkDisk_path, expt_info)

# Load the experiment file manager state from the pickle file
expt_file_manager = ExperimentFileManager.load_state(file_path)

# Retrieve the list of past session numbers
past_session_num = list(expt_file_manager.session_data.keys())

# Find the most recent session number
session_num = max(past_session_num)

# Retrieve the file name of the most recent session
file_name = expt_file_manager.session_data[session_num]['file_name']

# Validate the subject's initials and session number against the metadata
if (expt_file_manager.session_data[session_num]['sub_initial'] != subject_init) or \
   (expt_file_manager.session_data[session_num]['session_number'] != session_today):
    raise ValueError(
        f"Mismatch detected in metadata:\n"
        f"- Expected Subject Initials: {expt_file_manager.session_data[session_num]['sub_initial']}, "
        f"but received: {subject_init}.\n"
        f"- Expected Session Number: {expt_file_manager.session_data[session_num]['session_number']}, "
        f"but received: {session_today}."
    )

#%% Initialize communication class
communicator = CommunicateViaTextFile(networkDisk_path)
communicator.check_and_handle_file(file_name)

# Step 1: Wait for Initialization
print("Waiting for initialization command...")
communicator.confirm_communication()
print("Initialization confirmed.")

# Step 2: Wait for and confirm RGB values
while True:
    if communicator.terminate:
        break
    print("Waiting for RGB values...")
    communicator.confirm_RGBvals()
    print("RGB values confirmed.")

