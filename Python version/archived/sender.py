# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 21:57:13 2025

@author: brainardlab-adm
"""

import sys
import numpy as np
sys.path.append('c:\\users\\brainardlab\\documents\\github\\colorellipsoids\\python version')
from analysis.utils_communication import CommunicateViaTextFile, ExperimentFileManager

# Define the Dropbox path and file name
networkDisk_path = 'c:\Shares\BrainardLab'
expt_file_manager = ExperimentFileManager(subject_id= 1, networkDisk_path= networkDisk_path)
file_path, file_name = expt_file_manager.create_session_file(1)

#%% Initialize communication class
communicator = CommunicateViaTextFile(expt_file_manager.path_sub,
                                      timeout =120, retry_delay = 0.1, max_retries = 100)
communicator.check_and_handle_file(file_name)

# Step 1: Initialize
print("Initializing communication...")
communicator.initialize_communication()
print("Initialization complete.")

#%%
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