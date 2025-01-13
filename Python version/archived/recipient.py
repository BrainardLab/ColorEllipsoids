# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 23:45:21 2025

@author: brainardlab-adm
"""

import sys
sys.path.append('c:\\users\\brainardlab-adm\\documents\\github\\colorellipsoids\\python version')
from analysis.utils_communication import CommunicateViaTextFile

# Define the Dropbox path and file name
dropbox_path = 'c:\\users\\brainardlab-adm\\Aguirre-Brainard Lab Dropbox\\Fangfang Hong\\ELPS_analysis\\SanityChecks_DataFiles'
file_name = 'test_communication.txt'

# Initialize communication class
communicator = CommunicateViaTextFile(dropbox_path, timeout = 120, retry_delay=0.1, max_retries=10)
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
    