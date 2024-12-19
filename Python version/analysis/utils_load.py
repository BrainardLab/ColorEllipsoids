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