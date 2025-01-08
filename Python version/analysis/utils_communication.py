# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 20:55:45 2025

@author: Fangfang
"""

import time
from datetime import datetime
import socket
import os
import numpy as np

class CommunicateViaTextFile:
    def __init__(self, dropbox_file_path, retry_delay = 0.1, timeout = 30,
                 max_retries = 10):
        """
        Initialize the CommunicateViaTextFile class with the file path.
        
        Args:
            dropbox_file_path (str): The file path to which messages will be appended.
        """
        self.dropbox_file_path = dropbox_file_path
        self.computer_name = socket.gethostname()
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.max_retries = max_retries
        self.terminate = False
        
    def check_and_handle_file(self, file_name):
        """
        Checks if the directory exists, and handles file creation or renaming.

        Args:
            file_name (str): The name of the file to check or create.

        Returns:
            str: The full path of the newly created or handled file.
        """
        # Ensure the directory exists
        if not os.path.exists(self.dropbox_file_path):
            os.makedirs(self.dropbox_file_path)

        # Full path to the file
        file_path = os.path.join(self.dropbox_file_path, file_name)

        # Check if the file exists
        if os.path.exists(file_path):
            # # If the file exists, rename it with a timestamp
            # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            # name, ext = os.path.splitext(file_name)
            # new_file_name = f"{name}_{timestamp}{ext}"
            # old_file_path = os.path.join(self.dropbox_file_path, new_file_name)
            # os.rename(file_path, old_file_path)
            print('Found the file.')
        else:
            # Create a new file with the original name
            with open(file_path, 'w'):
                pass  # File is created and immediately closed

        self.dropbox_fullfile = file_path

    def append_message_to_file(self, message):
        """
        Appends a message with a timestamp to the specified file.
        
        Args:
            message (str): The message to append to the file.
        
        Raises:
            IOError: If the file cannot be opened for writing after the maximum retries.
        """
        retry_count = 0

        while retry_count < self.max_retries:
            try:
                # Open the file in append mode
                with open(self.dropbox_fullfile, 'a') as file:
                    # Get the current timestamp
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                    # Append the message with a timestamp to the file
                    file.write(f"{timestamp} - {self.computer_name}: {message}\n")
                    return  # Exit the function after successful write
            except IOError:
                # If file opening fails, increment retry count and pause
                retry_count += 1
                time.sleep(self.retry_delay)

        # If we reach here, it means we failed to open the file
        raise IOError(f"Failed to open file for writing after {self.max_retries} retries.")
        
    def extract_last_line(self):
        retry_count = 0
        while retry_count < self.max_retries:
            try:
                # Open the file for reading
                with open(self.dropbox_fullfile, 'r') as file:
                    last_line = ''
                    # Read the file line by line to get the last line
                    for line in file:
                        last_line = line.strip()  # Keep trimming whitespace
                return last_line
            except IOError:
                # If file opening fails, increment retry count and pause
                retry_count += 1
                time.sleep(self.retry_delay)

        # If we reach here, it means we failed to open the file
        raise IOError(f"Failed to open file for writing after {self.max_retries} retries.")
    
    def extract_last_word_in_file(self, last_line = None):
        """
        extract the last word.
    
        Args:
            word (str): The word to check.
    
        Returns:
            tuple: A tuple containing:
                - bool: True if the word is the last word of the last line, otherwise False.
                - str: The last line of the file.
        
        Raises:
            IOError: If the file cannot be opened for reading.
        """
        try:
            if last_line is None:
                # Read the file line by line to get the last line
                last_line = self.extract_last_line()
    
            # Split the last line into words
            if last_line:
                words = last_line.split()
                return words[-1]
    
            # If the file is empty, return False and an empty string
            return False, ""
        except IOError:
            raise IOError("Failed to open file for reading.")
        
    def check_last_word_in_file(self, word, last_word = None):
        """
        Checks if the specified word is the last word of the last line in the file.
    
        Args:
            word (str): The word to check.
    
        Returns:
            tuple: A tuple containing:
                - bool: True if the word is the last word of the last line, otherwise False.
                - str: The last line of the file.
        
        Raises:
            IOError: If the file cannot be opened for reading.
        """
        if last_word is None:
            last_word = self.extract_last_word_in_file()
        return last_word == word
            
    def extract_rgb_values(self, input_string):
        """
        Extracts the RGB values from the input string.

        Args:
            input_string (str): The input string containing RGB values.

        Returns:
            np.ndarray: A NumPy array of the RGB values.
        """
        try:            
            # Find the RGB part in the message
            rgb_part = input_string.split(' ')[-2]  # Extract the part before "Image_Display"
            
            # Extract individual R, G, B values
            r_value = float(rgb_part.split('_')[0][1:])  # Remove 'R' and convert to float
            g_value = float(rgb_part.split('_')[1][1:])  # Remove 'G' and convert to float
            b_value = float(rgb_part.split('_')[2][1:])  # Remove 'B' and convert to float
            
            # Return as a NumPy array
            return np.array([r_value, g_value, b_value])
        except Exception as e:
            raise ValueError(f"Failed to extract RGB values from input: {input_string}. Error: {e}")
            
    def initialize_communication(self):
        """
        Writes an initial message to the file and waits for a response from Unity 
        indicating that initialization is complete.
        
        Raises:
            IOError: If the file cannot be opened for writing or reading.
        """
        # Append the message to the file
        self.append_message_to_file("Set_Up_to_Communicate")

        start_time = time.time()
        # Wait for Unity to send back "Ready_To_Communicate"
        while True:
            is_ready_to_communicate = self.check_last_word_in_file("Ready_To_Communicate")
            if is_ready_to_communicate:
                break
            
            # Check if the timeout duration has been exceeded
            if time.time() - start_time > self.timeout:
                raise TimeoutError(f"Timeout: Did not receive 'Ready_To_Communicate' within {self.timeout} seconds.")
            # Pause for a short period to prevent CPU overload
            time.sleep(self.retry_delay)
            
    def confirm_communication(self):
        start_time = time.time()

        # Wait for command
        while True:
            is_set_up_to_communicate = self.check_last_word_in_file("Set_Up_to_Communicate")
            if is_set_up_to_communicate:
                # Append the message to the file
                self.append_message_to_file("Ready_To_Communicate")
                break
                
            # Check if the timeout duration has been exceeded
            if time.time() - start_time > self.timeout:
                raise TimeoutError(f"Timeout: Did not receive 'Set_Up_to_Communicate' within {self.timeout} seconds.")
            # Pause for a short period to prevent CPU overload
            time.sleep(self.retry_delay)
            
    def send_RGBvals(self, target_settings):
        """
        Sends the current RGB values for display to the file and waits for confirmation.
        
        Args:
            target_settings (list): RGB values [R, G, B] to be displayed.
        
        Raises:
            TimeoutError: If the recipient does not confirm the RGB values within the timeout duration.
        """
        # Create the message to indicate the current stimulus to display
        message_image_for_display = f"R{target_settings[0]:.4f}_G{target_settings[1]:.4f}_B{target_settings[2]:.4f} Image_Display"

        # Append the message to the file
        self.append_message_to_file(message_image_for_display)

        start_time = time.time()

        # Wait for Unity to send back a message indicating the image has been displayed
        while True:
            is_image_confirmed = self.check_last_word_in_file("Image_Confirmed")
            if is_image_confirmed:
                break

            # Check if the timeout duration has been exceeded
            if time.time() - start_time > self.timeout:
                raise TimeoutError("Timeout: the recipient did not confirm the RGB values in time.")

            # Pause for a short period to prevent CPU overload
            time.sleep(self.retry_delay)
            
    def confirm_RGBvals(self):
        start_time = time.time()
        
        #wait for command
        while True:
            last_line = self.extract_last_line()
            last_word = self.extract_last_word_in_file(last_line = last_line)
            is_image_display = self.check_last_word_in_file("Image_Display", last_word = last_word)
            is_done = self.check_last_word_in_file("Done", last_word = last_word)
            if is_done:
                self.terminate = True
                break
            
            if is_image_display:
                # extract the RGB value
                RGBvals = self.extract_rgb_values(last_line)
                # Create the message to indicate the current stimulus to display
                message_image_for_display = f"R{RGBvals[0]:.4f}_G{RGBvals[1]:.4f}_B{RGBvals[2]:.4f} Image_Confirmed"
                # Append the message to the file
                self.append_message_to_file(message_image_for_display)
                break
            
            # Check if the timeout duration has been exceeded
            if time.time() - start_time > self.timeout:
                raise TimeoutError("Timeout: the sender did not send out the RGB values in time.")

            # Pause for a short period to prevent CPU overload
            time.sleep(self.retry_delay)
            
    def finalize(self):
        """
        Appends a message to the file indicating that the sequence is done.
        """
        self.append_message_to_file("Done")
        self.terminate = True
            