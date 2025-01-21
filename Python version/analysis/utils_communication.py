# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 20:55:45 2025

@author: Fangfang
"""

import time
from datetime import datetime
import dill as pickled
import socket
import os
import numpy as np
import tkinter as tk

#%%
def get_experiment_info_custom():
    """
    Create a customized popup window to collect subject_id, subject_init, and session_today.
    
    Returns:
        tuple: subject_id (int), subject_init (str), session_today (int)
    """
    # Function to submit the data
    def submit():
        try:
            # Validate and retrieve data
            subject_id = int(subject_id_entry.get())
            subject_init = subject_init_entry.get().strip()
            session_today = int(session_today_entry.get())
            
            # Ensure no fields are empty
            if not subject_init:
                raise ValueError("Subject initials cannot be empty.")
            
            # Update result and close the window
            result["subject_id"] = subject_id
            result["subject_init"] = subject_init
            result["session_today"] = session_today
            popup.destroy()
        except ValueError as e:
            error_label.config(text=f"Error: {e}", fg="red")

    # Create the main popup window
    popup = tk.Tk()
    popup.title("Enter Experiment Info")
    popup.geometry("400x300")  # Set window size

    # Configure font size
    label_font = ("Arial", 14)
    entry_font = ("Arial", 12)
    button_font = ("Arial", 12)
    
    # Create labels and entry fields
    tk.Label(popup, text="Enter Subject ID:", font=label_font).pack(pady=5)
    subject_id_entry = tk.Entry(popup, font=entry_font)
    subject_id_entry.pack(pady=5)
    
    tk.Label(popup, text="Enter Subject Initials:", font=label_font).pack(pady=5)
    subject_init_entry = tk.Entry(popup, font=entry_font)
    subject_init_entry.pack(pady=5)
    
    tk.Label(popup, text="Enter Today's Session Number:", font=label_font).pack(pady=5)
    session_today_entry = tk.Entry(popup, font=entry_font)
    session_today_entry.pack(pady=5)
    
    # Error message label
    error_label = tk.Label(popup, text="", font=("Arial", 10))
    error_label.pack(pady=5)
    
    # Submit button
    tk.Button(popup, text="Submit", command=submit, font=button_font).pack(pady=10)
    
    # Center window
    popup.eval('tk::PlaceWindow . center')
    
    # Initialize result dictionary
    result = {}
    
    # Run the Tkinter event loop
    popup.mainloop()
    
    # Return the collected data
    if result:
        return result["subject_id"], result["subject_init"], result["session_today"]
    else:
        return None, None, None  # Return None values if the window was closed

#%% create a new file
class ExperimentFileManager:
    def __init__(self, subject_id, subject_init, networkDisk_path):
        """
        Initialize the file manager for a specific subject.
        
        Args:
            subject_id (str): Unique identifier for the subject.
            networkDisk_path (str): Base directory where files are stored.
        """
        self.subject_id = subject_id
        self.subject_init = subject_init
        self.networkDisk_path = networkDisk_path
        self._check_networkDisk_path()
        
        #create a path just for that subject
        #create subject directory if not exists
        self.path_sub = os.path.join(self.networkDisk_path, f'sub{subject_id}')
        os.makedirs(self.path_sub, exist_ok = True)
        
        self.session_data = {}  # Dictionary to store session metadata
        self.pickle_file = os.path.join(self.path_sub, f"sub{subject_id}_expt_record.pkl")
    
    def _check_networkDisk_path(self):
        # Check if the path exists
        if os.path.exists(self.networkDisk_path):
            print(f"The path exists: {self.networkDisk_path}")
        else:
            raise ValueError(f"The path does not exist: {self.networkDisk_path}")
            
    def _check_init_consistency(self):
        # Validate that subject initials match across history
        for session in self.session_data.values():
            if session["sub_initial"] != self.subject_init:
                raise ValueError(
                    f"Mismatch in subject initials: Found '{session['sub_initial']}' in history, "
                    f"but current subject initials are '{self.subject_init}'. Ensure consistency."
                )
    
    def _validate_session_num(self, session_num):
        # Retrieve past session numbers
        past_session_num = list(self.session_data.keys())
        
        # Validate session number
        if session_num < 1:
            raise ValueError("Session number must be larger than 1.")
        
        if not past_session_num:  # No previous session numbers
            if session_num != 1:
                raise ValueError("The first session must be 1.")
        else:  # There are previous sessions
            if session_num in past_session_num:
                # Check the status of the session
                if self.session_data[session_num]['status'] == 'Done':
                    raise ValueError("This session was already completed in the past.")
                else:
                    pressed_button = input(
                        "There is an existing file for this session,\n"
                        "but the status shows that the session was not completed.\n"
                        "Please confirm that this is true and press Y/N to proceed/stop: "
                    )
                    if pressed_button.lower() != "y":
                        print("Operation cancelled.")
                        return None, None
            elif session_num != (max(past_session_num) + 1):
                raise ValueError(
                    f"Previous session numbers are: {past_session_num}. "
                    f"The next one should be {max(past_session_num) + 1}."
                )
                    
    def create_session_file(self, session_num):
        # Validate that subject initials match across history
        self._check_init_consistency()
        
        # Validate session number
        self._validate_session_num(session_num)
            
        #Generate the file name and path:
        date_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        file_name = f"sub{self.subject_id}_session{session_num}_{date_time}.txt"
        file_path = os.path.join(self.path_sub, file_name)
        
        # Create the file with metadata
        with open(file_path, 'w') as file:
            file.write(f"Subject ID: {self.subject_id}\n")
            file.write(f"Session: {session_num}\n")
            file.write(f"Date and Time: {date_time}\n")
            
        # Update session data
        self.session_data[session_num] = {
            "sub_initial": self.subject_init,
            "file_name": file_name,
            "date_time": date_time,
            "session_number": session_num,
            "sender_path_sub": self.path_sub,
            "status": 'Created'
        }
        
        # Save the updated state
        self.save_state()
        
        print(f"File created and state saved: {file_path}")
        return file_path, file_name
    
    def status_updates(self, status, session_num = None):
        """
        Update the status of the latest session file from the sender's perspective.
        
        Args:
            recipient_status (str): The status to update. Possible values are 
                                    'Confirmed', 'Communicating', or 'Done'.
            session_num (int): default is the latest session
        
        Raises:
            ValueError: If the recipient_status is invalid or if there are no sessions.
        """
        valid_statuses = {'Confirmed', 'Communicating', 'Done'}
        
        # Validate the status
        if status not in valid_statuses:
            raise ValueError(f"Invalid status: {status}. Valid options are {valid_statuses}.")
        
        # Check if there is at least one session
        if not self.session_data:
            raise ValueError("No session data available to update.")
        
        # Determine which session to update
        if session_num is None:
            session_num = max(self.session_data.keys())  # Get the latest session number

        #update the status
        self.session_data[session_num]["status"] = status
        
        # Save the updated state
        self.save_state()
        
        print(f"Updated session {session_num} status to: {status}")
    
    def save_state(self):
        """
        Save the current state of the class as a pickle file.
        """
        with open(self.pickle_file, 'wb') as pkl_file:
            pickled.dump(self, pkl_file)
        print(f"State saved to pickle: {self.pickle_file}")
        
    def list_files(self):
        """
        List all files created for the subject and print them.
        
        Returns:
            list: List of file names.
        """
        file_names = [data["file_name"] for data in self.session_data.values()]
        
        # Print the file names
        print("Files created for the subject:")
        for file_name in file_names:
            print(f"- {file_name}")
        
        return file_names
    
    @staticmethod
    def load_state(pickle_file):
        """
        Load a saved instance of ExperimentFileManager from a pickle file.
        
        Args:
            pickle_file (str): Path to the pickle file.
    
        Returns:
            ExperimentFileManager: Loaded instance.
        """
        with open(pickle_file, 'rb') as pkl_file:
            instance = pickled.load(pkl_file)
        print(f"State loaded from pickle: {pickle_file}")
        return instance


#%%
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
            