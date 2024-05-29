"""
Facial Recognition Attendance System

A user interface and an entry point 
into the application. 

Author: Syed Wajahat Quadri
ID: w21043564
"""

import os
import customtkinter as ctk
from PIL import Image
from classifier import classify
from dataCollection import collectData

def main():
    """
    Main function to start the application.
    """
    app = MainWindow()
    app.mainloop()

class MainWindow(ctk.CTk):
    """
    Main application window for the Attendance System.
    """
    def __init__(self):
        super().__init__()
        self.geometry("500x360")
        self.resizable(False, False)
        self.wm_title("Attendance System")

        # Grid
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Fonts
        my_font = ("Helvetica", 14)

        # Image
        self.pic = ctk.CTkImage(dark_image=Image.open("assets/camera.jpg"), size=(480, 300))

        # Buttons
        self.image = ctk.CTkButton(
            self,
            text=None,
            fg_color="black",
            hover_color="black",
            height=400,
            corner_radius=0,
            border_spacing=8,
            image=self.pic,
        )
        self.start = ctk.CTkButton(
            self,
            text="Start",
            font=my_font,
            width=153,
            height=32,
            command=self.start_command,
        )
        self.add = ctk.CTkButton(
            self,
            text="Add Employee",
            font=my_font,
            width=153,
            height=32,
            command=self.addEmployee,
        )
        self.exit = ctk.CTkButton(
            self,
            text="Exit",
            font=my_font,
            width=153,
            height=32,
            command=self.close,
        )

        # Add buttons to the grid
        self.image.grid(
            row=0, column=0, columnspan=3, padx=10, pady=10, sticky="nsew"
        )
        self.start.grid(
            row=1, column=0, columnspan=1, padx=(10,5), pady=(0,10), sticky="nsew"
        )
        self.add.grid(
            row=1, column=1, columnspan=1, padx=(5,5), pady=(0,10), sticky="nsew"
        )
        self.exit.grid(
            row=1, column=2, columnspan=1, padx=(5,10), pady=(0,10), sticky="nsew"
        )

    def start_command(self):
        """
        Command to run the face classifier.
        """
        classify()
        
    def addEmployee(self):
        """
        Open an "add employee" popup window.
        """
        popup = Employee(self)
        popup.grab_set()  
        popup.attributes('-topmost', True)  
        popup.lift()  
        popup.focus_force() 

    def close(self):
        """
        Close the application.
        """
        self.quit()
          
class Employee(ctk.CTkToplevel):
    """
    Popup window for employee addition.
    """
    def __init__(self, parent):
        super().__init__(parent)

        self.geometry("300x160")
        self.wm_title("Add Employee")
        self.resizable(False, False)

        # Fonts
        my_font = ("Helvetica", 14)

        # label
        self.label = ctk.CTkLabel(self, text="Employee Name:", font=my_font, width=135)
        self.label.grid(row=0, column=0, columnspan=1, padx=(10, 5), pady=(10, 5))
        
        # Text Field
        self.entry = ctk.CTkEntry(self, width=135)
        self.entry.grid(row=0, column=1, columnspan=1, padx=(5, 10), pady=(10, 5))
        
        # Checkbox for user agreement
        self.agree_var = ctk.IntVar()
        self.agree_checkbox = ctk.CTkCheckBox(self, text="I agree to the terms and conditions.", variable=self.agree_var)
        self.agree_checkbox.grid(row=1, column=0, columnspan=2, padx=(10, 10), pady=(5, 5))
        
        # Error message label
        self.error_message = ctk.CTkLabel(self, text="", font=("Helvetica", 12), text_color="white", wraplength=280)
        self.error_message.grid(row=2, column=0, columnspan=2, padx=(10, 10), pady=(5, 5))

        # "Next" button
        self.next_button = ctk.CTkButton(self, width=135, height=32, text="Next", font=my_font, command=self.next)
        self.next_button.grid(row=3, column=0, columnspan=1, padx=(10, 5), pady=(5, 10))

        # "Cancel" button
        self.cancel_button = ctk.CTkButton(self, width=135, height=32, text="Cancel", font=my_font, command=self.cancel)
        self.cancel_button.grid(row=3, column=1, columnspan=1, padx=(5, 10), pady=(5, 10))

    def show_error(self, message, color):
        """
        Display an error message with the specified color.
        
        Parameters:
        message (str): The error message to display.
        color (str): The color of the error message.
        """
        self.error_message.configure(text=message, text_color=color)
        self.after(3000, lambda: self.error_message.configure(text=""))

    def next(self):
        """
        Validate employee information and create a new employee directory.
        """
        # Check if user agreed to the terms
        if not self.agree_var.get():
            self.show_error("You must agree to the terms and conditions to proceed.", "red")
            return
        
        # Presence check
        employee_name = self.entry.get()
        if not employee_name:
            self.show_error("Employee name cannot be empty.", "red")
            return
        
        # Length check
        min_length = 4
        max_length = 20
        name_length = len(employee_name)
        if name_length < min_length or name_length > max_length:
            self.show_error("Employee name must be between {} and {} characters long.".format(min_length, max_length), "red")
            return
        
        # Check if employee name already exists in the directory
        employees_directory = "employees"
        existing_names = os.listdir(employees_directory)
        if employee_name in existing_names:
            self.show_error("Employee name already exists.", "red")
            return
        
        employee_directory_path = os.path.join(employees_directory, employee_name)
        try:
            os.makedirs(employee_directory_path)
            self.show_error("Employee folder created successfully.", "green")
            self.grab_release()
            self.destroy()
            collectData(employee_directory_path)
        except FileExistsError:
            self.show_error("Employee folder already exists.", "red")

    def cancel(self):
        """
        Cancel the operation and close the popup window.
        """
        self.grab_release()
        self.destroy()
    
if __name__ == "__main__":
    main()
