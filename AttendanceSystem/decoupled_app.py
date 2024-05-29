"""
Decoupled functions from the next() 
function in Employee Class in 'app.py'.

Author: Syed Wajahat Quadri
ID: w21043564
"""

import os

def check_agreement(agree_var):
    """
    Checks if the user has agreed to the terms and conditions.
    """
    if not agree_var:
        return "You must agree to the terms and conditions to proceed.", "red"
    return None

def check_presence(employee_name):
    """
    Checks if the employee name is provided.
    """
    if not employee_name:
        return "Employee name cannot be empty.", "red"
    return None

def check_length(employee_name, min_length=4, max_length=20):
    """
    Checks if the employee name length is within the specified range.
    """
    name_length = len(employee_name)
    if name_length < min_length or name_length > max_length:
        return f"Employee name must be between {min_length} and {max_length} characters long.", "red"
    return None

def check_existing_employee(employee_name, employees_directory="employees"):
    """
    Checks if the employee name already exists in the directory.
    """
    existing_names = os.listdir(employees_directory)
    if employee_name in existing_names:
        return "Employee name already exists.", "red"
    return None

def create_employee_directory(employee_name, employees_directory="employees"):
    """
    Creates a directory for the new employee.
    """
    employee_directory_path = os.path.join(employees_directory, employee_name)
    try:
        os.makedirs(employee_directory_path)
        return "Employee folder created successfully.", "green"
    except FileExistsError:
        return "Employee folder already exists.", "red"
