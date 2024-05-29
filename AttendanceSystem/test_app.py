"""
Script to test the decoupled 
functions in 'decoupled_functions.py'

Author: Syed Wajahat Quadri
ID: w21043564
"""

import os
import pytest
from decoupled_app import check_agreement, check_presence, check_length, check_existing_employee, create_employee_directory

@pytest.fixture
def setup_employees_directory(tmp_path):
    """
    Fixture to set up a temporary employees directory for testing.
    """
    employees_directory = tmp_path / "employees"
    employees_directory.mkdir()
    return employees_directory

def test_check_agreement():
    """
    Test the check_agreement function to ensure it correctly identifies
    whether the user has agreed to the terms and conditions.
    """
    assert check_agreement(True) is None
    assert check_agreement(False) == ("You must agree to the terms and conditions to proceed.", "red")

def test_check_presence():
    """
    Test the check_presence function to ensure it correctly identifies
    whether the employee name is provided.
    """
    assert check_presence("John Doe") is None
    assert check_presence("") == ("Employee name cannot be empty.", "red")

def test_check_length():
    """
    Test the check_length function to ensure it correctly identifies
    whether the employee name length is within the specified range.
    """
    assert check_length("John") is None
    assert check_length("A" * 3) == ("Employee name must be between 4 and 20 characters long.", "red")
    assert check_length("A" * 21) == ("Employee name must be between 4 and 20 characters long.", "red")

def test_check_existing_employee(setup_employees_directory):
    """
    Test the check_existing_employee function to ensure it correctly identifies
    whether the employee name already exists in the directory.
    """
    employee_name = "John Doe"
    employees_directory = setup_employees_directory
    (employees_directory / employee_name).mkdir()
    assert check_existing_employee(employee_name, employees_directory) == ("Employee name already exists.", "red")
    assert check_existing_employee("Jane Doe", employees_directory) is None

def test_create_employee_directory(setup_employees_directory):
    """
    Test the create_employee_directory function to ensure it correctly creates
    a new employee directory and handles the case where the directory already exists.
    """
    employee_name = "John Doe"
    employees_directory = setup_employees_directory
    assert create_employee_directory(employee_name, employees_directory) == ("Employee folder created successfully.", "green")
    assert create_employee_directory(employee_name, employees_directory) == ("Employee folder already exists.", "red")
