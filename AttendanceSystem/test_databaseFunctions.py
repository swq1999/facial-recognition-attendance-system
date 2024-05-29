"""
Script to test the functions
in 'databaseFunctions.py'

Author: Syed Wajahat Quadri
ID: w21043564
"""

import os
import sqlite3
import pytest
from datetime import datetime
from databaseFunctions import create_database, insert_login, insert_logout, DATABASE_PATH

TEST_DATABASE_PATH = 'test_attendance_system.db'

# Override the DATABASE_PATH for testing
@pytest.fixture(autouse=True)
def override_database_path(monkeypatch):
    """
    Fixture to override the DATABASE_PATH for testing.
    """
    monkeypatch.setattr('databaseFunctions.DATABASE_PATH', TEST_DATABASE_PATH)
    yield
    if os.path.exists(TEST_DATABASE_PATH):
        os.remove(TEST_DATABASE_PATH)

@pytest.fixture
def setup_database():
    """
    Fixture to set up the database for testing.
    """
    create_database()
    yield
    if os.path.exists(TEST_DATABASE_PATH):
        os.remove(TEST_DATABASE_PATH)

def test_create_database(setup_database):
    """
    Test to check if the database and the 
    Attendance table are created successfully.
    """
    assert os.path.exists(TEST_DATABASE_PATH)

    conn = sqlite3.connect(TEST_DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('''SELECT name FROM sqlite_master WHERE type='table' AND name='Attendance';''')
    table_exists = cursor.fetchone()
    conn.close()

    assert table_exists is not None

def test_insert_login(setup_database):
    """
    Test to check the insert_login function.
    """
    insert_login('John Doe')

    conn = sqlite3.connect(TEST_DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('''SELECT Name, Login, Logout FROM Attendance WHERE Name = 'John Doe';''')
    row = cursor.fetchone()
    conn.close()

    assert row is not None
    assert row[0] == 'John Doe'
    assert row[1] is not None
    assert row[2] is None

def test_insert_logout(setup_database):
    """
    Test to check the insert_logout function.
    """
    insert_login('John Doe')
    insert_logout('John Doe')

    conn = sqlite3.connect(TEST_DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('''SELECT Name, Login, Logout FROM Attendance WHERE Name = 'John Doe';''')
    row = cursor.fetchone()
    conn.close()

    assert row is not None
    assert row[0] == 'John Doe'
    assert row[1] is not None
    assert row[2] is not None
