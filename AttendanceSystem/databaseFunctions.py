"""
SQLite Database Functions for Attendance System

Author: Syed Wajahat Quadri
ID: w21043564
"""

import sqlite3
import os
from datetime import datetime

DATABASE_PATH = 'attendance_system.db'

def create_database():
    """
    Create the SQLite database and the Attendance table if they do not exist.
    """
    if not os.path.exists(DATABASE_PATH):
        # Connect to the SQLite database
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        # Create a table if it does not exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Attendance (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                Name TEXT NOT NULL,
                Login TEXT NOT NULL,
                Logout TEXT
            )
        ''')
        
        # Commit and close connection
        conn.commit()
        conn.close()

def insert_login(name):
    """
    Insert a login record for the given user name.

    Parameters:
    name (str): The name of the user logging in.
    """
    # Verify database existance
    create_database()
    
    # Connect to SQLite database
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Get current time
    login_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Insert login record
    cursor.execute('''
        INSERT INTO Attendance (Name, Login)
        VALUES (?, ?)
    ''', (name, login_time))
    
    # Commit and close connection
    conn.commit()
    conn.close()

def insert_logout(name):
    """
    Insert a logout record for the given username 
    by updating the most recent login record.

    Parameters:
    name (str): The name of the user logging out.
    """
    # Verify database existance
    create_database()
    
    # Connect to SQLite database
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Get the current time
    logout_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # # Insert login record for the most recent login
    cursor.execute('''
        SELECT ID FROM Attendance
        WHERE Name = ? AND Logout IS NULL
        ORDER BY ID DESC
        LIMIT 1
    ''', (name,))
    
    # Fetch result
    row = cursor.fetchone()
    if row:
        latest_id = row[0]
        
        # Update the logout time
        cursor.execute('''
            UPDATE Attendance
            SET Logout = ?
            WHERE ID = ?
        ''', (logout_time, latest_id))
    
        # Commit
        conn.commit()

    # Close connection
    conn.close()
