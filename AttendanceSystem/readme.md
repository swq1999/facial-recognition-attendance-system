# Facial Recognition Attendance System
Welcome to the Facial Recognition Attendance System! This application utilizes cutting-edge facial recognition technology to streamline attendance tracking processes. 

## Introduction

The facial recognition attendance system is designed to simplify and enhance the attendance tracking process in various settings, including workplaces, schools, and organizations. By utilizing advanced facial recognition techniques, the system offers a seamless and efficient way to monitor attendance, ensuring accuracy and reliability.

## Key Features

- **Facial Detection**: The system utilizes a pre-trained Haar cascade for accurate detection of human faces in images.
- **Recognition Model**: A trained CNN is employed for facial recognition, enabling the system to identify individuals with high accuracy.
- **User-Friendly Interface**: The user interface `app.py` provides an intuitive platform for administrators to manage attendance records and monitor real-time data.
- **Check-in and Check-out**: The system allows users to conveniently check-in and check-out, providing a seamless attendance tracking experience.
- **Database Integration**: `DatabaseFunctions.py` facilitates seamless communication with the database, ensuring efficient storage and retrieval of attendance data.
- **Employee Addition**: The system includes functionality for adding new employees to the database, enabling easy management of personnel records.
- **Data Collection**: `DataCollection.py` provides functionalities for capturing user images during the employee addition process, ensuring comprehensive data collection for facial recognition training.

## Dependencies Installation

To install the required dependencies, follow these steps:

1. Make sure you have **Python** installed on your system. 
2. Navigate to the main directory where `requirements.txt` is located.
3. Run the following command in the terminal to install all the dependencies:

```bash
pip install -r requirements.txt
```

## Getting Started
To run the application, follow these simple steps:

1. Make sure you have Python installed on your system.
2. Navigate to the main directory where `app.py` is located.
3. Run the application by executing the following command in the terminal:

```bash
py app.py
```

## File Information
- `app.py`: This file contains the user interface elements for the application.
- `classifier.py`: This file contains the core functionality for facial recognition and attendance tracking.
- `databaseFunctions.py`: Contains functionalities to enable database communation.
- `dataCollection.py`: A script to capture user images during employee addition.
- `decoupled_app.py`: Contains decoupled functions from  `app.py` for functional and unit testing.
- `labels.json`: Consists of CNN model classes.
- `test_app.py`: Contains tests done to functions in `app.py`.
- `test_databaseFunctions.py`: Contains tests done to functions in `databaseFunctions.py`.
- `training.ipynb`: Python Notebook for training the CNN model. 
- `training.py`: Script to train the model from scratch after a new employee has been added.
- `webScraper.py`: Script that scrapes images from the internet.
- `transferLearning`: A script that was supposed to be used to train the CNN model containing previous employee data with new employee data. 



