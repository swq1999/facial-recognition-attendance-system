"""
Classifier to detect and recognize
faces.

Author: Syed Wajahat Quadri
ID: w21043564
"""

import cv2
import json
import numpy as np
from keras.models import load_model
from datetime import datetime
import time
from databaseFunctions import insert_login, insert_logout

def main():
    """
    Main function to start the classification process.
    """
    classify()

def load_class_names():
    """
    Loads class names from a JSON file.

    Returns:
    dict: A dictionary mapping class indices to class names.
    """
    with open('labels.json', 'r') as file:
        class_names = json.load(file)
    return class_names

def classify():
    """
    Perform real-time face detection and recognition using a webcam.
    """
    # Load background
    background = cv2.imread('assets/bg.png')
    
    # Load the pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Load the trained face recognition model
    model = load_model('models/model.keras')

    # Load class names
    class_names = load_class_names()

    # Open the camera
    cap = cv2.VideoCapture(0)  # Use 0 for default camera
    cap.set(3, 480)
    cap.set(4, 400)

    # Counter for consecutive detections
    consecutive_detections = 0
    detection_limit = 10

    # Flags
    face_detect = False
    show_greeting = False
    login = False
    
    while True:
        # Capture frames
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Unable to capture frame")
            break

        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Define input height and width
        height, width = 96, 96

        # Reset the background image to avoid overlaying text from previous iterations
        display_frame = background.copy()

        # Draw green rectangle around detected faces and recognize them
        for (x, y, w, h) in faces:
            # Extract the face region
            face = frame[y:y+h, x:x+w]
            
            # Preprocess the face for the model
            face_resized = cv2.resize(face, (height, width)) 
            face_array = np.array(face_resized) / 255.0 
            face_array = np.expand_dims(face_array, axis=0) 
            
            if face_detect:
                # Predict the identity
                prediction = model.predict(face_array)
                predicted_class = np.argmax(prediction, axis=1)[0]
                confidence = np.max(prediction)  # Get the confidence score
                
                # Check for consecutive detections
                if confidence >= 0.75: 
                    name = class_names[predicted_class]
                    confidence_text = f"{name} ({confidence * 100:.2f}%)"
                    consecutive_detections += 1
                    
                    if consecutive_detections == detection_limit:
                        # reset variables
                        consecutive_detections = 0
                        show_greeting = True
                        face_detect = False
                        # Add texts
                        confidence_display = f"{confidence * 100:.1f}%"
                        # Set time
                        current_time = time.localtime()
                        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", current_time)
                        
                        # Add to database
                        if login:
                            greeting_text = f"Hi {name}!"
                            insert_login(name)
                        else:
                            greeting_text = f"Bye {name}!"
                            insert_logout(name)
                else:
                    # If model not confident
                    confidence_text = "Unknown"
                    consecutive_detections = 0
                    
                # Draw face rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Calculate the size of the text bounding box
                text_size, _ = cv2.getTextSize(confidence_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                
                # Set the position of the background rectangle
                text_x, text_y = x, y - 10
                background_rect_start = (text_x, text_y - text_size[1])
                background_rect_end = (text_x + text_size[0], text_y + 5)
                
                # Draw background over rectangle
                cv2.rectangle(frame, background_rect_start, background_rect_end, (0, 0, 0), cv2.FILLED)
                
                # Write text on background
                cv2.putText(frame, confidence_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)

        if len(faces) == 0 and face_detect is True:
            cv2.putText(display_frame, "Face Not Found", (40, 505), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            show_greeting = False

        if show_greeting:
            cv2.putText(display_frame, greeting_text, (35, 485), cv2.FONT_HERSHEY_SIMPLEX, 1, (36, 255, 12), 2)
            cv2.putText(display_frame, confidence_display, (580, 485), cv2.FONT_HERSHEY_SIMPLEX, 1, (36, 255, 12), 2)
            cv2.putText(display_frame, formatted_time, (35, 525), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
        # Get the dimensions of the frame
        frame_height, frame_width, _ = frame.shape

        # Display the frame
        display_frame[44:44+frame_height, 33:33+frame_width] = frame
        cv2.imshow('Attendance System', display_frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Press 'i' to login
        if cv2.waitKey(1) & 0xFF == ord('i'):
            face_detect = True
            show_greeting = False
            login = True
        
        # Press 'o' to logout
        if cv2.waitKey(1) & 0xFF == ord('o'):
            face_detect = True
            show_greeting = False
            login = False

        # Press 's' to stop
        if cv2.waitKey(1) & 0xFF == ord('s'):
            face_detect = False
            show_greeting = False

    # Release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
