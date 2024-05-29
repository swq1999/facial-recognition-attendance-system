"""
Image collection script for new employees.

Author: Syed Wajahat Quadri
ID: w21043564
"""

import cv2
import os
import shutil
from training import train

def main():
    # Test
    collectData("employees/test")

def collectData(path):
    """
    Collects images from the webcam and 
    saves them to the specified directory.

    Parameters:
    path (str): The directory path where the collected images will be saved.
    """
    # Create a directory
    save_dir = path
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    # Initialize image count
    count = 0

    # Total number of images to capture
    total_images = 200

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break
        
        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)

        # Display image counter
        counter_text = f"{count+1}/{total_images}"
        cv2.putText(frame, counter_text, (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (36, 255, 12), 2)

        # Display instructions
        instructions_capture = "c - capture"
        instructions_quit = "q - quit"
        cv2.putText(frame, instructions_capture, (15, frame.shape[0] - 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, instructions_quit, (15, frame.shape[0] - 15), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)

        # Display the frame
        cv2.imshow('Frame Capture', frame)

        # Wait for key press
        key = cv2.waitKey(1) & 0xFF

        # Capture image when 'c' is pressed
        if key == ord('c'):
            img_name = os.path.join(save_dir, f'frame_{count}.jpg')
            cv2.imwrite(img_name, frame)
            count += 1
            print(f"Saved {img_name}")

            # Break the loop if image limit reached
            if count >= total_images:
                print(f"Collected {count} images.")
                cv2.putText(frame, "Please wait, Training...", (150, frame.shape[0] - 180), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
                train()
                break

        # Press 'q' to exit
        if key == ord('q'):
            # Delete the folder
            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)
                print(f"Deleted folder: {save_dir}")
            break

    # Release webcam and close window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
