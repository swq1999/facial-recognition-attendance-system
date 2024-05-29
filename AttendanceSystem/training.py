"""
Script to train and evaluate the 
CNN model for facial recognition.

Author: Syed Wajahat Quadri
ID: w21043564
"""

import os
import numpy as np
import json
import cv2
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from skimage.io import imread
import imageio
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l2

def main():
    """
    Main function to start model training.
    """
    train()

def preprocess(images_dir, height, width, data, labels, face_cascade, employee, max_images=175):
    """
    Preprocess images by detecting faces, resizing, and saving them.

    Args:
    images_dir (str): Directory containing employee images.
    height (int): Height for resizing images.
    width (int): Width for resizing images.
    data (list): List to store preprocessed image data.
    labels (list): List to store corresponding employee labels.
    face_cascade: Pre-trained face cascade classifier.
    employee (str): Name of the employee.
    max_images (int, optional): Maximum number of images to preprocess. Defaults to 175.
    """
    saved_images = 0  # Counter for saved images
    
    # Employee-specific directory if it doesn't exist
    employee_dir = os.path.join("preprocessedData", employee)
    os.makedirs(employee_dir, exist_ok=True)

    # To stop after reaching max images
    for index, name in enumerate(os.listdir(images_dir)):
        if saved_images >= max_images:
            break
        
        image_path = os.path.join(images_dir, name)
        
        # Read images
        try:
            image = imread(image_path)
        except Exception as e:
            print(f"skimage failed to read {image_path}.")
            try:
                image = imageio.imread(image_path)
            except Exception as e:
                print(f"Failed to read {image_path} with imageio. Skipping.")
                continue
        
        # Process images and detect faces
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        except Exception as e:
            print(f"Error processing {image_path}. Skipping.")
            continue

        # If at least one face is detected, crop the first face found
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face = image[y:y+h, x:x+w]

            # Ensure the face has 3 channels (RGB)
            if face.shape[-1] != 3:
                print(f"Image {image_path} does not have 3 channels. Skipping.")
                continue

            # Resize the face
            face = cv2.resize(face, (height, width))
            
            try:
                # Save the validated image in the employee-specific folder
                save_path = os.path.join(employee_dir, f"{employee}_{index}.jpg")
                cv2.imwrite(save_path, face)
                saved_images += 1  # Increment the counter for saved images
            except Exception as e:
                print(f"Failed to save {image_path}. Skipping.")
                continue
            
            # Save in variables
            data.append(face)
            labels.append(employee)

    print(f"{employee}: {saved_images} images")

def train():
    """
    Train the convolutional neural network (CNN) model.
    """
    # Define variables
    data = []
    labels = []
    height, width = 96, 96
    
    # Load pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Preprocess data
    employees_dir = "employees"
    employees = os.listdir(employees_dir)
            
    # Get images from employee directory and detect faces
    for employee in employees:
        images_dir = os.path.join(employees_dir, employee)
        preprocess(images_dir, height, width, data, labels, face_cascade, employee)
                
    # Saving labels
    with open("labels.json", "w") as file:
        json.dump(sorted(set(labels)), file)
                
    num_classes = len(set(labels))

    # Splitting Data
    X_train, X_test, y_train_names, y_test_names = train_test_split(data, labels, test_size=0.2, random_state=42)
    X_train, X_val, y_train_names, y_val_names = train_test_split(X_train, y_train_names, test_size=0.1, random_state=42)

    # Normalizing pixel values
    X_train = np.array(X_train) / 255.0  
    X_test = np.array(X_test) / 255.0
    X_val = np.array(X_val) / 255.0

    # Mapping employee names to numerical labels
    label_to_index = {label: index for index, label in enumerate(sorted(set(labels)))}
    y_train = np.array([label_to_index[label] for label in y_train_names])
    y_test = np.array([label_to_index[label] for label in y_test_names])
    y_val = np.array([label_to_index[label] for label in y_val_names])

    # Converting numerical labels to categorical format
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)
    y_val = to_categorical(y_val, num_classes=num_classes)

    # Calculating class weights
    class_weights = compute_class_weight(
        class_weight='balanced', 
        classes=np.unique(labels), 
        y=labels
    )

    class_weights_dict = dict(enumerate(class_weights))
    print(class_weights_dict)

    # Defining CNN Model
    model = Sequential([
        Conv2D(128, (3, 3), activation='relu', input_shape=(height, width, 3)),
        MaxPooling2D((2, 2)),
    
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
    
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
                
        Conv2D(512, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
    
        Conv2D(1024, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
    
        Flatten(),
        Dense(2048, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.5),
                
        Dense(num_classes, activation='softmax', name='output_layer')
    ])
    
    # Setting hyperparameters
    epochs = 33
    batch_size = 70
    lrate = 0.0001
        
    # Training the Model
    adam = Adam(learning_rate=lrate)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

    history = model.fit(X_train, y_train, epochs=epochs,
                        batch_size=batch_size, verbose=1,
                        class_weight=class_weights_dict, 
                        validation_data=(X_val, y_val),
                        callbacks=[early_stopping, reduce_lr])

    # Training Results
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
    print("--------------------------x--------------------------x--------------------------")
    print(f"Loss: {test_loss:.4f}")
    print("Accuracy: %.2f%%" % (test_accuracy * 100))

    # Save the model
    model.save("models/model.keras")

    # Predict on test data
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
        
    # Calculate evaluation metrics
    print("Classification Report:")
    print(classification_report(y_test_classes, y_pred_classes))
        
    accuracy = accuracy_score(y_test_classes, y_pred_classes)
    precision = precision_score(y_test_classes, y_pred_classes, average='weighted')
    recall = recall_score(y_test_classes, y_pred_classes, average='weighted')
    f1 = f1_score(y_test_classes, y_pred_classes, average='weighted')
        
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test_classes, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(set(labels)), yticklabels=sorted(set(labels)))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

if __name__ == "__main__":
    main()
