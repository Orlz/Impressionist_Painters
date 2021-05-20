#!/usr/bin/env python

""" 
==========================================================
UTILITIES SCRIPT: LeNet CNN tools for Image Classification 
==========================================================

This script contains 9 functions necessary for running the CNN_artists.py script. 

Some of these functions have been developed for use in class while the majority have been created by our group when completing 
the assignment. They are versatile functions which may be of use to other CNN image classifiers using tensorflow.

"""

"""
------------
Dependencies
------------
"""
#connecting to the image directory 
import os
import sys
sys.path.append(os.path.join(".."))
import glob


#image data manipulation
import cv2
import numpy as np
from tqdm import tqdm
from contextlib import redirect_stdout

#creating plots
import matplotlib.pyplot as plt

#sklearn
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

# TensorFlow tools
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, 
                                     MaxPooling2D, 
                                     Activation, 
                                     Flatten, 
                                     Dense)
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K


"""
------------------------------------------------
1. Function for extracting labels from filenames 
------------------------------------------------
"""

def listdir_nohidden(path):
    """
    This function extracts the label names by listing the names of folders within the training directory. 
    It does not list the names of hidden files which begin with a fullstop (.)  
    """
    # Create empty list
    label_names = []
    
    # For every name in training directory
    for name in os.listdir(path):
        # If it does not start with . (which hidden files do)
        if not name.startswith('.'):
            label_names.append(name)
            
    return label_names

"""

------------------------------------------------
2. Function for determining the image dimensions
------------------------------------------------
"""

def find_image_dimensions(train_data, test_data, label_names):
    """
    This function has a simple use - to find the minimum width and heigh of both the training and test data. 
    These minimum values can then be used as the minimum values for resizing after. 
    """
    # Create empty lists
    heights_train = []
    widths_train = []
    heights_test = []
    widths_test = []
    
    # Loop through the directories for each painter
    for name in label_names:
        
        # Take images in train data which are .jpg files
        train_images = glob.glob(os.path.join(train_data, name, "*.jpg"))
        
        # Loop through images in training data
        for image in train_images:
            # Load image
            loaded_img = cv2.imread(image)
            
            # Find dimensions of each image
            height, width, _ = loaded_img.shape
        
            # Append to lists
            heights_train.append(height)
            widths_train.append(width)
        
        # Take images in test data
        test_images = glob.glob(os.path.join(test_data, name, "*.jpg"))
        
        # Loop through images in test data
        for image in test_images:
        
            # Load image
            loaded_img = cv2.imread(image)
        
            # Find dimensions of each image
            height, width, _ = loaded_img.shape
        
            # Append to lists
            heights_test.append(height)
            widths_test.append(width)
            
    # Find the smallest image dimensions among all images 
    """
    We're choosing to use a square and so the minimum is found across both height and width of both sets
    """
    min_height = min(heights_train + heights_test + widths_train + widths_test)
    min_width = min(heights_train + heights_test + widths_train + widths_test)
    
    return min_height, min_width


"""
--------------------------------------------
3. Function to create the training data sets 
--------------------------------------------
"""
def create_trainX_trainY(train_data, min_height, min_width, label_names):
    """
    The function creates the trainX and trainY sets to be as follows:
    trainX = training data 
    trainY = training labels  
    """
    # Create empty array and list
    trainX = np.empty((0, min_height, min_width, 3))
    trainY = []
    
    # Loop through images in training data
    for name in label_names:
        images = glob.glob(os.path.join(train_data, name, "*.jpg"))
        
        # For each image
        for image in tqdm(images):
        
            # Load image
            loaded_img = cv2.imread(image)
        
            # Resize image with the specified dimensions using cv2's resize function
            resized_img = cv2.resize(loaded_img, (min_width, min_height), interpolation = cv2.INTER_AREA)
        
            # Create array of image
            image_array = np.array([np.array(resized_img)])
        
            # Append the image array to the trainX
            trainX = np.vstack((trainX, image_array))
            
            # Append the label name to the trainY list
            trainY.append(name)
        
    return trainX, trainY


"""
----------------------------------------
4. Function to create the test data sets 
----------------------------------------
"""
def create_testX_testY(test_data, min_height, min_width, label_names):
    """
    The function creates the testX and testY sets to be as follows:
    testX = validation data 
    testY = validation labels  
    """
    # Create empty array and list
    testX = np.empty((0, min_height, min_width, 3))
    testY = []
    
    # Loop through images in test data
    for name in label_names:
        images = glob.glob(os.path.join(test_data, name, "*.jpg"))
    
    # For each image
        for image in tqdm(images):
        
            # Load image
            loaded_img = cv2.imread(image)
        
            # Resize image
            resized_img = cv2.resize(loaded_img, (min_width, min_height), interpolation = cv2.INTER_AREA)
        
            # Create array
            image_array = np.array([np.array(resized_img)])
        
            # Append the image array to the testX
            testX = np.vstack((testX, image_array))
            # Append the label name to the testY list
            testY.append(name)
        
    return testX, testY

"""
-----------------------------
5. Normalizing and Binarizing 
-----------------------------
"""

def normalize_binarize(trainX, trainY, testX, testY):
    """
    This function applies normalization to the training and test data (trainX and testX)
    It also applies binarization to the training and test labels so they can be used in the model (trainY and testY)
    """
    
    # Normalize training and test data
    trainX_norm = trainX.astype("float") / 255.
    testX_norm = testX.astype("float") / 255.
    
    # Binarize training and test labels
    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.fit_transform(testY)
    
    return trainX_norm, trainY, testX_norm, testY


"""
-----------------------------
6. Creating the LeNet model 
-----------------------------
"""
def define_LeNet_model(min_width, min_height):
    """
    This function defines the LeNet model architecture. 
    It saves this as both a txt and png file in the output folder. 
    
    The LeNet model contains the following layers: 
    INPUT => CONV => ReLU => MAXPOOL => CONV => ReLU => MAXPOOL => FC => ReLU => FC
    """
    # Define model
    model = Sequential()

    # Add the first convolutional layer, ReLU activation function, and pooling layer
    # Convolutional layer (3x3 kernal)
    model.add(Conv2D(32, (3, 3), 
                     padding="same",  # adding a layer of padding with zeros
                     input_shape=(min_height, min_width, 3)))
    
    # Activation function
    model.add(Activation("relu"))
    
    # Max pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2), 
                           strides=(2, 2))) # stride of 2 horizontal, 2 vertical
    
    # Add second convolutional layer, ReLu activation function, and pooling layer
    # Convolutional layer
    model.add(Conv2D(50, (5, 5), 
                     padding="same"))
    
    # Activation function
    model.add(Activation("relu"))
    
    # Max pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2), 
                           strides=(2, 2))) #Again using a stride of 2 horizontal, 2 vertical
    
    # Add fully-connected layer
    model.add(Flatten()) # flattening layer 
    model.add(Dense(500)) # dense network with 500 nodes
    model.add(Activation("relu")) # activation function
    
    # Add output layer
    # softmax classifier
    model.add(Dense(10)) # dense layer of 10 nodes used to classify the images
    model.add(Activation("softmax"))

    # Define optimizer algorithm
    opt = SGD(lr=0.01)
    
    # Compile model
    model.compile(loss="categorical_crossentropy", 
                  optimizer=opt, 
                  metrics=["accuracy"])
    
    # Model summary
    model_summary = model.summary()
    
    # name for saving model summary
    model_path = os.path.join("..", "out", "model_summary.txt")
    # Save model summary
    with open(model_path, 'w') as f:
        with redirect_stdout(f):
            model.summary()
    

    # name for saving plot
    plot_path = os.path.join("..", "out", "LeNet_model.png")
    # Visualization of model
    plot_LeNet_model = plot_model(model,
                                  to_file = plot_path,
                                  show_shapes=True,
                                  show_layer_names=True)
    print(f"\n[INFO] Model architecture is saved as txt in '{model_path}' and as png in '{plot_path}'.")
    
    return model


"""
----------------------
7. Training the model 
----------------------
"""
def train_LeNet_model(model, trainX, trainY, testX, testY, n_epochs, batch_size):
    """
    Training the LeNet model on the training data and validating it on the test data.
    """
    # Train model
    H = model.fit(trainX, trainY, 
                  validation_data=(testX, testY), 
                  batch_size=batch_size, 
                  epochs=n_epochs, verbose=1)
    
    return H

"""
------------------------
8. Evaluating the model 
------------------------
"""
    
def evaluate_model(model, testX, testY, batch_size, label_names):
    """
    This function evaluates the trained model and saves the classification report in the out folder.
    """
    # Predictions
    predictions = model.predict(testX, batch_size=batch_size)
    
    # Classification report
    classification = classification_report(testY.argmax(axis=1),
                                                  predictions.argmax(axis=1),
                                                  target_names=label_names)
            
    # Print classification report to terminal
    print(classification)
    
    # name for saving report
    report_path = os.path.join("..", "out", "classification_report.txt")
    
    # Save classification report
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(classification_report(testY.argmax(axis=1),
                                                  predictions.argmax(axis=1),
                                                  target_names=label_names))
    
    print(f"\n[INFO] Classification report is saved as '{report_path}'.")


"""
------------------------
9. Plotting the results  
------------------------
"""

def plot_history(H, n_epochs):
    """
    This function plots the loss/accuracy of the model during training and saves this as png file in the out folder.
    It uses matplotlib tools to create the plot.
    """
    # name for saving output
    figure_path = os.path.join("..", "out", "model_history.png")
    # Visualize performance
    plt.style.use("fivethirtyeight")
    plt.figure()
    plt.plot(np.arange(0, n_epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, n_epochs), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, n_epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, n_epochs), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path)
    
    print(f"\n[INFO] Loss and accuracy across on training and validation is saved as '{figure_path}'.")