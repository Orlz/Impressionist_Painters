#!/usr/bin/env python

"""
This script builds a LeNet Convolutional Neural Network (CNN) to predict which artist a painting belongs to within a collection of 10 impressionist painters. 

The script tries to structure the code in such a way that it is built to run for this context of impressionist paintings, but could be easily adapted into a new context. 
To support this process, a number of the functions have been stored in the CNN_utils.py script, found in the utils folder. 

The script therefore calls many of the functions to be run here, but keeps the focus of the script on building the model. 

Parameters: 
  -t  --path2train:  <str> "Path to where the training data is stored" 
  -te --path2test:   <str> "Path to where the validation data is stored" 
  -n  --n_epochs:    <int> "Number of epochs to train the model on" 
  -b  --batch_size:  <int> "The size of the batch to train the model on"
  
Usage: 

$ python3 src/CNN_artists.py 

""" 


"""
=======================
Import the Dependencies
=======================

"""
# Operating system
import os
import sys
sys.path.append(os.path.join(".."))

# Data handling tools
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import glob
from contextlib import redirect_stdout

#Functions from the utils folder
import utils.CNN_utils as functions

#Commandline functionality 
import argparse

# Sklearn tools
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
==================
Argparse Arguments
==================

"""
# Initialize ArgumentParser class
ap = argparse.ArgumentParser()
    
# Argument 1: Path to training data
ap.add_argument("-t", "--path2train",
                type = str,
                required = False,
                help = "Path to the training data",
                default = "../data/training")
    
# Argument 2: Path to test data
ap.add_argument("-te", "--path2test",
                type = str,
                required = False,
                help = "Path to the test/validation data",
                default = "../data/validation")
    
# Argument 3: Number of epochs
ap.add_argument("-n", "--n_epochs",
                type = int,
                required = False,
                help = "The number of epochs to train the model on",
                default = 20)
    
# Argument 4: Batch size
ap.add_argument("-b", "--batch_size",
                type = int,
                required = False,
                help = "The size of the batch on which to train the model",
                default = 32)
    
# Parse arguments
args = vars(ap.parse_args()) 


"""
=============
Main Function
=============

"""
def main():
    
    """
    Create variables with the input parameters
    """
    train_data = args["path2train"]
    test_data = args["path2test"]
    n_epochs = args["n_epochs"]
    batch_size = args["batch_size"] 
    
    
    """
    Create the out directory, if it doesn't already exist 
    """
    dirName = os.path.join("..", "out")
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        
        # print that it has been created
        print("Directory " , dirName ,  " Created ")
    else:   
        # print that it exists
        print("Directory " , dirName ,  " already exists")

    
    """
    ==============
    Preprocessing
    ==============
    """
    
    print("\n Hi there! Are you ready to start classifying some impressionist paintings? \n I hope so, it's exciting stuff!")
    print("\n I'm about to initialize the construction of your LeNet convolutional neural network model...")     
    print("\n We'll start with the pre-processing. This might take a few minutes.") 
    
    """
    Labelling the data
    """
    # Create the list of label names
    label_names = functions.listdir_nohidden(train_data)
      
    """
    Resizing the images 
    """
    # Find the optimal dimensions to resize the images 
    print("\n[INFO] Estimating the optimal image dimensions to resize images...")
    min_height, min_width = functions.find_image_dimensions(train_data, test_data, label_names)
    print(f"\n[INFO] Input images are resized to dimensions of height = {min_height} and width = {min_width}...")
    
    
    # Training data: Resize and create trainX and trainY
    print("\n[INFO] Resizing training images and creating training data (trainX), and labels (trainY)...")
    trainX, trainY = functions.create_trainX_trainY(train_data, min_height, min_width, label_names)
    
    # Validation data: Resize and create testX and testY
    print("\n[INFO] Resizing validation images and creating validation data (testX), and labels (testY)...")
    testX, testY = functions.create_testX_testY(test_data, min_height, min_width, label_names)
    
    
    """
    Normalizing and Binarizing 
    """
    # Normalize the data and binarize the labels
    print("\n[INFO] Normalize training and validation data and binarizing training and validation labels...")
    trainX, trainY, testX, testY = functions.normalize_binarize(trainX, trainY, testX, testY)
    
    
    """
    ===============================
    Building and training the model
    ===============================
    """
    
    #We build the model here so that we can see the layers we're building into it
    print("\n[INFO] Preprocessing complete. I'm now defining the LeNet model architecture as follows...")
    print("\n INPUT => CONV => ReLU => MAXPOOL => CONV => ReLU => MAXPOOL => FC => ReLU => FC") 

    
    #Run the model
    model = define_LeNet_model(min_width, min_height)
    
    # Train model
    print("\n[INFO] The model's ready so we'll begin training it...\n\n")
    H = functions.train_LeNet_model(model, trainX, trainY, testX, testY, n_epochs, batch_size)
    
    print("\nTraining complete - thanks for your patience! We'll start to evaluate the model's performance") 
    
    
    """
    ====================
    Evaluating the model
    ====================
    """

    # Plot loss/accuracy history of the model
    functions.plot_history(H, n_epochs)
    
    # Evaluate model
    print("\n[INFO] Below is the classification report. This has been copied into the out directory\n")
    functions.evaluate_model(model, testX, testY, batch_size, label_names)
    
    # User message
    print("\n That's you all done - woohoo!\nYou have now defined and trained a CNN on impressionist paintings which is able to classify paintings by their artists.!")
    
    
    

"""
===================================
Function used to define LeNet Model
===================================

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
    # Convolutional layer (5x5 kernal)
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

    
# Close the main function 
if __name__=="__main__":
    main()  
