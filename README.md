Alt-Classifying Impressionist Paintings 
====

## Using a Convolutional Neural Network (CNN) with Tensorflow

This assignment builds upon the topic of classification, moving the methodological efforts from a shallow to deep learning approach. Here, a convolutional neural network (CNN) is built to predict which artist a painting belongs to within a collection of 10 impressionist painters. A key focus of the assignment is looking at how to take an image dataset and prepare it in such a way that it can be fed into the first layer of a CNN. The assignment provides a good foundation to build upon for the last assignment, where we will loop back to apply some of the learning here into the portfolio’s theme of street art and graffiti.

__Background Knowledge__

A CNN is a layered network made up of 3 main types of layers: 
1. Convolutional layers: For feature extraction using complex kernels 
2. Pooling layers: For pooling the convolutional layer results and compressing feature space
3. Activation layers: To optimise the learning by applying algorithms such as the ReLU  

These layers act as an unsupervised form of feature extraction from the data, taking the images in a raw form and transforming them into highly informative feature kernels. There can be a large number of these layers depending on the complexity needed, as a CNN performs better the deeper it is. After running through the defined number of layers, the model takes this densely processed output, flattens it, and feeds it as input into a final fully connected layer. The fully connected layer then performs in a similar fashion to the neural network from assignment02, passing the input through a number of hidden layers to make a classification prediction. In this way, a CNN combines supervised with unsupervised learning to enable to us perform rather complex classification tasks. 

__Assignment Task__
Build a deep learning model using convolutional neural networks which can classify paintings by their respective artists. 

The assignment requires a number of pre-processing steps to be taken, to get the data into a format which can be fed into the CNN. This includes handling the differing sizes of the images and labelling the data in a way which the model will be able to interpret.

We chose to use the LeNet model achitecture. 



Alt-Scripts and Data
---

__Data__

The data used in this assignment is an open-source Kaggle dataset containing images from 10 impressionist painters. The images have already been divided into training and validation sets, with 400 training images and 100 validation images per artist. Due to the large size of the data file (~2GB), the user is required to download the data from [this link](https://www.kaggle.com/delayedkarma/impressionist-classifier-data)and upload it to the repository’s data folder. 

Alternatively, a subset of the dataset has been provided in the data folder which the user is welcome to use. 

__Script__ 

There is just one script to be used in this assignment which can be found in the src folder: CNN_artists.py

There is also a CNN utility script found in the utils folder which contains a number of functions which can be used iteratievly for other CNN's. The decision to include all but the model's function in the utility script was to create a respository and script which could be easily generalised to new contexts. I build upon this in the next assignment. 


Alt-Methods 
---

This problem relates to classifying complex colour images into their respective artist category. The methodological approach took a three-fold approach: 
1. Pre-process the image data 
2. Build and train a model 
3. Evaluate the model’s performance

The pre-processing of the images involved extracting the class labels, determining the image dimensions, resizing all images to fit within these dimensions, converting the images into a stacked NumPy array, normalising the pixel values, and finally binarizing the class labels. 

The model was then built to fit with the LeNet architecture, which has 5 layers (2 convolutional layers, 2 max-pooling layers, and 1 dense fully connected layer). The LeNet architecture works with the ReLU activation function, except for the final fully connected layer where the 'softmax' function is used instead. The SDG optimiser was used (lr = 0.01).  

The model was then trained on the full dataset with 20 epochs and a batch size of 32 




Alt-Operating the Scripts
---

There are 3 steps to take to get your script up and running:
1. Clone the repository 
2. Create a virtual environment (Computer_Vision02) 
3. Run the 2 scripts using command line parameters

___Output will be saved in a new folder called out___

__1. Clone the repository__ 

The easiest way to access the files is to clone the repository from the command line using the following steps 

```bash
#clone repository as classification_benchmarks_orlz
git clone https://github.com/Orlz/CDS_Visual_Analytics.git classification_benchmarks_orlz

```


__2. Create the virtual environment__

You'll need to create a virtual environment which will allow you to run the script using all the relevant dependencies. This will require the requirements.txt file attached to this repository. 


To create the virtual environment you'll need to open your terminal and type the following code: 

```bash
bash create_virtual_environment.sh
```
And then activate the environment by typing: 
```bash
$ source Computer_Vision03/bin/activate
```


__3. Run the Script__

There script contains a number of command-line parameters which can be set by the user. The options for these are as follows: 


___Parameter options = 4___

| Letter call   | Are             | Required?| Input Type   | Description
| ------------- |:-------------:  |:--------:|:-------------:
|`-t`           | `--path2train`  | No       | String       | Path to the training directory (default: ../data/training)   | 
|`-te`          | `--path2test`   | No       | String       | Path to the validation directory (default:../data/validation)|
|`-n`           | `--n_epochs`    | No       | Integer      | Number of epochs to train model on (default: 20)             |
|`-b`           | `--batch_size`  | No       | Integer      | The batch size on which to train the data on (default: 32)   |


Below is an example of the command line arguments for 15 epochs instead of the default:


```bash
python3 src/CNN_artists.py n_epochs 15 
```


Alt-Evaluation of Results 
----

The model's history plot, found in the 'out' folder, indicates that our model is learning quite rapidly up until around 12 epochs. This is demonstrated by the falling blue train_loss curve and growing yellow training accuracy curve. At this point of 12 epochs, both curves begin to flatten and by 15 epochs it appears that the model’s learning has plateaued, suggesting the model does not need to run for further epochs. At the same time, the validation curves indicate the model is not generalising particularly well, with the validation loss curve initially falling but then rapidly rising after around 6 epochs (red). The almost flat validation accuracy curve indicates that the model cannot really manage unseen data with more training any better than it does at the beginning. This suggests that the learning trends seen are likely to be a case of over fitting, i.e., following the errors and trends in the training data too closely. Indeed, our two loss curves end at opposite ends of the plot – suggesting that there is much improvement we could make to the model! 

The classification report found in the out folder helps us to understand more about how the model manages the individual artists. Renoir and Van Gogh stand out as more distinguishable in style, with F1 scores of .47 and .46 respectively. This is considerably above chance levels of classification for 10 classes. At the same time, we see that the model struggles to classify the work of Pissarro who achieves an F1 score of .31. The overall weighted accuracy indicates that in general the model is operating with accuracy of 40%, which is notably above chance levels (10%) and indicates that with more data and fine tuning, it could be possible to build a much more accurate classifier – which would be of great interest to the world of impressionist art! 

To extend upon this assignment, once could attempt to run the dataset with a pre-trained CNN such as the VGG16 and see if this broad knowledge would help the model to become more generalisable. 



