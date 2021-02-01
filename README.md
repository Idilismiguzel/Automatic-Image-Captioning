[//]: # (Image References)
[image1]: ./images/coco-examples.jpg "COCO"
[image2]: ./images/encoder.png "Encoder"
[image3]: ./images/decoder.png "Decoder"
[image4]: ./images/encoder-decoder.png "Encoder-Decoder"

# Automatic Image Captioning 

## Project Overview

Create a deep learning architecture with two components: a CNN to transform the input image into a set of features, an RNN that turns those features into descriptive text aka captions. 

General information about the project:

1. The Microsoft **C**ommon **O**bjects in **CO**ntext (MS COCO) dataset is a large-scale dataset for scene understanding.  The dataset is commonly used to train and benchmark object detection, segmentation, and captioning algorithms.  

![image1]

2. Feature vectors for images are generated using a CNN based on the ResNet architecture.
4. Implemented a CNN to transform the input image into a set of features.
![image2]

3. Implemented an RNN decoder using LSTM cells.
![image3]
4. Trained the network around 10 hrs using GPU and achieved average loss of around 2%.
![image4]

The project is broken up into a few main parts in four Python notebooks

__0_Dataset.ipynb__ : Loading and Visualizing Microsoft Common Objects in COntext (MS COCO) dataset to train the network

__1_Preliminaries.ipynb__ : Design a CNN-RNN model for automatically generating image captions.

__2_Training.ipynb__ : Train the CNN-RNN model.

__3_Inference.ipynb__ : Use your trained model to generate captions for images in the test dataset.
