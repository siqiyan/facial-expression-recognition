## Introduction

This is a group project for university course.

This is a simple example of facial expression recognition with Multi-layer Convolutional Neural Network, implemented
in TensorFlow. It is still in proof-of-concept state, and later we will try using
different architectures and optimization algorithms to imporve the accuracy and
running speed, and we will implement more features like webcam input, face
replacement with emoji, etc.

## About Data Gathering

* The training data can be facial images of any size, RGB or grayscale.
* Images of facial emotions, in the following categories:
    * Anger
    * Disgust
    * Fear
    * Happiness
    * Sadness
    * Neutral (update)
* Front face and side face are okay.
* You need to crop the image to a resolution ratio of roughly 1:1.5 (width:height), since the
  program will resize all images to a fixed size while not keeping the
  resolution ratio.
* Put the images into the corresponding folder in ./dataset
