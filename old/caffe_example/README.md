## Description

Use caffe to train a simple multi-layered convolutional net. This is a
preparation for fine-tuning deeper Residual Net by He et al.

The model and the solver come from the official Caffe examples, I did some
minor modifications to the parameters to fit my GPU. See details here: 
https://github.com/BVLC/caffe/tree/master/examples/mnist

## Usage

First you need to install Caffe, which is very complicated, and I found the
easiest way is using Docker.

This page tells you how to do that:
https://github.com/BVLC/caffe/tree/master/docker

Then, you need to put the KDEF data folder to the current directory, and run
create_LMDB_from_KDEF.py, and it will generate two directories, which contains
the training data and the test data respectively.
Make sure you have lmdb and scipy package installed before running this script.

After everything done, simply run train_lenet.py, the parameters are defined
in lenet_solver.prototxt and lenet_train_test.prototxt.

## Future plan

Fine-tuning the 50 layer ResNet on KDEF dataset, which contains 4900 labeled
images of facial expression.
https://github.com/KaimingHe/deep-residual-networks
