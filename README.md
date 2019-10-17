# 3D-Autoencoder
A 3D auto-encoder project based on ShapeNet dataset

## Copyright
* This is an open source demo project from [Jingjing Yang](https://www.linkedin.com/in/jingjingyang801/)
* Any question, please contact [kdj842969@gmail.com](kdj842969@gmail.com)

## Introduction
* This project is a real 3D auto-encoder based on ShapeNet
* In this project, our input is real 3D object in 3d array format. And we use 3D convolution layer to learn the patterns of objects.

## Installation
* Our project is based on [Tensorflow](https://www.tensorflow.org) and [Keras](https://keras.io).
* In order to write our input to hdf5 file, we also need [h5py library](https://www.h5py.org).

## Dataset
* We use [3D ShapeNet](http://3dshapenets.cs.princeton.edu) as our dataset.
* To avoid use .off data, we use volumetric data in their source code.
* In order to simplify our training process, we only select 10 classes. Of course you can use more classes.
* The original input size is 30x30x30. However, in order to fit in our model, we padding them into 32x32x32. The original data looks like below (class for this object is "chair"):
<p align="center">
  <img src="https://github.com/kdj842969/3D-Autoencoder/blob/master/demo2.png" height="250">
</p>

## Architecture
* The architecture of this auto-encoder is shown below:
<p align="center">
  <img src="https://github.com/kdj842969/3D-Autoencoder/blob/master/architecture.png" height="500">
</p>

## Code
* read_off.py: label the original data, shuffle and padding the input, then convert them into hdf5 file.
* train.py: train the auto-encoder model.
* test.py: test the trained model with test set.
* test_vis.py: visulize the first 10 test results.
* autoencoder.h5: a trained model. If you don't want to train the model yourself, you can directly use this file and run test.py to see the results. The training loss for this trained model should be 0.0062.

## Results
* Training loss:

<p align="center">
  <img src="https://github.com/kdj842969/3D-Autoencoder/blob/master/trainingloss.png" height="250">
</p>
                                                                                         
* Validation loss:
<p align="center">
  <img src="https://github.com/kdj842969/3D-Autoencoder/blob/master/validationloss.png" height="250">
</p>
                                                                                           
* Reconstruction example:
  * demo 1 (class: "airplane")
  <p align="center">
    <img src="https://github.com/kdj842969/3D-Autoencoder/blob/master/result/demo0.png" height="250">
  </p>
                                                                                           
  * demo 2 (class: "bathtub")
  <p align="center">
    <img src="https://github.com/kdj842969/3D-Autoencoder/blob/master/result/demo5.png" height="250">
  </p>
                                                                                           
  * demo 3 (class: "chair")
  <p align="center">
    <img src="https://github.com/kdj842969/3D-Autoencoder/blob/master/result/demo6.png" height="250">
  </p>
                                                                                           
