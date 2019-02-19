# Tensorflow-Model-Inference

[![Build Status](https://travis-ci.org/Vearol/Tensorflow-Model-Inference.svg?branch=master)](https://travis-ci.org/Vearol/Tensorflow-Model-Inference)
[![Build status](https://ci.appveyor.com/api/projects/status/mxbh149x0e2altcg/branch/master?svg=true)](https://ci.appveyor.com/project/Vearol/tensorflow-model-inference/branch/master)

With this project you can train your models using Tensorflow library on python, and export trained model to your C++ project 
to use it without including official TF library for C++.

You export your trained weights and biases to a `.npz` files, archived tensors and then load them for an inference.

### Model export
* Requirements: tensorflow and numpy packages
* Tensorflow model training: make sure you update [tf.GraphKeys.TRAINABLE_VARIABLES](https://www.tensorflow.org/api_docs/python/tf/GraphKeys#TRAINABLE_VARIABLES) during the training
* Tensorflow session saver: check how do you save your session, because you will need to restore it in order to export your model to `.npy` files. Currently my exporting script works with `.meta` files. But it doesn't matter if you know how to restore your session with graph.

### Inference in C++
#### Create model
For C++ inference you have to manually initialize your layers with properties similar to your tensorflow model.</br>
For example: for a convolution layer you need to specify
1. Input shape
1. Filter shape
1. Filters count
1. Padding type
1. Activation function
1. Layer name - must be the same as in Tensorflow</br>
#### How fast does it work
Time required to prepare VGG-16 model and load weights and biases is around 1.3 seconds </br>
`18:22:53.062 info T#8648 create_layers - Loading layers...` </br>
`18:22:54.382 info T#8648 read_image - Reading image`
</br> </br>
Time required to feed forward a test image from Tiny-ImageNet is around 1.1 seconds </br>
`18:55:43.429 info T#7995 main - Running inference...` </br>
`18:55:44.571 info T#7995 main - Output ready`

### Example - VGG-16
As an example of usage I have chosen VGG-16 CNN model, which was trained on [TinyImageNet](https://tiny-imagenet.herokuapp.com/). </br> More details about VGG model and Tiny-ImageNet could be found in this article: [VGGNet and Tiny ImageNet](https://learningai.io/projects/2017/06/29/tiny-imagenet.html) </br>
</br> Model consists of 13 layers(10 layers of convolution and 3 dense layers), and takes **956 Mb** of storage. 
</br> Saved in `.npz` files it takes **476 Mb**, which is relatively small.

### Submodules: 
* yannpp: https://github.com/ribtoks/yannpp - math lib for neural networks on C++
* cnpy: https://github.com/rogersce/cnpy - numpy for C++, used for extracting data from `.npz` files
