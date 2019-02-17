# Tensorflow-Model-Inference

With this project you can train your models using Tensorflow library on python, and export trained model to your C++ project 
to use it without including official TF library for C++.

You export your trained weights and biases to a .npz files, archived tensors and then load them for an inference.

Project uses yannpp: https://github.com/ribtoks/yannpp - math lib for neural networks on C++,
         and cnpy: https://github.com/rogersce/cnpy - numpy for C++ for extracting data from .npz files.
