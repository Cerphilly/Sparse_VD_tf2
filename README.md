# Sparse_VD_tf2
Implementation of Sparse Variational Dropout in Tensorflow2!

Original paper: https://arxiv.org/pdf/1701.05369.pdf

Official repo: https://github.com/bayesgroup/variational-dropout-sparsifies-dnn

## Requirements

* TensorFlow >= 2.0

```shell
pip install tensorflow
```

## How to Test

run LeNet.py to test training of LeNet5 by MNIST dataset.

## Result

96% accuracy & 95% sparsity after 20 epochs of training. 
