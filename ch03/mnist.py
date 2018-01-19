# MNIST classification
# Neural network parameters (weights and biases) are random.
# So the accuracy approximately equals to 0.1.

import sys
import os
import numpy as np
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from sigmoid import sigmoid
from softmax import softmax


def get_test_data():
    """Returns the test data."""
    (x_train, t_train), (x_test, t_test) = load_mnist(
        flatten=True, normalize=False, one_hot_label=False)
    return x_test, t_test


def create_network():
    """
    Returns the randomly initialized network.
    In the textbook, this method gets parameters from file.
    """
    network = {}
    dim0 = 784  # 28x28 (pixels in an image)
    dim1 = 100  # chosen arbitrarily
    dim2 = 50  # chosen arbitrarily
    dim3 = 10  # the number of kinds of labels (0, 1, .. , 9)

    # layer-1: (dim0, 1)-matrix -> (dim1, 1)-matrix
    network['W1'] = np.random.rand(dim0, dim1)
    network['b1'] = np.random.rand(dim1)

    # layer-2: (dim1, 1)-matrix -> (dim2, 1)-matrix
    network['W2'] = np.random.rand(dim1, dim2)
    network['b2'] = np.random.rand(dim2)

    # layer-3: (dim2, 1)-matrix -> (dim3, 1)-matrix
    network['W3'] = np.random.rand(dim2, dim3)
    network['b3'] = np.random.rand(dim3)
    return network


def predict(network, x):
    """forward propagation"""
    # weights# forward propagation
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    # biases
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1  # input to layer-1
    z1 = sigmoid(a1)  # output from layer-1
    a2 = np.dot(z1, W2) + b2  # input to layer-2
    z2 = sigmoid(a2)  # output from layer-2
    a3 = np.dot(z2, W3) + b3  # input to layer-3
    y = softmax(a3)

    return y


if __name__ == '__main__':
    # test data(image, label)
    images, labels = get_test_data()
    network = create_network()

    hit_num = 0
    for i in range(len(images)):
        y = predict(network, images[i])
        predicted = np.argmax(y)
        if predicted == labels[i]:
            hit_num += 1

    print('Accuracy: {0}/{1} = {2}'.format(hit_num,
                                           len(images), hit_num / len(images)))
