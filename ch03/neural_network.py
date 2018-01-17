# neural network sample
# layer-0 consists of 2-neurons
# layer-1 consists of 3-neurons
# layer-2 consists of 2-neurons
# layer-3 consists of 2-neurons

import numpy as np
from sigmoid import sigmoid


def create_network():
    """return the initial network.(no side effect)"""
    network = {}

    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network


def identity_function(x):
    return x


def forward(network, x):
    # weights
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    # biases
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1  # input to layer-1
    z1 = sigmoid(a1)  # output from layer-1
    a2 = np.dot(z1, W2) + b2  # input to layer-2
    z2 = sigmoid(a2)  # output from layer-2
    a3 = np.dot(z2, W3) + b3  # input to layer-3
    y = identity_function(a3)

    return y


network = create_network()
input = np.array([1.0, 0.5])
output = forward(network, input)
print(output)
