# mod ch05.two_layer_net

import sys
import os
sys.path.append(os.pardir)
from ch03.sigmoid import sigmoid
from ch03.softmax import softmax
from ch04.numerical_gradient_batch import numerical_gradient_batch
import numpy as np
from ch05.ReluLayer import ReluLayer
from ch05.SoftmaxWithLossLayer import SoftmaxWithLossLayer
from collections import OrderedDict


class AffineLayer:
    """
    forward/backward propagation for Affine transformation: Y = XW + B

    Mod ch05.AffineLayer.
    In MultiLayerNet.predict, ch05.AffineLayer is difficult to use
    because its `forward` method needs arguments.
    """

    def __init__(self, W, b):
        """
        W: (n, m) matrix
        b: (1, m) matrix. This is NOT (batch_size, m) matrix because bias is common for all data in `x`.
        """
        self.x = None
        self.W = W
        self.b = b
        self.dW = None
        self.db = None

    def forward(self, x):
        """
        return y = xW + b

        x: (batch_size, n) matrix
        """
        self.x = x
        return np.dot(self.x, self.W) + self.b

    def backward(self, doutdy):
        """
        return d(out)/dx, d(out)/dW

        d(out)/dx = d(out)/dy * [W]T
        d(out)/dW = [x]T * d(out)/dy
        d(out)/db = d(out)/dy
        ([A]T means the transpose matrix of A.)

        doutdy: (batch_size, m) matrix (`batch_size` and `m` must be consistent with those of `forward`.)
        """
        doutdx = np.dot(doutdy, self.W.T)
        self.dW = np.dot(self.x.T, doutdy)
        self.db = np.sum(doutdy, axis=0)
        return doutdx


class MultiLayerNet:
    def __init__(self, input_size, hidden_size_list, output_size, weight_init_std=0.01, weight_decay_lambda=0.0):
        """
        Initialize n-layers network.

        Weight matrices are initialized randomly.
        Biases are zero vectors.

        input_size: the number of neurons in layer-0(input to this network)
        hidden_size_list: the numbers of neurons in layer-1, 2, ..., n-1(hidden layers)
        output_size: the number of neurons in layer-n(output of this network)
        weight_init_std: weight to the weight matrices
        weight_decay_lambda: coefficient of L2-norm weight decay(lambda * W**2 /2)
        """
        self.weight_decay_lambda = weight_decay_lambda

        self.hidden_size_list = hidden_size_list
        hidden_layers_num = len(self.hidden_size_list)

        # params
        self.params = {}
        all_size_list = [input_size] + self.hidden_size_list + [output_size]
        for index in range(1, len(all_size_list)):
            self.params['W{0}'.format(index)] = weight_init_std * \
                np.random.randn(all_size_list[index - 1], all_size_list[index])
            self.params['b{0}'.format(index)] = np.zeros(all_size_list[index])

        # layers
        self.layers = OrderedDict()
        for index in range(1, hidden_layers_num + 1):
            self.layers['Affine{0}'.format(index)] = AffineLayer(
                self.params['W{0}'.format(index)], self.params['b{0}'.format(index)])
            self.layers['Relu{0}'.format(index)] = ReluLayer()
        self.layers['Affine{0}'.format(hidden_layers_num + 1)] = AffineLayer(self.params['W{0}'.format(
            hidden_layers_num + 1)], self.params['b{0}'.format(hidden_layers_num + 1)])
        self.lastLayer = SoftmaxWithLossLayer()

    def predict(self, x):
        """
        x: input to this network(NumPy array)
        """
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        """
        x: input(NumPy array)
        t: teacher data(NumPy array)
        """
        weight_decay = 0
        for index in range(1, len(self.hidden_size_list) + 2):
            W = self.params['W{0}'.format(index)]
            weight_decay += (1 / 2) * self.weight_decay_lambda * np.sum(W**2)
        return self.lastLayer.forward(self.predict(x), t) + weight_decay

    def accuracy(self, x, t):
        """
        Returns the accuracy of this network.

        x: input(NumPy array)
        t: teacher data(NumPy array)
        """

        y = self.predict(x)
        y_labels = np.argmax(y, axis=1)
        t_labels = np.argmax(t, axis=1)

        num_all = x.shape[0]
        num_hit = np.sum(y_labels == t_labels)
        return num_hit / float(num_all)

    def gradient(self, x, t):
        """
        Calculate gradient by backpropagation.
        """
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        for index in range(1, len(self.hidden_size_list) + 2):
            dW = self.layers['Affine{0}'.format(index)].dW
            db = self.layers['Affine{0}'.format(index)].db
            grads['W{0}'.format(
                index)] = dW + self.weight_decay_lambda * self.layers['Affine{0}'.format(index)].W
            grads['b{0}'.format(index)] = db

        return grads


if __name__ == '__main__':
    net = TwoLayerNet(784, 100, 10)
    print("net.params['W1'].shape = {0}".format(net.params['W1'].shape))
    print("net.params['b1'].shape = {0}".format(net.params['b1'].shape))
    print("net.params['W2'].shape = {0}".format(net.params['W2'].shape))
    print("net.params['b2'].shape = {0}".format(net.params['b2'].shape))
