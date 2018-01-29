# mod ch04.two_layer_net

import sys
import os
sys.path.append(os.pardir)
from ch03.sigmoid import sigmoid
from ch03.softmax import softmax
from ch04.numerical_gradient_batch import numerical_gradient_batch
import numpy as np
from ch05.AffineLayer import AffineLayer
from ch05.ReluLayer import ReluLayer
from ch05.SoftmaxWithLossLayer import SoftmaxWithLossLayer


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        """
        Initialize 2-layer network.

        Weight matrices are initialized randomly.
        Biases are zero vectors.

        input_size: the number of neurons in layer-0(input to this network)
        hidden_size: the number of neurons in layer-1(hidden layer)
        output_size: the number of neurons in layer-2(output of this network)
        weight_init_std: weight to the weight matrices
        """

        self.params = {}
        self.params['W1'] = weight_init_std * \
            np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * \
            np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # layers
        self.layers = {}
        self.layers['Affine1'] = AffineLayer()
        self.layers['Relu1'] = ReluLayer()
        self.layers['Affine2'] = AffineLayer()
        self.lastLayer = SoftmaxWithLossLayer()

    def predict(self, x):
        """
        x: input to this network(NumPy array)
        """
        x = self.layers['Affine1'].forward(
            x, self.params['W1'], self.params['b1'])
        x = self.layers['Relu1'].forward(x)
        x = self.layers['Affine2'].forward(
            x, self.params['W2'], self.params['b2'])
        return x

    def loss(self, x, t):
        """
        x: input(NumPy array)
        t: teacher data(NumPy array)
        """

        return self.lastLayer.forward(self.predict(x), t)

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
        num_hit = np.sum(y == t)
        return num_hit / float(num_all)

    def numerical_gradient(self, x, t):
        # loss_W = lambda W: self.loss(x, t)
        def loss_W(W): return self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient_batch(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient_batch(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient_batch(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient_batch(loss_W, self.params['b2'])
        return grads

    def gradient(self, x, t):
        """
        Calculate gradient by backpropagation.

        In the textbook, `AffineLayer` has `dW` and `db` fields.
        That implementation seems simple now...
        """
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        dout, doutdW2, doutdb2 = self.layers['Affine2'].backward(dout)
        dout = self.layers['Relu1'].backward(dout)
        dout, doutdW1, doutdb1 = self.layers['Affine1'].backward(dout)

        grads = {}
        grads['W1'] = doutdW1
        grads['b1'] = doutdb1
        grads['W2'] = doutdW2
        grads['b2'] = doutdb2

        return grads


if __name__ == '__main__':
    net = TwoLayerNet(784, 100, 10)
    print("net.params['W1'].shape = {0}".format(net.params['W1'].shape))
    print("net.params['b1'].shape = {0}".format(net.params['b1'].shape))
    print("net.params['W2'].shape = {0}".format(net.params['W2'].shape))
    print("net.params['b2'].shape = {0}".format(net.params['b2'].shape))
