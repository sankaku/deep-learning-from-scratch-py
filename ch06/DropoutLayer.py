# Layer for Dropout
# mod ch05.ReluLayer

import numpy as np


class DropoutLayer:
    def __init__(self, dropout_ratio=0.5):
        """
        initialization

        dropout_ratio: this ratio of nodes are droppped out
        """
        self.mask = None
        self.dropout_ratio = dropout_ratio

    def forward(self, x):
        """x: NumPy array"""
        self.mask = np.random.rand(*x.shape) > self.dropout_ratio
        return x * self.mask

    def backward(self, dout):
        """dout: NumPy array"""
        return dout * self.mask


if __name__ == '__main__':
    dropout = DropoutLayer()
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    print('x        = {0}'.format(x))

    forward = dropout.forward(x)
    print('forward  = {0}'.format(forward))

    backward = dropout.backward(1)
    print('backward = {0}'.format(backward))
