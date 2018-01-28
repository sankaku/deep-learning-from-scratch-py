import sys
import os
sys.path.append(os.pardir)
from ch03.softmax import softmax
from ch04.cross_entropy_error import cross_entropy_error
import numpy as np


class SoftmaxWithLossLayer:
    """
    x -> [Softmax] -> y -> [CrossEntropyError with t] -> out

    In the textbook, this class has `loss` field.
    """

    def __init__(self):
        self.y = None  # output from Softmax
        self.t = None  # teacher data

    def forward(self, x, t):
        """
        x: input to softmax
        t: teacher data
        """
        self.t = t
        self.y = softmax(x)
        loss = cross_entropy_error(self.y, self.t)
        return loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        doutdx = (self.y - self.t) / batch_size
        return doutdx


if __name__ == '__main__':
    softmax_with_loss_layer = SoftmaxWithLossLayer()

    # forward(non-batch)
    x = np.array([5, 1, 0]) # x is like t
    t = np.array([1, 0, 0])
    loss = softmax_with_loss_layer.forward(x, t)
    print('loss = {0}'.format(loss))

    # backward
    dout = 1
    doutdx = softmax_with_loss_layer.backward(dout)
    print('doutdx = {0}'.format(doutdx))

    # forward(batch)
    x = np.array([[5, 1, 0], [3, 0, 2], [1, 1, 5]]) # x[1] and x[2] have large difference with t
    t = np.array([1, 0, 0])
    loss = softmax_with_loss_layer.forward(x, t)
    print('loss = {0}'.format(loss))

    # backward
    dout = 1
    doutdx = softmax_with_loss_layer.backward(dout)
    print('doutdx = {0}'.format(doutdx))
