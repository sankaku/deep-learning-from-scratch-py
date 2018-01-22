# WIP

import sys
import os
sys.path.append(os.pardir)
from cross_entropy_error import cross_entropy_error
from ch03.softmax import softmax
import numpy as np


class simpleNet:
    def __init__(self, row, col):
        self.__W = np.random.randn(row, col)

    def get_W(self):
        return self.__W

    def predict(self, x):
        return np.dot(x, self.__W)

    def get_loss(self, x, t):
        y = softmax(self.predict(x))
        return cross_entropy_error(y, t)


if __name__ == '__main__':
    net = simpleNet(2, 3)

    print('W = {0}'.format(net.get_W()))
