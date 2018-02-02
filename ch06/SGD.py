# Stocastic Gradient Descent optimizer

import sys
import os
sys.path.append(os.pardir)
import numpy as np
from ch06.plot_optimization_process import plot_optimization_process


class SGD:
    def __init__(self, learning_rate=0.01):
        """
        initialize

        learning_rate: learning rate for iteration
        """
        self.learning_rate = learning_rate

    def update(self, params, gradients):
        """
        update params

        params: dictionary of parameters
        gradients: dictionary of gradients
        """
        # update
        for key in params.keys():
            params[key] -= self.learning_rate * gradients[key]


if __name__ == '__main__':
    optimizer = SGD()
    plot_optimization_process(optimizer)
