# AdaGrad optimizer

import sys
import os
sys.path.append(os.pardir)
import numpy as np
from ch06.plot_optimization_process import plot_optimization_process


class AdaGrad:
    def __init__(self, learning_rate=0.01):
        """
        initialize

        learning_rate: learning rate for iteration
        """
        self.learning_rate = learning_rate
        self.h = None

    def update(self, params, gradients):
        """
        update params

        params: dictionary of parameters
        gradients: dictionary of gradients
        """
        # to avoid 1/(sqrt(0))
        epsilon = 1e-7

        # initialize h
        if self.h is None:
            self.h = {}
            for key, value in params.items():
                self.h[key] = np.zeros_like(value)

        # update
        for key in params.keys():
            # Hadamard product(not matrix product)
            self.h[key] += gradients[key] * gradients[key]
            params[key] -= self.learning_rate * \
                gradients[key] / (np.sqrt(self.h[key]) + epsilon)


if __name__ == '__main__':
    optimizer = AdaGrad()
    plot_optimization_process(optimizer)
