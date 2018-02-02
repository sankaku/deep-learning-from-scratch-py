# Momentum optimizer

import sys
import os
sys.path.append(os.pardir)
import numpy as np
from ch06.plot_optimization_process import plot_optimization_process



class Momentum:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        """
        initialize

        learning_rate: learning rate for iteration
        momentum: coefficient for `v`. This value works like viscosity.
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.v = None

    def update(self, params, gradients):
        """
        update params

        params: dictionary of parameters
        gradients: dictionary of gradients
        """
        # initialize v
        if self.v is None:
            self.v = {}
            for key, value in params.items():
                self.v[key] = np.zeros_like(value)

        # update
        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - \
                self.learning_rate * gradients[key]
            params[key] += self.v[key]


if __name__ == '__main__':
    optimizer = Momentum()
    plot_optimization_process(optimizer)
