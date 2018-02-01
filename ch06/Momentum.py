# Momentum optimizer

import sys
import os
sys.path.append(os.pardir)
import numpy as np
from ch04.numerical_gradient_batch import numerical_gradient_batch
import matplotlib.pyplot as plt


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
    iterate_num = 10000
    optimizer = Momentum()

    f = lambda x: (1/20) * (x[0] ** 2) + x[1] ** 2
    # Initial params. This is the beginning point.
    params = {'x': 100.0, 'y': 100.0}
    params_array = np.array([[params['x'], params['y']]])
    for i in range(iterate_num):
        tmp_gradients = numerical_gradient_batch(
            f, np.array([params['x'], params['y']]))
        gradients = {'x': tmp_gradients[0], 'y': tmp_gradients[1]}
        optimizer.update(params, gradients)
        params_array = np.vstack(
            (params_array, np.array([params['x'], params['y']])))
    print(params)
    print(params_array.shape)
    x = params_array[:, 0]
    y = params_array[:, 1]
    plt.plot(x, y, '.')
    plt.grid()
    plt.show()
