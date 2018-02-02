# Visualize the optimization process for the example:
# function: f(x, y) = (1/20) * (x ** 2) + y ** 2
#   - This func varies very slowly for x and the minimal point is (0, 0).
# beginning point: (100.0, 100.0)

import numpy as np
from ch04.numerical_gradient_batch import numerical_gradient_batch
import matplotlib.pyplot as plt


def make_params_history(optimizer):
    iterate_num = 10000

    f = lambda x: (1/20) * (x[0] ** 2) + x[1] ** 2

    # Initial params. This is the beginning point.
    params = {'x': 100.0, 'y': 100.0}
    params_history = np.array([[params['x'], params['y']]])
    for i in range(iterate_num):
        grad = numerical_gradient_batch(
            f, np.array([params['x'], params['y']]))
        gradients = {'x': grad[0], 'y': grad[1]}
        optimizer.update(params, gradients)
        params_history = np.vstack(
            (params_history, np.array([params['x'], params['y']])))
    print('minimal point: {0}'.format(params))
    return params_history


def plot_optimization_process(optimizer):
    params_history = make_params_history(optimizer)
    x = params_history[:, 0]
    y = params_history[:, 1]
    plt.plot(x, y, '.')
    plt.grid()
    plt.show()
