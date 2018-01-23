# get the numerical gradient of multivariable function f(x1, x2, ...)

import numpy as np
from numerical_diff import show
from numerical_gradient import numerical_gradient


def numerical_gradient_batch(f, x):
    """
    Returns the gradient of f at x[0], at x[1], ... .
    x is a matrix(including array).
    """
    h = 1e-4  # rule of thumb

    if x.ndim == 1: # if x is an array
        gradient = numerical_gradient(f, x)
    else: # if x is an matrix(not array)
        gradient = np.zeros_like(x)
        for index, x_i in enumerate(x):
            gradient[index] = numerical_gradient(f, x_i)
    return gradient


def get_unit_vector(dimension, index):
    vector = np.zeros(dimension)
    vector[index] = 1
    return vector


if __name__ == '__main__':
    print('z = x ** 2 + y ** 2')

    def f1(x): return x[0] ** 2 + x[1] ** 2
    show(f1, np.array([3.0, 4.0]), numerical_gradient,
         'numerical_gradient', np.array([6.0, 8.0]))
    show(f1, np.array([5.0, 0.0]), numerical_gradient,
         'numerical_gradient', np.array([10.0, 0.0]))
    show(f1, np.array([0.0, 0.0]), numerical_gradient,
         'numerical_gradient', np.array([0.0, 0.0]))

    def f2(x): return x[0] + x[1] ** 2 + x[2] ** 3
    show(f2, np.array([1.0, 2.0, 3.0]), numerical_gradient,
         'numerical_gradient', np.array([1.0, 4.0, 27.0]))

    # x can be a matrix
    print(numerical_gradient_batch(f1, np.array([[3.0, 4.0], [5.0, 0.0], [0.0, 0.0]])))
