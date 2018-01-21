# get the numerical gradient of multivariable function f(x1, x2, ...)

import numpy as np
from numerical_diff import show


def numerical_gradient(f, x):
    """
    Returns the gradient of f at x.
    x is a NumPy array.
    """
    h = 1e-4  # rule of thumb
    gradient = np.zeros_like(x)

    for index in range(x.size):
        h_vector = get_unit_vector(x.size, index) * h
        # f(x_i + h)
        f_plus = f(x + h_vector)

        # f(x_i - h)
        f_minus = f(x - h_vector)

        # print('h_vector = {0}, x + h_vector = {1}, x - h_vector = {2}'.format(h_vector, x + h_vector, x - h_vector))
        # print('f_plus = {0}, f_minus = {1}'.format(f_plus, f_minus))
        gradient[index] = (f_plus - f_minus) / (2 * h)
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
