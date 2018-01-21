# find the minimal value by gradient method

import numpy as np
from numerical_gradient import numerical_gradient


def gradient_descent(f, initial_x, learning_rate, iterate_num):
    """
    Get the minimal point.

    f: target function
    initial_x: Initial point for x. Iteration starts from this point.
    learning_rate: learning rate
    iterate_num: the number of iteration
    """
    x = initial_x

    for i in range(iterate_num):
        diff = learning_rate * numerical_gradient(f, x)
        x = x - diff
    return x


if __name__ == '__main__':
    print('z = x**2 + y**2')
    minimal_point = gradient_descent(
        lambda x: x[0]**2 + x[1]**2, np.array([3, 3]), 0.1, 100)
    print('minimal_point = {0}'.format(minimal_point))

    print('\nIf the learning_rate is too large (0.99) ...')
    minimal_point = gradient_descent(
        lambda x: x[0]**2 + x[1]**2, np.array([3, 3]), 0.99, 100)
    print('minimal_point = {0}'.format(minimal_point))
