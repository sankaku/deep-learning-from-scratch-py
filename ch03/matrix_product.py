# Sample of calculating the product of matrices by NumPy

import numpy as np


def product(m1, m2):
    """returns the product of the metrices: m1 and m2"""
    return np.dot(m1, m2)


def rotate(input, theta):
    """ rotate a vector by rotation matrix"""
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])
    return product(rotation_matrix, input)


if __name__ == '__main__':
    m1 = np.array([[1, 1, 0], [0, 1, 1]])
    m2 = np.array([[1, 0], [0, 0], [0, 1]])
    print('{0}\n dot\n {1}\n =\n {2}'.format(m1, m2, product(m1, m2)))

    # not a simple example
    x = np.array([1, 0])
    theta = np.pi / 2
    y = rotate(x, theta)
    print('{0} === (rotate {1}[rad]) ===> {2}'.format(x, theta, y))
