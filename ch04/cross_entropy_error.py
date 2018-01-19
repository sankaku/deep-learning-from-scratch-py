import numpy as np


def cross_entropy_error(y, t):
    delta = 1e-7  # to avoid log(0)
    return - np.sum(t * np.log(y + delta))


if __name__ == '__main__':
    t = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])

    y1 = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    y2 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    y3 = np.array([0.1, 0.9, 0.3, 0.3, 0, 0, 0.2, 0, 0, 0.6])

    print('t = {0}'.format(t))
    print('y1 = {0}, cross_entropy_error = {1}'.format(
        y1, cross_entropy_error(y1, t)))
    print('y2 = {0}, cross_entropy_error = {1}'.format(
        y2, cross_entropy_error(y2, t)))
    print('y3 = {0}, cross_entropy_error = {1}'.format(
        y3, cross_entropy_error(y3, t)))
    print('t = {0}, cross_entropy_error = {1}'.format(
        t, cross_entropy_error(t, t)))
