import numpy as np


def mean_squared_error(y, t):
    return (1 / 2) * np.sum((y - t) ** 2)


if __name__ == '__main__':
    t = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])

    y1 = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    y2 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    y3 = np.array([0.1, 0.9, 0.3, -0.3, 0, 0, 0.2, 0, 0, 0.6])

    print('t = {0}'.format(t))
    print('y1 = {0}, mean_squared_error = {1}'.format(
        y1, mean_squared_error(y1, t)))
    print('y2 = {0}, mean_squared_error = {1}'.format(
        y2, mean_squared_error(y2, t)))
    print('y3 = {0}, mean_squared_error = {1}'.format(
        y3, mean_squared_error(y3, t)))
    print('t = {0}, mean_squared_error = {1}'.format(
        t, mean_squared_error(t, t)))
