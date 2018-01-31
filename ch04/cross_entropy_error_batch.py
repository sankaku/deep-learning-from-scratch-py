# batch version

import numpy as np


def cross_entropy_error(y, t):
    delta = 1e-7  # to avoid log(0)

    # if y is not a matrix
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    # if labels are one-hot vectors
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return - np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size


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

    ts = np.array([[0, 0, 1, 0, 0],
                   [1, 0, 0, 0, 0],
                   [0, 0, 0, 1, 0]])
    ys = np.array([[0.1, 0.4, 0.6, 0, 0.3],
                   [0.3, 0.5, 0.1, 0.7, 0],
                   [0, 0, 0, 1, 0]])

    print('ts: {0}, ys: {1}'.format(ts.shape, ys.shape))
    print('ys = {0}, cross_entropy_error = {1}'.format(
        ys, cross_entropy_error(ys, ts)))
