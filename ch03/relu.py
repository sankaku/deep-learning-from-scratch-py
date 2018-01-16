# plot ReLU(Rectified Linear Unit)

import numpy as np
import matplotlib.pyplot as plt


def relu(x):
    """
    Definition of ReLU.
    x and output are NumPy arrays.
    """
    return np.maximum(0, x)


def plot():
    x = np.arange(-10, 10, 0.1)
    y = relu(x)

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal',
                         autoscale_on=True, xlim=(-10, 10), ylim=(-0.1, 10))
    plt.plot(x, y)
    plt.grid(linestyle='--')
    plt.xticks(np.arange(-10, 10, 1))
    plt.yticks(np.arange(0, 10, 1))
    plt.show()


if __name__ == '__main__':
    plot()
