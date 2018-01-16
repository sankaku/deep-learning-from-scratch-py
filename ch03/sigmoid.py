# plot sigmoid

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """
    Definition of sigmoid function.
    x and output are NumPy arrays.
    """
    return 1 / (1 + np.exp(-x))


def plot():
    x = np.arange(-10, 10, 0.1)
    y = sigmoid(x)

    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.grid()
    plt.show()


if __name__ == '__main__':
    plot()
