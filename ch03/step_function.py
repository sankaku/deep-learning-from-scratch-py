# plot step function

import numpy as np
import matplotlib.pyplot as plt


def step_function(x):
    """
    Definition of step function.
    x and output are NumPy arrays.
    """
    y = x > 0
    return y.astype(np.int)


def plot():
    x = np.arange(-10, 10, 0.1)
    y = step_function(x)

    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()


if __name__ == '__main__':
    plot()
