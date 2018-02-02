# Adam optimizer

import sys
import os
sys.path.append(os.pardir)
import numpy as np
from ch06.plot_optimization_process import plot_optimization_process


class Adam:
    """
    Implementation of https://arxiv.org/abs/1412.6980v9 [Algorithm 1]
    """

    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999):
        """
        initialize

        learning_rate: learning rate for iteration
        beta1: exponential decay rate for 1st order moment `m`
        beta2: exponential decay rate for 2nd order moment `v`
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2

        self.m = None  # 1st order moment
        self.v = None  # 2nd order moment
        self.step = 0  # iteration number

    def update(self, params, gradients):
        """
        update params

        params: dictionary of parameters
        gradients: dictionary of gradients
        """
        # to avoid 1/(sqrt(0))
        epsilon = 1e-7

        # initialize m
        if self.m is None:
            self.m = {}
            for key, value in params.items():
                self.m[key] = np.zeros_like(value)
        # initialize v
        if self.v is None:
            self.v = {}
            for key, value in params.items():
                self.v[key] = np.zeros_like(value)

        # update
        self.step += 1
        for key in params.keys():
            self.m[key] = self.beta1 * self.m[key] + \
                (1 - self.beta1) * gradients[key]
            self.v[key] = self.beta2 * self.v[key] + \
                (1 - self.beta2) * (gradients[key] * gradients[key])
            m_bias_corrected = self.m[key] / (1 - self.beta1 ** self.step)
            v_bias_corrected = self.v[key] / (1 - self.beta2 ** self.step)
            params[key] -= self.learning_rate * m_bias_corrected / \
                (epsilon + np.sqrt(v_bias_corrected))


if __name__ == '__main__':
    optimizer = Adam()
    plot_optimization_process(optimizer)
