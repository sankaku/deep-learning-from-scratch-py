# Visualize the effect of Xavier initialization

import sys
import os
sys.path.append(os.pardir)
from ch03.sigmoid import sigmoid
import numpy as np
import matplotlib.pyplot as plt


class Network:
    """
    Neural network for visualizing the distribution of activations(outputs from sigmoid)
    """

    def __init__(self, node_num=100, hidden_layers=5):
        """
        Initialize network

        node_num: number of nodes in each hidden layer
        hidden_layers: number of hidden layers
        """
        self.node_num = node_num
        self.hidden_layers = hidden_layers

    def make_activations(self):
        activations = {}
        x = np.random.randn(1000, 100)

        for layer_num in range(self.hidden_layers):
            if layer_num != 0:
                x = activations[layer_num - 1]

            # Xavier initialization
            w = np.random.randn(self.node_num, self.node_num) / \
                np.sqrt(self.node_num)

            activations[layer_num] = sigmoid(np.dot(x, w))

        return activations


class Plot:
    def plot(self, activations):
        for index, activation in activations.items():
            plt.subplot(1, len(activations), index + 1)
            plt.title('{0}-layer'.format(index + 1))
            plt.hist(activation.flatten(), bins=30, range=(0, 1))
        plt.show()


if __name__ == '__main__':
    node_num = 100
    hidden_layers = 5
    network = Network(node_num, hidden_layers)
    activations = network.make_activations()

    plot = Plot()
    plot.plot(activations)
