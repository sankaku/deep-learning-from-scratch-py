# Visualize overfitting with weight decay
# mod ch06.plot_overfitting.py


import sys
import os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from ch06.multi_layer_net import MultiLayerNet
from ch06.SGD import SGD
from ch06.plot_overfitting import train
from ch06.plot_overfitting import plot_accuracies
import matplotlib.pyplot as plt


if __name__ == '__main__':
    network, train_accuracies, test_accuracies = train(
        100, 10000, 0.1, weight_decay_lambda=0.1)
    plot_accuracies(train_accuracies, test_accuracies)
