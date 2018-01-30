# mod ch04.train_mnist_by_two_layer_net.py

import sys
import os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from ch05.two_layer_net import TwoLayerNet
import matplotlib.pyplot as plt


def train(batch_size, iterate_num, learning_rate):
    """
    Get the appropriate network parameters (weights, biases) by gradient method.

    In the process of gradient method,
    training data are choosed randomly in each step(mini-batch gradient method).

    batch_size: data of this number are choosed from training data in each step
    iterate_num: the number of iteration for gradient method
    learning_rate: learning rate for gradient method
    """
    # get training data and test data(test data are not used below.)
    (x_train, t_train), (x_test, t_test) = load_mnist(
        normalize=True, one_hot_label=True)

    # initialized TwoLayerNet
    network = TwoLayerNet(28 * 28, 50, 10)  # each image has 28*28 pixels

    # losses in each step
    losses = []

    for i in range(iterate_num):
        # choose the training data for this step
        indices = np.random.choice(len(x_train), batch_size)
        x_train_batch = x_train[indices]
        t_train_batch = t_train[indices]

        # calculate the grad
        # grads = network.numerical_gradient(x_train_batch, t_train_batch)
        grads = network.gradient(x_train_batch, t_train_batch)

        # update the network parameters
        network.params['W1'] -= learning_rate * grads['W1']
        network.params['b1'] -= learning_rate * grads['b1']
        network.params['W2'] -= learning_rate * grads['W2']
        network.params['b2'] -= learning_rate * grads['b2']

        # record loss
        loss = network.loss(x_train_batch, t_train_batch)
        print('loss = {0}'.format(loss))
        losses.append(loss)

        # show accuracy
        if i%(iterate_num/10) == 0:
            print('train_acc = {0}'.format(network.accuracy(x_train, t_train)))
            print('test_acc = {0}'.format(network.accuracy(x_test, t_test)))

    return network, losses


def plot_losses(losses):
    """
    Plot `losses` to visualize the process of iterative learning
    """
    x = np.arange(0, len(losses))
    y = losses
    plt.plot(x, y)
    plt.grid()
    plt.show()


if __name__ == '__main__':
    network, losses = train(100, 10000, 0.1)
    print('W1 = {0}, b1 = {1}, W2 = {2}, b2 = {3}'.format(
        network.params['W1'], network.params['b1'], network.params['W2'], network.params['b2']))
    print('losses = {0}'.format(losses))
    plot_losses(losses)
