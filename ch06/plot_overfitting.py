# Visualize overfitting
# mod ch05.train_mnist_by_two_layer_net.py


import sys
import os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from ch06.multi_layer_net import MultiLayerNet
from ch06.SGD import SGD
import matplotlib.pyplot as plt


def train(batch_size, iterate_num, learning_rate, weight_decay_lambda=0.0):
    """
    MNIST training with too many network parameters and too small dataset

    batch_size: data of this number are choosed from training data in each step
    iterate_num: the number of iteration for backpropagation
    learning_rate: learning rate for backpropagation
    weight_decay_lambda: coefficient of L2-norm weight decay
    """
    # get training data and test data(test data are not used below.)
    (x_train, t_train), (x_test, t_test) = load_mnist(
        normalize=True, one_hot_label=True)

    epoch_size = 100

    # reduce training dataset to overfit
    x_train = x_train[:300]
    t_train = t_train[:300]

    # initialized MultiLayerNet
    hidden_size_list = [100, 100, 100, 100, 100, 100]  # many layers to overfit
    # each image has 28*28 pixels
    network = MultiLayerNet(28 * 28, hidden_size_list,
                            10, weight_decay_lambda=weight_decay_lambda)

    # accuracies
    train_accuracies = []
    test_accuracies = []

    # optimizer
    optimizer = SGD()

    for i in range(iterate_num):
        # choose the training data for this step
        indices = np.random.choice(len(x_train), batch_size)
        x_train_batch = x_train[indices]
        t_train_batch = t_train[indices]

        # calculate the grad
        # grads = network.numerical_gradient(x_train_batch, t_train_batch)
        grads = network.gradient(x_train_batch, t_train_batch)

        # update the network parameters
        optimizer.update(network.params, grads)

        # record accuracy
        if i % epoch_size == 0:
            train_accuracy = network.accuracy(x_train, t_train)
            test_accuracy = network.accuracy(x_test, t_test)
            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)
            print('Done {0}/{1}. train_accuracy = {2}, test_accuracy = {3}'.format(
                i + 1, iterate_num, train_accuracy, test_accuracy))

    return network, train_accuracies, test_accuracies


def plot_accuracies(train_accuracies, test_accuracies):
    """
    Plot `train_accuracies` and `test_accuracies`
    """
    x = np.arange(0, len(train_accuracies))
    y = train_accuracies
    plt.plot(x, y, label='train')
    x = np.arange(0, len(test_accuracies))
    y = test_accuracies
    plt.plot(x, y, label='test')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    network, train_accuracies, test_accuracies = train(100, 10000, 0.1)
    plot_accuracies(train_accuracies, test_accuracies)
