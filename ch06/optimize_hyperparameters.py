# optimize hyperparameters: learning rate and weight decay lambda
# mod ch06.plot_overfitting.py


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

    epoch_size = 5

    # reduce training dataset to time-saving
    x_train = x_train[:500]
    t_train = t_train[:500]

    # split training data into for training and for validation
    validation_rate = 0.2
    split_mask = (np.random.rand(x_train.shape[0])) < validation_rate
    x_validate = x_train[split_mask]
    t_validate = t_train[split_mask]
    # `~split_mask` means `not(split_mask)` for each element.
    x_train = x_train[~split_mask]
    t_train = t_train[~split_mask]

    # initialized MultiLayerNet
    hidden_size_list = [100, 100, 100, 100, 100, 100]  # many layers to overfit
    # each image has 28*28 pixels
    network = MultiLayerNet(28 * 28, hidden_size_list,
                            10, weight_decay_lambda=weight_decay_lambda)

    # accuracies
    train_accuracies = []
    test_accuracies = []
    validation_accuracies = []

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
            validation_accuracy = network.accuracy(x_validate, t_validate)
            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)
            validation_accuracies.append(validation_accuracy)
            # print('Done {0}/{1}. train_accuracy = {2}, test_accuracy = {3}, validation_accuracy = {4}'.format(
            #     i + 1, iterate_num, train_accuracy, test_accuracy, validation_accuracy))
    print('validation_accuracies ~ {0}'.format(
        np.average(validation_accuracies[-1])))

    return network, train_accuracies, validation_accuracies


def plot_accuracies(train_accuracies, validation_accuracies):
    """
    Plot `train_accuracies` and `validation_accuracies`
    """
    x = np.arange(0, len(train_accuracies))
    y = train_accuracies
    plt.plot(x, y, label='train')
    x = np.arange(0, len(validation_accuracies))
    y = validation_accuracies
    plt.plot(x, y, label='validation')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    trial_optimize_num = 20
    for i in range(trial_optimize_num):
        learning_rate = 10 ** np.random.uniform(-6, -2)  # 1e-6 ... 1e-2
        weight_decay_lambda = 10 ** np.random.uniform(-8, -4)  # 1e-8 ... 1e-4
        print('learning_rate = {0}, weight_decay_lambda = {1}'.format(
            learning_rate, weight_decay_lambda))
        network, train_accuracies, validation_accuracies = train(
            100, 100, learning_rate=learning_rate, weight_decay_lambda=weight_decay_lambda)
        # network, train_accuracies, validation_accuracies = train(100, 100, learning_rate=0.005, weight_decay_lambda=1e-7)
        plot_accuracies(train_accuracies, validation_accuracies)
