import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from ch05.two_layer_net import TwoLayerNet
import numpy as np

def gradient_check():
  (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

  network = TwoLayerNet(28 * 28, 50, 10)

  # sampling data
  x_sample = x_train[:3]
  t_sample = t_train[:3]

  # gradient by numerical gradient
  gradient_numerical = network.numerical_gradient(x_sample, t_sample)
  # gradient by backpropagation
  gradient_backpropagation = network.gradient(x_sample, t_sample)

  # get differences between gradient_numerical and gradient_backpropagation
  for key in gradient_numerical.keys():
    diff = np.average(np.abs(gradient_numerical[key] - gradient_backpropagation[key]))
    print('{0}: {1}'.format(key, diff))

if __name__ == '__main__':
  gradient_check()
  
