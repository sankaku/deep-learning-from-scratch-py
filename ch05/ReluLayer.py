import numpy as np


class ReluLayer:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        """x: NumPy array"""
        self.mask = (x > 0)
        return x * self.mask

    def backward(self, dout):
        """dout: NumPy array"""
        return dout * self.mask


if __name__ == '__main__':
    relu = ReluLayer()

    # forward
    x = np.array([[1, 2, 3, -4, -5, -6, 7, 8, 9],
                  [-1, -2, -3, 4, 5, 6, -7, -8, -9]])
    y = relu.forward(x)
    print('y = {0}'.format(y))

    # backward
    dy = np.random.rand(2, 9)
    dx = relu.backward(dy)
    print('dy = {0}\ndx = {1}'.format(dy, dx))
