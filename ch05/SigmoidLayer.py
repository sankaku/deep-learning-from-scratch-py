import numpy as np


class SigmoidLayer:
    def __init__(self):
        self.out = None
        pass

    def forward(self, x):
        """x: NumPy array"""
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, dout):
        """
        y = sigmoid(x) = 1/(1+exp(-x))
        d(out)/dx = d(out)/dy * dy/dx = d(out)/dy * y * (1-y)
        """
        return dout * self.out * (1.0 - self.out)


if __name__ == '__main__':
    sigmoid = SigmoidLayer()
    # forward
    x = np.array([[1, 2, 3, -4, -5, -6, 7, 8, 9],
                  [-1, -2, -3, 4, 5, 6, -7, -8, -9]])
    y = sigmoid.forward(x)
    print('y = {0}'.format(y))
    # backward
    dy = np.random.rand(2, 9)
    dx = sigmoid.backward(dy)
    print('dy = {0}\ndx = {1}'.format(dy, dx))
