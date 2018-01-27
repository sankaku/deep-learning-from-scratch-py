import numpy as np


class AffineLayer:
    """
    forward/backward propagation for Affine transformation: Y = XW + B

    In the textbook, `doutdW` and `doutdb` are fields.
    """

    def __init__(self):
        self.x = None
        self.W = None
        self.b = None

    def forward(self, x, W, b):
        """
        return y = xW + b

        x: (batch_size, n) matrix
        W: (n, m) matrix
        b: (1, m) matrix. This is NOT (batch_size, m) matrix because bias is common for all data in `x`.
        """
        self.x = x
        self.W = W
        self.b = b
        return np.dot(self.x, self.W) + self.b

    def backward(self, doutdy):
        """
        return d(out)/dx, d(out)/dW

        d(out)/dx = d(out)/dy * [W]T
        d(out)/dW = [x]T * d(out)/dy
        d(out)/db = d(out)/dy
        ([A]T means the transpose matrix of A.)

        doutdy: (batch_size, m) matrix (`batch_size` and `m` must be consistent with those of `forward`.)
        """
        doutdx = np.dot(doutdy, W.T)
        doutdW = np.dot(x.T, doutdy)
        doutdb = np.sum(doutdy, axis=0)
        return doutdx, doutdW, doutdb


if __name__ == '__main__':
    affine_layer = AffineLayer()

    # non-batch
    # Be aware of '[[1, 2, 0]]' not '[1, 2, 0]'.
    # np.array([1,2,3]).T = np.array([1,2,3]), not np.array([[1],[2],[3]]).
    x = np.array([[1, 2, 0]])  # (1, 3) matrix
    W = np.array([[1, 0], [0, 1], [0, 0]])  # (3, 2) matrix
    b = np.array([[1, 1]])  # (1, 2) matrix
    y = affine_layer.forward(x, W, b)
    print('y = {0}'.format(y))
    doutdy = np.array([[1, 0]])  # (1, 2) matrix
    doutdx, doutdW, doutdb = affine_layer.backward(doutdy)
    print('doutdx = {0}, doutdW = {1}, doutdb = {2}'.format(
        doutdx, doutdW, doutdb))

    # batch
    x = np.array([[1, 2, 0], [0, 1, 2]])  # (2, 3) matrix
    W = np.array([[1, 0], [0, 1], [0, 0]])  # (3, 2) matrix
    b = np.array([[1, 1]])  # (1, 2) matrix
    y = affine_layer.forward(x, W, b)
    print('y = {0}'.format(y))
    doutdy = np.array([[1, 0], [0, 1]])  # (2, 2) matrix
    doutdx, doutdW, doutdb = affine_layer.backward(doutdy)
    print('doutdx = {0}, doutdW = {1}, doutdb = {2}'.format(
        doutdx, doutdW, doutdb))
