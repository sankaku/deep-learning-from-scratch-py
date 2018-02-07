import unittest
import sys
import os
sys.path.append(os.pardir)
import numpy as np
from ch07.simple_convolution import *


class TestSimpleConvolution(unittest.TestCase):

    def test_pad_1(self):
        """
        test with (2, 2, 2)-tensor
        """
        x = np.array([[[1,  2], [3,  4]],
                      [[5,  6], [7,  8]]])
        padding = 1
        actual = pad(x, padding)
        expected = np.array([[[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0]], [
                            [0, 0, 0, 0], [0, 5, 6, 0], [0, 7, 8, 0], [0, 0, 0, 0]]])

        np.testing.assert_array_equal(actual, expected)

    def test_pad_2(self):
        """
        test with (1, 2, 2)-tensor
        """
        x = np.array([[[1,  2], [3,  4]]])
        padding = 1
        actual = pad(x, padding)
        expected = np.array([[[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0]]])

        np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
