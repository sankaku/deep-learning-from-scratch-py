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
        expected = np.array(
            [[[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0]]])

        np.testing.assert_array_equal(actual, expected)

    def test_get_output_size_1(self):
        """
        test with padding=1, stride=1
        """
        actual = get_output_size(5, 8, 3, 4, 1, 1)
        expected = (5, 7)
        self.assertEquals(actual, expected)

    def test_get_output_size_2(self):
        """
        test with invalid values

        In this case, output_height and output_width are not integer
        and `ValueError` is raised.
        """
        expected = ValueError

        with self.assertRaises(expected):
            actual = get_output_size(5, 8, 3, 4, 1, 3)

    def test_convolute_1(self):
        """
        simple test

        x.shape == filters.shape
        """
        x = np.array([[[1,  2], [3,  4]],
                      [[5,  6], [7,  8]]])
        filters = np.array([[[1,  0], [2,  0]],
                            [[3,  0], [4,  0]]])
        actual = convolute(x, filters, padding=0, stride=1)
        expected = np.array([[[7]], [[43]]])

        np.testing.assert_array_equal(actual, expected)

    def test_convolute_2(self):
        """
        x.shape = (3, 4, 5)
        filters.shape = (3, 2, 3)
        """
        x = np.arange(3 * 4 * 5).reshape(3, 4, 5)
        filters = np.array([[[1,  0,  0], [0,  1,  0]],
                            [[0,  0,  1], [1,  0,  0]],
                            [[0,  1,  0], [0,  0,  1]]])

        actual = convolute(x, filters, padding=0, stride=2)
        expected = np.array([[[6,  10.], [26,  30]],
                             [[47,  51], [67,  71]],
                             [[88,  92], [108, 112]]])

        np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
