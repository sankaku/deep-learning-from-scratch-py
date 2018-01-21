# numerical differentiation

import numpy as np

def numerical_diff(f, x):
    """
    Returns the derivative of f at x.
    """
    h = 1e-4  # rule of thumb
    return (f(x + h) - f(x - h)) / (2 * h)  # central difference


def naive_numerical_diff1(f, x):
    """
    Simple implementation by mathematical definition.
    """
    h = 1e-50  # '1e-4 for h is too large. It should be inifinitesimal.'
    return (f(x + h) - f(x - h)) / (2 * h)  # central difference


def naive_numerical_diff2(f, x):
    """
    Simple implementation by mathematical definition.
    """
    h = 1e-4  # rule of thumb
    return (f(x + h) - f(x)) / h  # 'Central diff is tricky.'


def show(f, x, diff, diff_label, expected):
    """
    f: function to be differentiated
    x: calculate the derivative at this point
    diff: function to get the derivative
    diff_label: name of `diff`
    expected: mathematically expected value for f'(x)
    """

    actual = diff(f, x)
    epsilon = 1e-5
    message = ''
    if np.linalg.norm(actual - expected) < epsilon:
        message = 'correct'
    else:
        message = 'wrong'

    print('[{0}] diff at {1} = {2} =====> {3}'.format(
        diff_label, x, actual, message))


if __name__ == '__main__':
    print('f(x) = x')
    show(lambda x: x, 0, numerical_diff, 'numerical_diff', 1)
    show(lambda x: x, 0, naive_numerical_diff1, 'naive_numerical_diff1', 1)
    show(lambda x: x, 0, naive_numerical_diff2, 'naive_numerical_diff2', 1)
    print()
    print('f(x) = x ** 2')
    show(lambda x: x ** 2, 2, numerical_diff, 'numerical_diff', 4)
    show(lambda x: x ** 2, 2, naive_numerical_diff1, 'naive_numerical_diff1', 4)
    show(lambda x: x ** 2, 2, naive_numerical_diff2, 'naive_numerical_diff2', 4)
    print()
    print('f(x) = x ** 3')
    show(lambda x: x ** 3, 5, numerical_diff, 'numerical_diff', 75)
    show(lambda x: x ** 3, 5, naive_numerical_diff1, 'naive_numerical_diff1', 75)
    show(lambda x: x ** 3, 5, naive_numerical_diff2, 'naive_numerical_diff2', 75)
