import numpy as np


def OR(x1, x2):
    """returns x1 OR x2"""
    # vectorize
    x = np.array([x1, x2])
    # weight
    w = np.array([0.4, 0.4])
    # bias
    b = -0.1
    sum = np.sum(w * x) + b
    if sum <= 0:
        return 0
    else:
        return 1


def result(x1, x2):
    return '{0} OR {1} => {2}'.format(x1, x2, OR(x1, x2))


if __name__ == '__main__':
    print(result(0, 0))
    print(result(0, 1))
    print(result(1, 0))
    print(result(1, 1))
