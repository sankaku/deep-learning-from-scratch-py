import numpy as np

def softmax_with_overflow(x):
    numerators = np.exp(x)
    denominator = np.sum(numerators)
    return numerators / denominator


def softmax(x):
    """return softmax for the NumPy array: x.""" 
    # constant for removing overflow
    c = np.max(x)
    numerators = np.exp(x - c)
    denominator = np.sum(numerators)
    return numerators / denominator


if __name__ == '__main__':
    x1 = np.array([0.1, 0.5, 1.0])
    x2 = np.array([1000, 2000, 3000]) # this input may cause a overflow

    print('[softmax_with_overflow]')
    print('{0} => {1}'.format(x1, softmax_with_overflow(x1)))
    print('{0} => {1}'.format(x2, softmax_with_overflow(x2)))

    print('\n[softmax]')
    print('{0} => {1}'.format(x1, softmax(x1)))
    print('{0} => {1}'.format(x2, softmax(x2)))
