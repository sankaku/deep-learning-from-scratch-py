# Simple Convolution
#
# implement to understand the works of Convolution layer

import numpy as np


def convolute(x, filters, padding=0, stride=1):
    """
    Convolute input with filters

    x: (batches, channels, height, width)-tensor input
    filters: (channels, filter_heights, filter_width)-tensor
    padding: Width to expand the edge of input. Expanded areas are padding with 0
    stride: Width to slide filter
    """

    batches = x.shape[0]
    channels = x.shape[1]
    height = x.shape[2]
    width = x.shape[3]

    if filters.shape[0] != channels:
        raise ValueError('Channel numbers of filters must be equal to that of x.: x.shape = {0}, filters.shape = {1}'.format(
            x.shape, filters.shape))
    filter_height = filters.shape[1]
    filter_width = filters.shape[2]

    output_height, output_width = get_output_size(
        height, width, filter_height, filter_width, padding, stride)
    output = np.zeros([batches, channels, output_height, output_width])

    # padding
    x = pad(x, padding)

    # convolute
    for oh_num in range(output_height):
        for ow_num in range(output_width):
            tmp_h = stride * oh_num
            tmp_w = stride * ow_num
            sliced = x[:, :, tmp_h:tmp_h + filter_height,
                       tmp_w:tmp_w + filter_width]
            output[:, :, oh_num, ow_num] = np.sum(
                np.sum(sliced * filters, axis=2), axis=2)
    return output


def pad(x, padding):
    """
    pad with `pad_val`

    (batches, channels, height, width) -> (batches, channels, height + 2*padding, width + 2*padding)
    x: (batches, channels, height, width)-tensor
    padding: padding size for `height` and `width`
    """
    pad_val = 0
    return np.pad(x, pad_width=[(0, 0), (0, 0), (padding, padding), (padding, padding)], mode='constant', constant_values=pad_val)


def get_output_size(height, width, filter_height, filter_width, padding, stride):
    """
    return (output_height, output_width)

    You can see below by drawing a figure.
    height + 2 * padding = filter_height + (output_height - 1) * stride
    width + 2 * padding = filter_width + (output_width - 1) * stride

    height: height of input
    width: width of input
    filter_height: height of filter
    filter_width: width of filter
    padding: width to expand the edge of input
    stride: Width to slide filter
    """
    output_height = 1 + (height + 2 * padding - filter_height) / stride
    output_width = 1 + (width + 2 * padding - filter_width) / stride

    if output_height != int(output_height):
        raise ValueError(
            'output_height({0}) must be integer.'.format(output_height))
    if output_width != int(output_width):
        raise ValueError(
            'output_width({0}) must be integer.'.format(output_width))

    return int(output_height), int(output_width)
