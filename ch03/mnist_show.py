# Display a MNIST image sample

import sys
import os
import numpy as np
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from PIL import Image


def show_image(image):
    img = Image.fromarray(np.uint8(image))
    img.show()


(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

show_image(x_train[0].reshape(28, 28))
