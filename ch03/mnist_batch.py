# (cf. ./mnist.py)
# MNIST classification (batch version)
# Neural network parameters (weights and biases) are random.
# So the accuracy approximately equals to 0.1.

# Something wrong with this code ... elements of `predicted` are all the same?

import numpy as np
from mnist import get_test_data, create_network, predict

if __name__ == '__main__':
    # test data(image, label)
    images, labels = get_test_data()
    network = create_network()

    batch_size = 100
    hit_num = 0
    for i in range(0, len(images), batch_size):
        y = predict(network, images[i:i + batch_size])
        predicted = np.argmax(y, axis=1)
        hit_num += np.sum(predicted == labels[i:i + batch_size])

    print('Accuracy: {0}/{1} = {2}'.format(hit_num,
                                           len(images), hit_num / len(images)))
