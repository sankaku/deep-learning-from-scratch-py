# Show the gradient of z = x**2 + y**2 by arrows
# https://matplotlib.org/examples/pylab_examples/quiver_demo.html

import matplotlib.pyplot as plt
import numpy as np

X, Y = np.meshgrid(np.arange(-2, 2, .25), np.arange(-2, 2, .25))
U = - 2 * X
V = - 2 * Y

plt.figure()
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.title('gradient of z = x**2 + y**2')
plt.quiver(X, Y, U, V, units='width')
plt.show()