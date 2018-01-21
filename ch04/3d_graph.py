# 3D plot of z = x**2 + y**2
# https://matplotlib.org/mpl_examples/mplot3d/wire3d_demo.py
# https://matplotlib.org/mpl_toolkits/mplot3d/api.html#mpl_toolkits.mplot3d.axes3d.Axes3D.plot_wireframe

import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt


ax = plt.figure().add_subplot(111, projection='3d')

X, Y = np.meshgrid(np.arange(-2, 2, .2), np.arange(-2, 2, .2))
Z = X ** 2 + Y ** 2
ax.plot_wireframe(X, Y, Z, linewidths=0.5, colors="red")
plt.show()
