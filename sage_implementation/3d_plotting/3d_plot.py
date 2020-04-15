#!/usr/bin/env python3

from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.axes(projection="3d")

#z_points = 15 * np.random.random(100)
#x_points = np.cos(z_points) + 0.1 * np.random.randn(100)
#y_points = np.sin(z_points) + 0.1 * np.random.randn(100)
#ax.scatter3D(x_points, y_points, z_points, c=z_points, cmap='hsv');

"""
x_points = np.array([.7031, 12.9871, 427.7926, 4.5025, 2244.0386, 0.4926, 7.3762, 156.0496, 6718.6143, 3.2504, 1048.303])
y_points = np.array([6, 7, 8, 6, 7, 6, 7, 8, 9, 6, 7])
z_points = np.array([6, 7, 9, 17, 19, 6, 7, 9, 11, 17, 19])
ax.plot_trisurf(x_points, y_points, z_points, linewidth=0.2, antialiased=False, cmap='winter')

ax.set_xlabel('Time')
ax.set_ylabel('Variables')
ax.set_zlabel('Max Degree')

x_points = np.array([.5767, 10.9896, 367.9587, 3.9714, 1805.9777, 0.4167, 4.2917, 72.2739, 2458.3658, 2.4883, 505.8548])
y_points = np.array([6, 7, 8, 6, 7, 6, 7, 8, 9, 6, 7])
z_points = np.array([6, 7, 9, 17, 19, 6, 7, 9, 11, 17, 19])
ax.plot_trisurf(x_points, y_points, z_points, linewidth=0.2, antialiased=False, cmap='twilight')
"""

#CPU Iterations
x_points = np.array([.1667, 1.7631, 10.2623, 41.7174, 62.2786, 32.275, 7.581])
y_points = np.array([8, 18, 22, 24, 14, 6, 1])
z_points = np.array([3, 4, 5, 6, 7, 8, 9])

ax.scatter3D(x_points, y_points, z_points, c=z_points, cmap='Reds');
ax.plot_trisurf(x_points, y_points, z_points, linewidth=0.2, antialiased=False, cmap='GnBu')

#GPU Iterations
x_points = np.array([.1302, 1.046, 5.0956, 18.2455, 27.6642, 15.8163, 4.2714])
y_points = np.array([8, 18, 22, 24, 14, 6, 1])
z_points = np.array([3, 4, 5, 6, 7, 8, 9])

ax.scatter3D(x_points, y_points, z_points, c=z_points, cmap='Reds');
ax.plot_trisurf(x_points, y_points, z_points, linewidth=0.2, antialiased=False, cmap='summer')

ax.set_xlabel('Iteration Time')
ax.set_ylabel('Variables')
ax.set_zlabel('Max Degree')

plt.show()