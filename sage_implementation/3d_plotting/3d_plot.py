#!/usr/bin/env python3

from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt



fig = plt.figure()
ax = plt.axes(projection="3d")

x_points = np.array([.7031, 12.9871, 427.7926, 4.5025, 2244.0386, 0.4926, 7.3762, 156.0496, 6718.6143, 3.2504, 1048.303])
y_points = np.array([6, 7, 8, 6, 7, 6, 7, 8, 9, 6, 7])
z_points = np.array([6, 7, 9, 17, 19, 6, 7, 9, 11, 17, 19])
ax.plot_trisurf(x_points, y_points, z_points, linewidth=0.2, antialiased=False, cmap='winter')

ax.set_xlabel('Time')
ax.set_ylabel('Variables')
ax.set_zlabel('Max Degree')
ax.set_title('Field Sizes 65521/32003')

x_points = np.array([.5767, 10.9896, 367.9587, 3.9714, 1805.9777, 0.4167, 4.2917, 72.2739, 2458.3658, 2.4883, 505.8548])
y_points = np.array([6, 7, 8, 6, 7, 6, 7, 8, 9, 6, 7])
z_points = np.array([6, 7, 9, 17, 19, 6, 7, 9, 11, 17, 19])
ax.plot_trisurf(x_points, y_points, z_points, linewidth=0.2, antialiased=False, cmap='twilight')



fig = plt.figure()
ax = plt.axes(projection="3d")

#CPU Iterations
x_points = np.array([.1667, 1.7631, 10.2623, 41.7174, 62.2786, 32.275, 7.581])
y_points = np.array([8, 18, 22, 24, 14, 6, 1])
z_points = np.array([3, 4, 5, 6, 7, 8, 9])

ax.scatter3D(x_points, y_points, z_points, c=z_points, cmap='Reds');
ax.plot_trisurf(x_points, y_points, z_points, linewidth=0.2, antialiased=False, cmap='Reds')

#GPU Iterations
x_points = np.array([.1302, 1.046, 5.0956, 18.2455, 27.6642, 15.8163, 4.2714])
y_points = np.array([8, 18, 22, 24, 14, 6, 1])
z_points = np.array([3, 4, 5, 6, 7, 8, 9])

ax.scatter3D(x_points, y_points, z_points, c=z_points, cmap='Reds');
ax.plot_trisurf(x_points, y_points, z_points, linewidth=0.2, antialiased=False, cmap='summer')

ax.set_xlabel('Iteration Time')
ax.set_ylabel('Pairs')
ax.set_zlabel('Max Degree')
ax.set_title('Iteration Statistics - Katsura-8 - 32003')




fig = plt.figure()
ax = plt.axes(projection="3d")


#CPU Iterations
x_points = np.array([0.0224, 0.0483, 0.2937, 1.2089, 4.8563, 14.8531, 36.0413, 89.6144, 135.7663, 170.4749, 216.9355, 
        142.5655, 209.563, 6.5576, 11.886, 7.5991, 0.0035, 0.0035, 0.0006])
y_points = np.array([1, 2, 6, 14, 34, 58, 100, 137, 199, 204, 200, 155, 92, 83, 131, 157, 45, 68, 6])
z_points = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])

ax.scatter3D(x_points, y_points, z_points, c=z_points, cmap='Reds');
ax.plot_trisurf(x_points, y_points, z_points, linewidth=0.2, antialiased=False, cmap='Reds')

#GPU Iterations
x_points = np.array([0.038, 0.0451, 0.2377, 0.9131, 3.3268, 8.8899, 19.9679, 44.9176, 70.2492, 81.8717, 93.9579, 
        69.4461, 92.789, 4.5246, 8.3854, 6.2813, 0.0035, 0.0035, 0.0005])
y_points = np.array([1, 2, 6, 14, 34, 58, 100, 137, 199, 204, 200, 155, 92, 83, 131, 157, 45, 68, 6])
z_points = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])

ax.scatter3D(x_points, y_points, z_points, c=z_points, cmap='Reds');
ax.plot_trisurf(x_points, y_points, z_points, linewidth=0.2, antialiased=False, cmap='summer')


ax.set_xlabel('Iteration Time')
ax.set_ylabel('Pairs')
ax.set_zlabel('Max Degree')
ax.set_title('Iteration Statistics - Cyclic-7 - 32003')

plt.show()