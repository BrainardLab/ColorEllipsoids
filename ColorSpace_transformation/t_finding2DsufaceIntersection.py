#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 20:37:14 2024

@author: fangfang
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Ellipsoid parameters (eigenvectors and radii)
center = np.array([0, 0, 0])  # Center of the ellipsoid
radii = np.array([2.0, 1.0, 1.5])  # Semi-axes (radii)
eigenvectors = np.eye(3)  # Identity matrix, ellipsoid aligned with the axes

# Define the plane using a normal vector
normal_vector = np.array([1, 1, 1])  # Normal vector of the plane

# Ensure the normal vector is normalized
normal_vector = normal_vector / np.linalg.norm(normal_vector)

# Create two orthogonal vectors that lie on the plane
v1 = np.array([-1, 1, 0])  # An arbitrary vector on the plane
v1 = v1 - np.dot(v1, normal_vector) * normal_vector  # Make v1 orthogonal to the normal
v1 = v1 / np.linalg.norm(v1)

v2 = np.cross(normal_vector, v1)  # v2 is orthogonal to both normal_vector and v1
v2 = v2 / np.linalg.norm(v2)

# Construct matrix A for the ellipsoid
A = eigenvectors @ np.diag(1 / radii**2) @ eigenvectors.T

# Compute the quadratic form in the plane's coordinate system
M = np.array([[v1 @ A @ v1, v1 @ A @ v2],
              [v2 @ A @ v1, v2 @ A @ v2]])
eigvals, eigvecs = np.linalg.eigh(M)

# Ellipse semi-axes and rotation in the plane's local coordinates
semi_axes = 1 / np.sqrt(eigvals)
ellipse_rotation = eigvecs

# Parametrize the ellipse in the plane's local coordinate system
angles = np.linspace(0, 2 * np.pi, 100)
ellipse_local = np.array([semi_axes[0] * np.cos(angles), semi_axes[1] * np.sin(angles)])
ellipse_local_rotated = ellipse_rotation @ ellipse_local

# Transform the ellipse from the plane's local coordinates to global 3D coordinates
ellipse_3D = (center[:, None] + 
              ellipse_local_rotated[0, :] * v1[:, None] + 
              ellipse_local_rotated[1, :] * v2[:, None])

# Make the plane larger by extending the mesh grid range
plane_range = 3
x_plane, y_plane = np.meshgrid(np.linspace(-plane_range, plane_range, 10), 
                               np.linspace(-plane_range, plane_range, 10))
z_plane = center[2] + (x_plane * normal_vector[0] + y_plane * normal_vector[1]) / -normal_vector[2]

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create the ellipsoid surface using the eigenvectors and radii
u, v = np.linspace(0, 2 * np.pi, 100), np.linspace(0, np.pi, 50)
x = radii[0] * np.outer(np.cos(u), np.sin(v))
y = radii[1] * np.outer(np.sin(u), np.sin(v))
z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

ellipsoid_surface = eigenvectors @ np.array([x.flatten(), y.flatten(), z.flatten()])
x_surface = ellipsoid_surface[0, :].reshape(x.shape) + center[0]
y_surface = ellipsoid_surface[1, :].reshape(y.shape) + center[1]
z_surface = ellipsoid_surface[2, :].reshape(z.shape) + center[2]

ax.plot_surface(x_surface, y_surface, z_surface, color='lightblue', alpha=0.3)

# Plot the plane
ax.plot_surface(x_plane, y_plane, z_plane, color='lightgreen', alpha=0.5)

# Plot the intersection ellipse
ax.plot(ellipse_3D[0, :], ellipse_3D[1, :], ellipse_3D[2, :], color='r', linewidth=2)

# Labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.view_init(elev=20, azim=-120)
plt.show()