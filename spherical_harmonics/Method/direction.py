import numpy as np

def getDirection(phi, theta):
    return np.array([
        np.sin(theta) * np.sin(phi),
        np.sin(theta) * np.cos(phi),
        np.cos(theta)], dtype=float)

def getDirections(phi_array, theta_array):
    directions = np.zeros([phi_array.shape[0], theta_array.shape[0], 3])
    for i in range(phi_array.shape[0]):
        for j in range(theta_array.shape[0]):
            directions[i][j] = getDirection(phi_array[i], theta_array[j])

    return directions
