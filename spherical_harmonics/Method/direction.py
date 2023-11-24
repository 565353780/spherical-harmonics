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

def getParam(direction):
    norm = np.linalg.norm(direction)

    if np.linalg.norm(direction) == 0:
        print('[ERROR][direction::getParam]')
        print('\t direction is 0 vector!')
        return None, None

    norm_direction = np.array(direction) / norm

    theta = np.arccos(norm_direction[2])

    sin_theta = np.sin(theta)
    if sin_theta == 0:
        return 0, theta

    phi = np.arctan2(norm_direction[0] / sin_theta, norm_direction[1] / sin_theta)
    if phi < 0:
        phi += 2.0 * np.pi

    return phi, theta
