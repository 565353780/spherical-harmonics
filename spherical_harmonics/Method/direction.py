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

def getPhiTheta(direction):
    norm = np.linalg.norm(direction)

    if norm == 0:
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

def getPhisThetas(directions):
    norm = np.linalg.norm(directions, axis=1)

    valid_mask = (norm > 0) & (directions[:, 2] != 1.0)

    valid_norm = norm[valid_mask]

    valid_norm = np.vstack([valid_norm, valid_norm, valid_norm]).transpose(1, 0)

    norm_directions = np.zeros_like(directions, dtype=float)
    norm_directions[:, 2] = 1.0

    norm_directions[valid_mask] = directions[valid_mask] / valid_norm

    thetas = np.arccos(norm_directions[:, 2])

    sin_thetas = np.sin(thetas[valid_mask])

    phis = np.zeros_like(thetas, dtype=float)

    phis[valid_mask] = np.arctan2(
        norm_directions[valid_mask][:, 0] / sin_thetas,
        norm_directions[valid_mask][:, 1] / sin_thetas)

    add_idx = phis < 0

    phis[add_idx] += 2.0 * np.pi

    return phis, thetas
