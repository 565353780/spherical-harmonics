import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from functools import partial
from mpl_toolkits.mplot3d import Axes3D

from data_convert.Method.data import toData

from spherical_harmonics.Method.values_3d import getSH3DModelValue, getSH3DValue
from spherical_harmonics.Method.direction import getDirections

def render3DSurface(directions: np.ndarray, values: np.ndarray):
    r = np.abs(values) * directions.transpose(2, 0, 1)

    colormap = cm.ScalarMappable(cmap=plt.get_cmap("RdYlBu_r"))
    colormap.set_clim(-0.5, 0.5)
    limit = 0.5

    fig = plt.figure(figsize=plt.figaspect(1.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([0.75, 0.75, 1, 1]))
    ax.plot_surface(
        r[0], r[1], r[2], facecolors=colormap.to_rgba(values), rstride=1, cstride=1
    )
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()
    return True

def renderBatchSH3DFunction(sh_function, method_name):
    phi = np.linspace(0, 2 * np.pi, 181)
    theta = np.linspace(0, np.pi, 91)

    theta_2d, phi_2d = np.meshgrid(theta, phi)

    Ylm = toData(sh_function(
        toData(phi_2d, method_name),
        toData(theta_2d, method_name), method_name=method_name),
        'numpy', np.float64)

    xyz_2d = getDirections(phi, theta)

    if not render3DSurface(xyz_2d, Ylm):
        print('[ERROR][render::renderBatchSHFunction]')
        print('\t render3DSurface failed!')
        return False

    return True

def renderSH3DFunction(sh_function, method_name, use_batch=True):
    if use_batch and method_name != 'math':
        return renderBatchSH3DFunction(sh_function, method_name)

    phi = np.linspace(0, 2 * np.pi, 181)
    theta = np.linspace(0, np.pi, 91)

    Ylm = np.zeros([phi.shape[0], theta.shape[0]], dtype=float)

    for i in range(phi.shape[0]):
        for j in range(theta.shape[0]):
            Ylm[i][j] = toData(sh_function(
                toData([phi[i]], method_name),
                toData([theta[j]], method_name), method_name=method_name),
                'numpy', np.float64)

    xyz_2d = np.zeros([phi.shape[0], theta.shape[0], 3])
    for i in range(phi.shape[0]):
        for j in range(theta.shape[0]):
            xyz_2d[i][j] = [
                np.sin(theta[j]) * np.sin(phi[i]),
                np.sin(theta[j]) * np.cos(phi[i]),
                np.cos(theta[j]),
            ]

    if not render3DSurface(xyz_2d, Ylm):
        print('[ERROR][render::renderSHFunction]')
        print('\t render3DSurface failed!')
        return False

    return True

def renderSH3DSurface(degree, idx, method_name='math'):
    sh_function = partial(getSH3DValue, degree, idx)
    return renderSH3DFunction(sh_function, method_name)

def renderSH3DModelSurface(degree_max, params, method_name='math'):
    sh_function = partial(getSH3DModelValue, degree_max, params=params)
    return renderSH3DFunction(sh_function, method_name)
