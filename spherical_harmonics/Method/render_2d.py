import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from functools import partial
from mpl_toolkits.mplot3d import Axes3D

from data_convert.Method.data import toData

from spherical_harmonics.Config.constant import PI_H
from spherical_harmonics.Method.values_2d import getSH2DModelValue, getSH2DValue
from spherical_harmonics.Method.direction import getDirections


def render2DCurve(directions: np.ndarray, values: np.ndarray):
    r = np.abs(values) * directions.transpose(1, 0)

    colormap = cm.ScalarMappable(cmap=plt.get_cmap("RdYlBu_r"))
    colormap.set_clim(-0.5, 0.5)
    limit = 0.5

    fig = plt.figure(figsize=plt.figaspect(1.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.get_proj = lambda: np.dot(
        Axes3D.get_proj(ax), np.diag([0.75, 0.75, 1, 1]))
    ax.plot(r[0], r[1], r[2])
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()
    return True


def renderBatchSH2DFunction(sh_function, method_name):
    phi = np.linspace(0, 2 * np.pi, 181)

    Ylm = toData(
        sh_function(toData(phi, method_name), method_name=method_name),
        "numpy",
        np.float64,
    )

    xyz_2d = getDirections(phi, np.array([PI_H], dtype=float)).reshape(-1, 3)

    if not render2DCurve(xyz_2d, Ylm):
        print("[ERROR][render::renderBatchSHFunction]")
        print("\t render2DCurve failed!")
        return False

    return True


def renderSH2DFunction(sh_function, method_name, use_batch=True):
    if use_batch and method_name != "math":
        return renderBatchSH2DFunction(sh_function, method_name)

    phi = np.linspace(0, 2 * np.pi, 181)

    Ylm = np.zeros(phi.shape[0], dtype=float)

    for i in range(phi.shape[0]):
        Ylm[i] = toData(
            sh_function(toData([phi[i]], method_name),
                        method_name=method_name),
            "numpy",
            np.float64,
        )

    xyz_2d = np.zeros([phi.shape[0], 3])
    for i in range(phi.shape[0]):
        xyz_2d[i] = [np.sin(phi[i]), np.cos(phi[i]), 0]

    if not render2DCurve(xyz_2d, Ylm):
        print("[ERROR][render::renderSHFunction]")
        print("\t render2DCurve failed!")
        return False

    return True


def renderSH2DCurve(degree, idx, method_name="math"):
    sh_function = partial(getSH2DValue, degree, idx)
    return renderSH2DFunction(sh_function, method_name)


def renderSH2DModelCurve(degree_max, params, method_name="math"):
    sh_function = partial(getSH2DModelValue, degree_max, params=params)
    return renderSH2DFunction(sh_function, method_name)
