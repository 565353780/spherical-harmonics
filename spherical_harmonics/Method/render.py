import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from spherical_harmonics.Method.values import getSHValue

def renderSHFunction(degree, idx, method_name='math'):
    theta = np.linspace(0, 2 * np.pi, 181)
    phi = np.linspace(0, np.pi, 91)
    theta_2d, phi_2d = np.meshgrid(theta, phi)

    Ylm = np.zeros(theta_2d.shape, dtype=float)

    for i in range(theta.shape[0]):
        for j in range(phi.shape[0]):
            Ylm[i][j] = getSHValue(degree, idx, theta[i], phi[j], method_name)

    xyz_2d = np.array(
        [
            np.sin(phi_2d) * np.sin(theta_2d),
            np.sin(phi_2d) * np.cos(theta_2d),
            np.cos(phi_2d),
        ]
    )

    r = np.abs(Ylm) * xyz_2d


    colormap = cm.ScalarMappable(cmap=plt.get_cmap("RdYlBu_r"))
    colormap.set_clim(-0.5, 0.5)
    limit = 0.5

    fig = plt.figure(figsize=plt.figaspect(1.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([0.75, 0.75, 1, 1]))
    ax.plot_surface(
        r[0], r[1], r[2], facecolors=colormap.to_rgba(Ylm.real), rstride=1, cstride=1
    )
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()
    return True
