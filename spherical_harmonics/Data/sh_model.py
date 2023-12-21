import numpy as np
from typing import Union

from spherical_harmonics.Config.constant import DEGREE_MAX
from spherical_harmonics.Method.data import toData
from spherical_harmonics.Method.render import renderSHModelSurface
from spherical_harmonics.Method.values import getSHValues, getSHModelValue
from spherical_harmonics.Method.params import getParams

class SHModel(object):
    def __init__(self, degree_max: int=0, params: Union[list, None]=None, method_name: str='numpy', dtype=None) -> None:
        self.degree_max = degree_max
        self.method_name = method_name
        self.dtype = dtype

        self.params = None

        self.updateParams()

        assert self.params is not None
        return

    def updateParams(self) -> bool:
        self.params = getParams(self.degree_max, self.params, self.method_name, self.dtype)
        return True

    def upperDegree(self):
        self.degree_max += 1
        self.degree_max = min(self.degree_max, DEGREE_MAX)

        self.updateParams()
        return True

    def lowerDegree(self):
        self.degree_max -= 1
        self.degree_max = max(self.degree_max, 0)

        self.updateParams()
        return True

    def getValue(self, phi, theta):
        return getSHModelValue(self.degree_max, phi, theta, self.params, self.method_name, self.dtype)

    def solveParams(self, phis: Union[list, np.ndarray], thetas: Union[list, np.ndarray], dists: Union[list, np.ndarray]) -> bool:
        values = np.array(getSHValues(self.degree_max, phis, thetas, 'numpy', np.float64)).transpose(1, 0)
        params = np.linalg.lstsq(values, dists, rcond=-1)[0]
        self.params = toData(params, self.method_name, self.dtype)
        return True

    def render(self):
        params = self.params
        if self.method_name in ['torch', 'jittor']:
            params = self.params.detach().cpu()

        if not renderSHModelSurface(self.degree_max, params, self.method_name):
            print('[ERROR][SHModel::render]')
            print('\t renderSHModelSurface failed!')
            return False

        return True
