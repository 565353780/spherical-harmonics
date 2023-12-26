import numpy as np
from typing import Union

from spherical_harmonics.Method.render import renderSHModelSurface
from spherical_harmonics.Method.values import getSHValues, getSHModelValue
from spherical_harmonics.Data.base_model import BaseModel

class SH3DModel(BaseModel):
    def __init__(self, degree_max: int=0, method_name: str='numpy', dtype=None) -> None:
        BaseModel.__init__(self, degree_max, method_name, dtype)
        return

    def getValue(self, phi, theta):
        return getSHModelValue(self.degree_max, phi, theta, self.params, self.method_name, self.dtype)

    def solveParams(self, phis: Union[list, np.ndarray], thetas: Union[list, np.ndarray], dists: Union[list, np.ndarray]) -> bool:
        values = np.array(getSHValues(self.degree_max, phis, thetas, 'numpy', np.float64)).transpose(1, 0)
        return BaseModel.solveParams(self, values, dists)

    def render(self):
        params = self.params
        if self.method_name in ['torch', 'jittor']:
            params = self.params.detach().cpu()

        if not renderSHModelSurface(self.degree_max, params, self.method_name):
            print('[ERROR][SHModel::render]')
            print('\t renderSHModelSurface failed!')
            return False

        return True
