import torch
import numpy as np
from typing import Union

from spherical_harmonics.Config.degrees import DEGREE_MAX_3D
from spherical_harmonics.Method.render_3d import renderSH3DModelSurface
from spherical_harmonics.Method.values_3d import getSH3DValues, getSH3DModelValue
from spherical_harmonics.Method.params import get3DParams
from spherical_harmonics.Data.sh_base_model import SHBaseModel

class SH3DModel(SHBaseModel):
    def __init__(self, degree_max: int=0, method_name: str='numpy', dtype=None) -> None:
        SHBaseModel.__init__(self, degree_max, method_name, dtype)
        return

    def updateDegree(self) -> bool:
        SHBaseModel.updateDegree(self)
        if DEGREE_MAX_3D > 0:
            self.degree_max = min(self.degree_max, DEGREE_MAX_3D)
        return True

    def isDegreeMax(self):
        if DEGREE_MAX_3D < 0:
            return False
        return self.degree_max == DEGREE_MAX_3D

    def updateParams(self) -> bool:
        self.params = get3DParams(self.degree_max, self.params, self.method_name, self.dtype)
        return True

    def getValue(self, phi, theta):
        return getSH3DModelValue(self.degree_max, phi, theta, self.params, self.method_name, self.dtype)

    def solveParams(self, phis: Union[list, np.ndarray], thetas: Union[list, np.ndarray], dists: Union[list, np.ndarray]) -> bool:
        values = np.array(getSH3DValues(self.degree_max, phis, thetas, 'numpy', np.float64)).transpose(1, 0)
        return SHBaseModel.solveParams(self, values, dists)

    def getDiffValues(self, phis: Union[list, np.ndarray, torch.Tensor],
                      thetas: Union[list, np.ndarray, torch.Tensor],
                      dists: Union[list, np.ndarray, torch.Tensor],
                      method_name: str='torch', dtype=None) -> Union[list, np.ndarray, torch.Tensor]:
        values = np.array(getSH3DValues(self.degree_max, phis, thetas, method_name, dtype)).transpose(1, 0)
        return SHBaseModel.getDiffValues(values, dists, method_name, dtype)

    def render(self):
        params = self.params
        if self.method_name in ['torch', 'jittor']:
            params = self.params.detach().cpu()

        if not renderSH3DModelSurface(self.degree_max, params, self.method_name):
            print('[ERROR][SHModel::render]')
            print('\t renderSH3DModelSurface failed!')
            return False

        return True
