import numpy as np
from typing import Union

from spherical_harmonics.Method.render_2d import renderSH2DModelCurve
from spherical_harmonics.Method.values_2d import getSH2DValues, getSH2DModelValue
from spherical_harmonics.Method.params import get2DParams
from spherical_harmonics.Data.sh_base_model import SHBaseModel

class SH2DModel(SHBaseModel):
    def __init__(self, degree_max: int=0, method_name: str='numpy', dtype=None) -> None:
        SHBaseModel.__init__(self, degree_max, method_name, dtype)
        return

    def updateParams(self) -> bool:
        self.params = get2DParams(self.degree_max, self.params, self.method_name, self.dtype)
        return True

    def getValue(self, phi):
        return getSH2DModelValue(self.degree_max, phi, self.params, self.method_name, self.dtype)

    def solveParams(self, phis: Union[list, np.ndarray], dists: Union[list, np.ndarray]) -> bool:
        values = np.array(getSH2DValues(self.degree_max, phis, 'numpy', np.float64)).transpose(1, 0)
        return SHBaseModel.solveParams(self, values, dists)

    def render(self):
        params = self.params
        if self.method_name in ['torch', 'jittor']:
            params = self.params.detach().cpu()

        if not renderSH2DModelCurve(self.degree_max, params, self.method_name):
            print('[ERROR][SHModel::render]')
            print('\t renderSH2DModelCurve failed!')
            return False

        return True
