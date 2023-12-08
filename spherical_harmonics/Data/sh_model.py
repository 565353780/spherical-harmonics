from typing import Union

from spherical_harmonics.Method.render import renderSHModelSurface
from spherical_harmonics.Method.values import getSHModelValue
from spherical_harmonics.Method.params import getParams

class SHModel(object):
    def __init__(self, degree_max: int=0, params: Union[list, None]=None, method_name: str='numpy', dtype=None) -> None:
        self.degree_max = degree_max
        self.params = getParams(degree_max, params, method_name, dtype)
        self.method_name = method_name
        self.dtype = dtype

        assert self.params is not None
        return

    def getValue(self, phi, theta):
        return getSHModelValue(self.degree_max, phi, theta, self.params, self.method_name, self.dtype)

    def render(self):
        params = self.params
        if self.method_name in ['torch', 'jittor']:
            params = self.params.detach().cpu()

        if not renderSHModelSurface(self.degree_max, params, self.method_name):
            print('[ERROR][SHModel::render]')
            print('\t renderSHModelSurface failed!')
            return False

        return True
