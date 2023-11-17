from spherical_harmonics.Method.render import renderSHModelSurface
from spherical_harmonics.Method.values import getSHModelValue

class SHModel(object):
    def __init__(self, degree_max=0, params=None, method_name='numpy') -> None:
        self.degree_max = degree_max
        if params is None:
            self.params = [0 for _ in range((self.degree_max+1)**2)]
        else:
            self.params = params
            assert len(params) == (self.degree_max+1)**2
        self.method_name = method_name
        return

    def getValue(self, phi, theta):
        return getSHModelValue(self.degree_max, phi, theta, self.params, self.method_name)

    def render(self):
        if not renderSHModelSurface(self.degree_max, self.params, self.method_name):
            print('[ERROR][SHModel::render]')
            print('\t renderSHModelSurface failed!')
            return False

        return True
