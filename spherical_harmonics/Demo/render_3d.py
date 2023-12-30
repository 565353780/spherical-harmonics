import sys
sys.path.append('../data-convert/')

from spherical_harmonics.Data.sh_3d_model import SH3DModel

def demo():
    degree_max = 3
    params = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    method_name = 'numpy'

    sh_3d_model = SH3DModel(degree_max, method_name)
    sh_3d_model.setParams(params)
    sh_3d_model.render()
    return True
