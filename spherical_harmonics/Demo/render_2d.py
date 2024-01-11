import sys

sys.path.append("../data-convert/")


def demo():
    from spherical_harmonics.Data.sh_2d_model import SH2DModel

    degree_max = 3
    params = [0, 0, 0, 0, 0, 1, 0]
    method_name = "numpy"

    sh_2d_model = SH2DModel(degree_max, method_name)
    sh_2d_model.setParams(params)
    sh_2d_model.render()
    return True
