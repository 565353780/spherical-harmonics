from spherical_harmonics.Data.sh_model import SHModel

def demo():
    degree_max = 3
    params = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    method_name = 'jittor'

    sh_model = SHModel(degree_max, params, method_name)
    sh_model.render()
    return True
