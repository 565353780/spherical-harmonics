import numpy as np
from typing import Union

from spherical_harmonics.Method.data import toData

def getParams(degree_max: int, params: Union[list, None]=None, method_name: str='numpy', dtype=None):
    new_params = np.array([0.0 for _ in range((degree_max+1)**2)], dtype=np.float64)

    if params is None:
        return toData(new_params, method_name, dtype)

    params = toData(params, 'numpy', np.float64)

    common_num = min(new_params.shape[0], params.shape[0])

    new_params[:common_num] = params

    return toData(new_params, method_name, dtype)
