import numpy as np
from typing import Union

from data_convert.Method.data import toData

def get2DParams(degree_max: int, params: Union[list, None]=None, method_name: str='numpy', dtype=None):
    new_params = np.array([0.0 for _ in range(2*degree_max+1)], dtype=np.float64)

    if params is None:
        return toData(new_params, method_name, dtype)

    params = toData(params, 'numpy', np.float64)

    common_num = min(new_params.shape[0], params.shape[0])

    new_params[:common_num] = params[:common_num]

    return toData(new_params, method_name, dtype)

def get3DParams(degree_max: int, params: Union[list, None]=None, method_name: str='numpy', dtype=None):
    new_params = np.array([0.0 for _ in range((degree_max+1)**2)], dtype=np.float64)

    if params is None:
        return toData(new_params, method_name, dtype)

    params = toData(params, 'numpy', np.float64)

    common_num = min(new_params.shape[0], params.shape[0])

    new_params[:common_num] = params[:common_num]

    return toData(new_params, method_name, dtype)
