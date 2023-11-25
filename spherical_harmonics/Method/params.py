from typing import Union

from spherical_harmonics.Method.data import toData

def getParams(degree_max: int, params: Union[list, None]=None, method_name: str='numpy', dtype=None):
    if params is None:
        params = [0 for _ in range((degree_max+1)**2)]

    if len(params) != (degree_max+1)**2:
        print('[ERROR][params::getParams]')
        print('\t params size not matched with degree_max!')
        print('\t params size =', len(params), ', degree_max =', degree_max)
        return None

    return toData(params, method_name, dtype)
