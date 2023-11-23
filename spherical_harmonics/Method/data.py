import jittor
import numpy
import torch

def list2numpy(data: list, dtype=numpy.float64):
    return numpy.array(data, dtype=dtype)

def list2torch(data: list, dtype=torch.float64):
    return torch.Tensor(data).type(dtype)

def list2jittor(data: list, dtype=jittor.float64):
    return jittor.Var(data, dtype)

def numpy2list(data: numpy.ndarray):
    return data.tolist()

def numpy2torch(data: numpy.ndarray):
    return torch.from_numpy(data)

def numpy2jittor(data: numpy.ndarray):
    return jittor.Var(data)

def torch2numpy(data: torch.Tensor):
    if data.device != 'cpu':
        return data.cpu().numpy()
    return data.numpy()

def torch2list(data: torch.Tensor):
    return torch2numpy(data).tolist()

def torch2jittor(data: torch.Tensor):
    return jittor.Var(torch2numpy(data))

def jittor2numpy(data: jittor.Var):
    return data.numpy()

def jittor2list(data: jittor.Var):
    return jittor2numpy(data).tolist()

def jittor2torch(data: jittor.Var):
    return torch.from_numpy(jittor2numpy(data))

def toList(params):
    if isinstance(params, list):
        return params

    if isinstance(params, numpy.ndarray):
        return numpy2list(params)

    if isinstance(params, torch.Tensor):
        return torch2list(params)

    if isinstance(params, jittor.Var):
        return jittor2list(params)

    print('[ERROR][data::toList]')
    print('\t method not defined for input data type!')
    print('\t params.type:', type(params))
    return None

def toNumpy(params, dtype=numpy.float64):
    if isinstance(params, numpy.ndarray):
        return params.astype(dtype)

    if isinstance(params, list):
        return list2numpy(params, dtype)

    if isinstance(params, torch.Tensor):
        return torch2numpy(params)

    if isinstance(params, jittor.Var):
        return jittor2numpy(params)

    print('[ERROR][data::toNumpy]')
    print('\t method not defined for input data type!')
    print('\t params.type:', type(params))
    return None

def toTorch(params, dtype=torch.float64):
    if isinstance(params, torch.Tensor):
        return params

    if isinstance(params, list):
        return list2torch(params, dtype)

    if isinstance(params, numpy.ndarray):
        return numpy2torch(params)

    if isinstance(params, jittor.Var):
        return jittor2torch(params)

    print('[ERROR][data::toTorch]')
    print('\t method not defined for input data type!')
    print('\t params.type:', type(params))
    return None

def toJittor(params, dtype=jittor.float64):
    if isinstance(params, jittor.Var):
        params.astype(params, dtype)
        return params

    if isinstance(params, list):
        return list2jittor(params, dtype)

    if isinstance(params, numpy.ndarray):
        return numpy2jittor(params)

    print('[ERROR][data::toJittor]')
    print('\t method not defined for input data type!')
    print('\t params.type:', type(params))
    return None

def toData(params, method_name, dtype=None):
    match method_name:
        case 'math':
            return toList(params)
        case 'numpy':
            if dtype is None:
                return toNumpy(params)
            return toNumpy(params, dtype)
        case 'torch':
            if dtype is None:
                return toTorch(params)
            return toTorch(params, dtype)
        case 'jittor':
            if dtype is None:
                return toJittor(params)
            return toJittor(params, dtype)
