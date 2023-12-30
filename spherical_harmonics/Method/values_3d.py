import jittor
import numpy
import torch
import math
from scipy.special import sph_harm

from data_convert.Method.data import toData

from spherical_harmonics.Config.weights import W0, W1, W2, W3, W4
from spherical_harmonics.Method.values_base import getDeg0Value

def get3DParamIdx(degree, idx):
    real_idx = idx + degree
    param_idx = degree**2 + real_idx
    return param_idx

def is3DDegreeAndIdxValid(degree, idx):
    if degree < 0 or degree > 4:
        return False

    if idx < -degree or idx > degree:
        return False

    return True

def get3DWeight(degree, idx):
    assert is3DDegreeAndIdxValid(degree, idx)
    real_idx = abs(idx)
    match degree:
        case 0:
            return W0[real_idx]
        case 1:
            return W1[real_idx]
        case 2:
            return W2[real_idx]
        case 3:
            return W3[real_idx]
        case 4:
            return W4[real_idx]

def get3DBaseValue(idx, phi, theta, method):
    if idx == 0:
        return 1.0

    st = method.sin(theta) ** abs(idx)

    if idx > 0:
        return method.cos(1.0 * idx * phi) * st

    return method.sin(-1.0 * idx * phi) * st

def getDeg1ThetaValue(idx, theta, method):
    match abs(idx):
        case 0:
            return method.cos(theta)
        case 1:
            return 1.0

def getDeg2ThetaValue(idx, theta, method):
    match abs(idx):
        case 0:
            ct = method.cos(theta)
            return 3.0 * ct * ct - 1.0
        case 1:
            return method.cos(theta)
        case 2:
            return 1.0

def getDeg3ThetaValue(idx, theta, method):
    match abs(idx):
        case 0:
            ct = method.cos(theta)
            return (5.0 * ct * ct - 3.0) * ct
        case 1:
            ct = method.cos(theta)
            return (5.0 * ct * ct - 1.0)
        case 2:
            return method.cos(theta)
        case 3:
            return 1.0

def getDeg4ThetaValue(idx, theta, method):
    match abs(idx):
        case 0:
            ct = method.cos(theta)
            return (35.0 * ct * ct - 30.0) * ct * ct + 3.0
        case 1:
            ct = method.cos(theta)
            return (7.0 * ct * ct - 3.0) * ct
        case 2:
            ct = method.cos(theta)
            return (7.0 * ct * ct - 1.0)
        case 3:
            return method.cos(theta)
        case 4:
            return 1.0

def get3DValue(degree, idx, phi, theta, method, method_name):
    assert is3DDegreeAndIdxValid(degree, idx)

    match degree:
        case 0:
            return getDeg0Value(phi, method_name)
        case 1:
            return get3DBaseValue(idx, phi, theta, method) * getDeg1ThetaValue(idx, theta, method)
        case 2:
            return get3DBaseValue(idx, phi, theta, method) * getDeg2ThetaValue(idx, theta, method)
        case 3:
            return get3DBaseValue(idx, phi, theta, method) * getDeg3ThetaValue(idx, theta, method)
        case 4:
            return get3DBaseValue(idx, phi, theta, method) * getDeg4ThetaValue(idx, theta, method)

def getSH3DValueWithMethod(degree, idx, phi, theta, method, method_name):
    weight = get3DWeight(degree, idx)
    value = get3DValue(degree, idx, phi, theta, method, method_name)
    assert weight is not None
    assert value is not None
    return weight * value

def getMathSH3DValue(degree, idx, phi, theta):
    if degree == 0:
        if isinstance(phi, list):
            return [1.0 for _ in range(len(phi))]
        return 1.0

    if not isinstance(phi, list):
        return getSH3DValueWithMethod(degree, idx, phi, theta, math, 'math')

    value_list = []
    for p, t in zip(phi, theta):
        value_list.append(getSH3DValueWithMethod(degree, idx, p, t, math, 'math'))
    return value_list

def getNumpySH3DValue(degree, idx, phi, theta, dtype=numpy.float64):
    return toData(getSH3DValueWithMethod(degree, idx, phi, theta, numpy, 'numpy'), 'numpy', dtype)

def getTorchSH3DValue(degree, idx, phi, theta, dtype=torch.float64):
    return toData(getSH3DValueWithMethod(degree, idx, phi, theta, torch, 'torch').to(phi.device), 'torch', dtype).to(phi.device)

def getJittorSH3DValue(degree, idx, phi, theta, dtype=jittor.float64):
    return toData(getSH3DValueWithMethod(degree, idx, phi, theta, jittor, 'jittor').to(phi.device), 'jittor', dtype).to(phi.device)

def getScipySH3DValue(degree, idx, phi, theta):
    complex_value = sph_harm(abs(idx), degree, phi, theta)

    if idx < 0:
        return numpy.sqrt(2) * (-1) ** idx * complex_value.imag
    if idx > 0:
        return numpy.sqrt(2) * (-1) ** idx * complex_value.real
    return complex_value.real

def getSH3DValue(degree, idx, phi, theta, method_name='math', dtype=None):
    assert method_name in ['math', 'numpy', 'torch', 'jittor', 'scipy']

    match method_name:
        case 'math':
            return getMathSH3DValue(degree, idx, phi, theta)
        case 'numpy':
            if dtype is None:
                return getNumpySH3DValue(degree, idx, phi, theta)
            return getNumpySH3DValue(degree, idx, phi, theta, dtype)
        case 'torch':
            if dtype is None:
                return getTorchSH3DValue(degree, idx, phi, theta)
            return getTorchSH3DValue(degree, idx, phi, theta, dtype)
        case 'jittor':
            if dtype is None:
                return getJittorSH3DValue(degree, idx, phi, theta)
            return getJittorSH3DValue(degree, idx, phi, theta, dtype)
        case 'scipy':
            return getScipySH3DValue(degree, idx, phi, theta)

def getSH3DValues(degree_max, phi, theta, method_name, dtype=None):
    values = []

    if method_name != 'math' or not isinstance(phi, list):
        for degree in range(degree_max+1):
            for idx in range(-degree, degree+1, 1):
                values.append(getSH3DValue(degree, idx, phi, theta, method_name, dtype))
        return values

    values_list = [[] for _ in range(len(phi))]
    for degree in range(degree_max+1):
        for idx in range(-degree, degree+1, 1):
            for i in range(len(values_list)):
                values_list[i].append(getSH3DValue(degree, idx, phi[i], theta[i], method_name))
    return values_list

def getSH3DModelValue(degree_max, phi, theta, params, method_name, dtype=None):
    assert len(params) == (degree_max+1)**2

    values = getSH3DValues(degree_max, phi, theta, method_name, dtype)

    if method_name != 'math' or not isinstance(phi, list):
        value = 0
        for i in range(len(params)):
            value += params[i] * values[i]

        return value

    value_list = [0 for _ in range(len(phi))]

    for i in range(len(value_list)):
        for j in range(len(params)):
            value_list[i] += params[j] * values[i][j]

    return value_list
