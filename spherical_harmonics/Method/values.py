import jittor
import numpy
import torch
import math
from scipy.special import sph_harm

from spherical_harmonics.Config.weights import W0, W1, W2, W3, W4
from spherical_harmonics.Method.data import toData

def isDegreeAndIdxValid(degree, idx):
    if degree < 0 or degree > 4:
        return False

    if idx < -degree or idx > degree:
        return False

    return True

def getWeight(degree, idx):
    assert isDegreeAndIdxValid(degree, idx)
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

def getDeg0Value(phi, method_name):
    try:
        return toData(numpy.ones_like(phi.detach().cpu(), dtype=float), method_name)
    except:
        return toData(numpy.ones_like(phi, dtype=float), method_name)

def getBaseValue(idx, phi, theta, method):
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

def getValue(degree, idx, phi, theta, method, method_name):
    assert isDegreeAndIdxValid(degree, idx)

    match degree:
        case 0:
            return getDeg0Value(phi, method_name)
        case 1:
            return getBaseValue(idx, phi, theta, method) * getDeg1ThetaValue(idx, theta, method)
        case 2:
            return getBaseValue(idx, phi, theta, method) * getDeg2ThetaValue(idx, theta, method)
        case 3:
            return getBaseValue(idx, phi, theta, method) * getDeg3ThetaValue(idx, theta, method)
        case 4:
            return getBaseValue(idx, phi, theta, method) * getDeg4ThetaValue(idx, theta, method)

def getSHValueWithMethod(degree, idx, phi, theta, method, method_name):
    weight = getWeight(degree, idx)
    value = getValue(degree, idx, phi, theta, method, method_name)
    assert weight is not None
    assert value is not None
    return weight * value

def getMathSHValue(degree, idx, phi, theta):
    if degree == 0:
        if isinstance(phi, list):
            return [1.0 for _ in range(len(phi))]
        return 1.0

    if not isinstance(phi, list):
        return getSHValueWithMethod(degree, idx, phi, theta, math, 'math')

    value_list = []
    for p, t in zip(phi, theta):
        value_list.append(getSHValueWithMethod(degree, idx, p, t, math, 'math'))
    return value_list

def getNumpySHValue(degree, idx, phi, theta):
    return getSHValueWithMethod(degree, idx, phi, theta, numpy, 'numpy')

def getTorchSHValue(degree, idx, phi, theta):
    return getSHValueWithMethod(degree, idx, phi, theta, torch, 'torch').to(phi.device)

def getJittorSHValue(degree, idx, phi, theta):
    return getSHValueWithMethod(degree, idx, phi, theta, jittor, 'jittor').to(phi.device)

def getScipySHValue(degree, idx, phi, theta):
    complex_value = sph_harm(abs(idx), degree, phi, theta)

    if idx < 0:
        return numpy.sqrt(2) * (-1) ** idx * complex_value.imag
    if idx > 0:
        return numpy.sqrt(2) * (-1) ** idx * complex_value.real
    return complex_value.real

def getSHValue(degree, idx, phi, theta, method_name='math'):
    assert method_name in ['math', 'numpy', 'torch', 'jittor', 'scipy']

    match method_name:
        case 'math':
            return getMathSHValue(degree, idx, phi, theta)
        case 'numpy':
            return getNumpySHValue(degree, idx, phi, theta)
        case 'torch':
            return getTorchSHValue(degree, idx, phi, theta)
        case 'jittor':
            return getJittorSHValue(degree, idx, phi, theta)
        case 'scipy':
            return getScipySHValue(degree, idx, phi, theta)

def getParamIdx(degree, idx):
    real_idx = idx + degree
    param_idx = degree**2 + real_idx
    return param_idx

def getSHModelValue(degree_max, phi, theta, params, method_name):
    assert len(params) == (degree_max+1)**2

    if method_name != 'math' or not isinstance(phi, list):
        value = 0
        for degree in range(degree_max+1):
            for idx in range(-degree, degree+1, 1):
                param_idx = getParamIdx(degree, idx)
                param = params[param_idx]
                if param == 0:
                    continue

                value += param * getSHValue(degree, idx, phi, theta, method_name)
        return value

    value_list = [0 for _ in range(len(phi))]
    for degree in range(degree_max+1):
        for idx in range(-degree, degree+1, 1):
            param_idx = getParamIdx(degree, idx)
            param = params[param_idx]
            if param == 0:
                continue
            for i in range(len(value_list)):
                value_list[i] += param * getSHValue(degree, idx, phi[i], theta[i], method_name)
    return value_list
