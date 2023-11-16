import jittor
import numpy
import torch
import math
from scipy.special import sph_harm

from spherical_harmonics.Config.weights import W0, W1, W2, W3

def isDegreeAndIdxValid(degree, idx):
    if degree < 0 or degree > 3:
        return False

    if idx < -degree or idx > degree:
        return False

    return True


def getWeight(degree, idx):
    assert isDegreeAndIdxValid(degree, idx)
    real_idx = idx + degree
    match degree:
        case 0:
            return W0[real_idx]
        case 1:
            return W1[real_idx]
        case 2:
            return W2[real_idx]
        case 3:
            return W3[real_idx]

def getValue(degree, idx, theta, phi, method):
    assert isDegreeAndIdxValid(degree, idx)
    match degree:
        case 0:
            return 1
        case 1:
            match idx:
                case 0:
                    ct = method.cos(theta)
                    return ct
                case 1:
                    st = method.sin(theta)
                    cp = method.cos(phi)
                    return st * cp
                case -1:
                    st = method.sin(theta)
                    sp = method.sin(phi)
                    return st * sp
        case 2:
            match idx:
                case 0:
                    ct = method.cos(theta)
                    return 3.0 * ct * ct - 1.0
                case 1:
                    ct = method.cos(theta)
                    st = method.sin(theta)
                    cp = method.cos(phi)
                    return st * ct * cp
                case -1:
                    ct = method.cos(theta)
                    st = method.sin(theta)
                    sp = method.sin(phi)
                    return st * ct * sp
                case 2:
                    st = method.sin(theta)
                    c2p = method.cos(2.0 * phi)
                    return st * st * c2p
                case -2:
                    st = method.sin(theta)
                    s2p = method.sin(2.0 * phi)
                    return st * st * s2p
        case 3:
            match idx:
                case 0:
                    ct = method.cos(theta)
                    return (5.0 * ct * ct - 3.0) * ct
                case 1:
                    ct = method.cos(theta)
                    st = method.sin(theta)
                    cp = method.cos(phi)
                    return (5.0 * ct * ct - 1.0) * st * cp
                case -1:
                    ct = method.cos(theta)
                    st = method.sin(theta)
                    sp = method.sin(phi)
                    return (5.0 * ct * ct - 1.0) * st * sp
                case 2:
                    ct = method.cos(theta)
                    st = method.sin(theta)
                    c2p = method.cos(2.0 * phi)
                    sp = method.sin(phi)
                    return ct * st * st * c2p
                case -2:
                    ct = method.cos(theta)
                    st = method.sin(theta)
                    s2p = method.sin(2.0 * phi)
                    return ct * st * st * s2p
                case 3:
                    st = method.sin(theta)
                    c3p = method.cos(3.0 * phi)
                    return st * st * st * c3p
                case -3:
                    st = method.sin(theta)
                    s3p = method.sin(3.0 * phi)
                    return st * st * st * s3p

def getSHValueWithMethod(degree, idx, theta, phi, method):
    weight = getWeight(degree, idx)
    value = getValue(degree, idx, theta, phi, method)
    assert weight is not None
    assert value is not None
    return weight * value

def getMathSHValue(degree, idx, theta, phi):
    return getSHValueWithMethod(degree, idx, theta, phi, math)

def getNumpySHValue(degree, idx, theta, phi):
    return getSHValueWithMethod(degree, idx, theta, phi, numpy)

def getTorchSHValue(degree, idx, theta, phi):
    return getSHValueWithMethod(degree, idx, theta, phi, torch)

def getJittorSHValue(degree, idx, theta, phi):
    return getSHValueWithMethod(degree, idx, theta, phi, jittor)

def getScipySHValue(degree, idx, theta, phi):
    complex_value = sph_harm(abs(idx), degree, theta, phi)

    if idx < 0:
        return numpy.sqrt(2) * (-1) ** idx * complex_value.imag
    if idx > 0:
        return numpy.sqrt(2) * (-1) ** idx * complex_value.real
    return complex_value.real

def getSHValue(degree, idx, theta, phi, method_name='math'):
    assert method_name in ['math', 'numpy', 'torch', 'jittor', 'scipy']

    match method_name:
        case 'math':
            return getMathSHValue(degree, idx, theta, phi)
        case 'numpy':
            return getNumpySHValue(degree, idx, theta, phi)
        case 'torch':
            return getTorchSHValue(degree, idx, theta, phi)
        case 'jittor':
            return getJittorSHValue(degree, idx, theta, phi)
        case 'scipy':
            return getScipySHValue(degree, idx, theta, phi)
