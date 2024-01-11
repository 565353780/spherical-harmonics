import jittor
import numpy
import torch
import math
from scipy.special import sph_harm

from data_convert.Method.data import toData

from spherical_harmonics.Config.constant import PI
from spherical_harmonics.Method.values_base import getDeg0Value


def get2DValue(degree, idx, phi, method, method_name):
    if degree == 0:
        return getDeg0Value(phi, method_name)

    if idx == 0:
        return method.cos(1.0 * degree * phi)
    return method.sin(1.0 * degree * phi)


def getSH2DValueWithMethod(degree, idx, phi, method, method_name):
    value = get2DValue(degree, idx, phi, method, method_name)
    assert value is not None
    return value


def getMathSH2DValue(degree, idx, phi):
    if degree == 0:
        if isinstance(phi, list):
            return [1.0 for _ in range(len(phi))]
        return 1.0

    if not isinstance(phi, list):
        return getSH2DValueWithMethod(degree, idx, phi, math, "math")

    value_list = []
    for p in phi:
        value_list.append(getSH2DValueWithMethod(degree, idx, p, math, "math"))
    return value_list


def getNumpySH2DValue(degree, idx, phi, dtype=numpy.float64):
    return toData(
        getSH2DValueWithMethod(degree, idx, phi, numpy,
                               "numpy"), "numpy", dtype
    )


def getTorchSH2DValue(degree, idx, phi, dtype=torch.float64):
    return toData(
        getSH2DValueWithMethod(degree, idx, phi, torch,
                               "torch"), "torch", dtype
    ).to(phi.device)


def getJittorSH2DValue(degree, idx, phi, dtype=jittor.float64):
    return toData(
        getSH2DValueWithMethod(degree, idx, phi, jittor,
                               "jittor").to(phi.device),
        "jittor",
        dtype,
    ).to(phi.device)


def getScipySH2DValue(degree, idx, phi):
    complex_value = sph_harm(abs(idx), degree, phi, PI)

    if idx < 0:
        return numpy.sqrt(2) * (-1) ** idx * complex_value.imag
    if idx > 0:
        return numpy.sqrt(2) * (-1) ** idx * complex_value.real
    return complex_value.real


def getSH2DValue(degree, idx, phi, method_name="math", dtype=None):
    assert method_name in ["math", "numpy", "torch", "jittor", "scipy"]

    match method_name:
        case "math":
            return getMathSH2DValue(degree, idx, phi)
        case "numpy":
            if dtype is None:
                return getNumpySH2DValue(degree, idx, phi)
            return getNumpySH2DValue(degree, idx, phi, dtype)
        case "torch":
            if dtype is None:
                return getTorchSH2DValue(degree, idx, phi)
            return getTorchSH2DValue(degree, idx, phi, dtype)
        case "jittor":
            if dtype is None:
                return getJittorSH2DValue(degree, idx, phi)
            return getJittorSH2DValue(degree, idx, phi, dtype)
        case "scipy":
            return getScipySH2DValue(degree, idx, phi)


def getSH2DValues(degree_max, phi, method_name, dtype=None):
    values = []

    if method_name != "math" or not isinstance(phi, list):
        values.append(getSH2DValue(0, 0, phi, method_name, dtype))
        for degree in range(1, degree_max + 1):
            values.append(getSH2DValue(degree, 0, phi, method_name, dtype))
            values.append(getSH2DValue(degree, 1, phi, method_name, dtype))
        return values

    values_list = [[] for _ in range(len(phi))]
    for i in range(len(values_list)):
        values_list[i].append(getSH2DValue(0, 0, phi[i], method_name))
    for degree in range(1, degree_max + 1):
        for i in range(len(values_list)):
            values_list[i].append(getSH2DValue(degree, 0, phi[i], method_name))
            values_list[i].append(getSH2DValue(degree, 1, phi[i], method_name))
    return values_list


def getSH2DModelValue(degree_max, phi, params, method_name, dtype=None):
    assert len(params) == 2 * degree_max + 1

    values = getSH2DValues(degree_max, phi, method_name, dtype)

    if method_name != "math" or not isinstance(phi, list):
        value = 0
        for i in range(len(params)):
            value += params[i] * values[i]

        return value

    value_list = [0 for _ in range(len(phi))]

    for i in range(len(value_list)):
        for j in range(len(params)):
            value_list[i] += params[j] * values[i][j]

    return value_list
