import numpy

from data_convert.Method.data import toData

def getDeg0Value(phi, method_name):
    try:
        return toData(numpy.ones_like(phi.detach().cpu(), dtype=float), method_name)
    except:
        return toData(numpy.ones_like(phi, dtype=float), method_name)
