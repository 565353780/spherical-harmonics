import numpy as np
from typing import Union
from copy import deepcopy

from data_convert.Method.data import toData


class SHBaseModel(object):
    def __init__(
        self, degree_max: int = 0, method_name: str = "numpy", dtype=None
    ) -> None:
        self.degree_max = degree_max
        self.method_name = method_name
        self.dtype = dtype

        self.params = None

        self.updateDegree()
        self.updateParams()

        assert self.params is not None
        return

    def resetDegree(self) -> bool:
        if self.degree_max == 0:
            return True

        self.degree_max = 0
        self.updateParams()
        return True

    def reset(self) -> bool:
        self.resetDegree()

        self.params = toData([0.0], "numpy", np.float64)
        return True

    def toDict(self) -> dict:
        return {
            "degree_max": self.degree_max,
            "params": self.getParams().tolist(),
        }

    def loadDict(self, sh_dict: dict) -> bool:
        self.degree_max = int(sh_dict["degree_max"])
        self.params = sh_dict["params"]
        self.updateDegree()
        self.updateParams()
        return True

    def updateParams(self) -> bool:
        print("[ERROR][SHBaseModel::updateParams]")
        print("\t please finish this function first!")
        return False

    def isDegreeMax(self) -> bool:
        print("[ERROR][SHBaseModel::isDegreeMax]")
        print("\t please finish this function first!")
        return False

    def updateDegree(self) -> bool:
        self.degree_max = max(self.degree_max, 0)
        return True

    def isDegreeMin(self):
        return self.degree_max == 0

    def getDegree(self) -> int:
        return self.degree_max

    def getParams(self) -> np.ndarray:
        return deepcopy(self.params)

    def upperDegree(self):
        if self.isDegreeMax():
            return False

        self.degree_max += 1
        self.updateParams()
        return True

    def lowerDegree(self):
        if self.isDegreeMin():
            return False

        self.degree_max -= 1
        self.updateParams()
        return True

    def setParams(self, params: Union[list, np.ndarray]) -> bool:
        self.params = params

        self.updateParams()
        return True

    def solveParams(
        self, values: Union[list, np.ndarray], dists: Union[list, np.ndarray]
    ) -> bool:
        self.params = np.linalg.lstsq(values, dists, rcond=None)[0]

        self.updateParams()
        return True
