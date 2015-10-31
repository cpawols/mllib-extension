import operator
import numpy as np
__author__ = 'krzysztof'


class EavFormatter:
    """A class for converting numpy array to EAV format"""
    def __init__(self, eav):
        self.eav = eav

    @staticmethod
    def from_array(array):
        rows = range(array.shape[0])
        cols = array.dtype.names
        list_of_eav = [[(r, c, array[c][r]) for c in cols] for r in rows]
        return reduce(operator.add, list_of_eav)

    @staticmethod
    def to_array(eav):
        rows = max([x[0] for x in eav])
        cols = len(eav)/(rows+1)
        formats = list(set([(x[1], float) for x in eav]))
        array = np.array([tuple([0]*(rows+1))]*cols, dtype=formats)
        for t in eav:
            array[t[0]][t[1]] = t[2]
        return array







