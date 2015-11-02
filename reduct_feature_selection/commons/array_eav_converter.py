"""
A class for converting numpy array to EAV format
"""
import operator
import numpy as np
__author__ = 'krzysztof'


class EavConverter:

    def __init__(self, eav):
        self.eav = eav

    @staticmethod
    def convert_to_eav(array):
        """
        this function convert numpy array to eav format (list of tuples (enitity, attribute, value))
        :param array: numpy array
        :return: list of eav tuples
        """
        if array.size:
            rows = range(array.shape[0])
            colnames = array.dtype.names
            list_of_eav = ([(r, c, array[c][r]) for c in colnames] for r in rows)
            return reduce(operator.add, list_of_eav)
        return []

    @staticmethod
    def convert_to_array(eav):
        """
        this function convert eav format (list of tuples (enitity, attribute, value)) to numpy array
        :param eav: array list of eav tuples
        :return: numpy
        """
        if eav:
            rows_size = max([x[0] for x in eav])
            cols_size = len(eav)/(rows_size+1)
            formats = list(set([(x[1], float) for x in eav]))
            array = np.array([tuple([0]*(rows_size+1))]*cols_size, dtype=formats)
            for t in eav:
                array[t[0]][t[1]] = t[2]
            return array
        return np.array([])
