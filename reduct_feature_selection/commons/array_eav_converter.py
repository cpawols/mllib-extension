"""
A class for converting numpy array to EAV format
"""
import operator
import numpy as np
from settings import conf, sc

__author__ = 'krzysztof'


class Eav:
    def __init__(self, eav):
        self.eav = eav
        self._obj_index = self.update_index(0)
        self._attr_index = self.update_index(1)

    @classmethod
    def from_array(cls, array):
        return cls(cls.convert_to_eav(array))

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

    def convert_to_array(self):
        """
        this function convert eav format (list of tuples (enitity, attribute, value)) to numpy array
        :param eav: array list of eav tuples
        :return: numpy
        """
        if self.eav:
            rows_size = max([x[0] for x in self.eav])
            cols_size = len(self.eav) / (rows_size + 1)
            formats = list(set([(x[1], float) for x in self.eav]))
            array = np.array([tuple([0] * (rows_size + 1))] * cols_size, dtype=formats)
            for t in self.eav:
                array[t[0]][t[1]] = t[2]
            return array
        return np.array([])

    def update_index(self, attr_obj):
        objects = set([x[attr_obj] for x in self.eav])
        index = {x: [] for x in objects}
        for ind, obj in enumerate(self.eav):
            index[obj[attr_obj]].append(ind)
        return index

    def get_object(self, obj):
        return self._obj_index[obj]

    def get_attribute(self, attr):
        return self._attr_index[attr]

    def get_obj_count(self):
        objects = [x[0] for x in self.eav]
        return len(set(objects))

    def get_attr_count(self):
        objects = [x[1] for x in self.eav]
        return len(set(objects))

    def sort(self):
        eav_rdd = sc.parallelize(self.eav)
        self.eav = eav_rdd.map(lambda x: ((x[1], x[2], x[0]), 1)).sortByKey()\
            .map(lambda (k, v): (k[2], k[0], k[1])).collect()
        self.update_index(0)
        self.update_index(1)

    @staticmethod
    def _compare(iterator):
        yield sorted(iterator, key=lambda x: (x[1], x[2], x[0]))

    def merge_sort(self):
        num_chunks = 10
        eav_rdd_part = sc.parallelize(self.eav, num_chunks)
        self.eav = eav_rdd_part.mapPartitions(Eav._compare)\
            .reduce(lambda x, y: sorted(x+y)).collect()
        self.update_index(0)
        self.update_index(1)

if __name__ == "__main__":
    None
