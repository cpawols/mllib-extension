"""
A class for converting numpy array to EAV format
"""
import operator
import numpy as np

__author__ = 'krzysztof'


class Eav:
    def __init__(self, eav):
        self.eav = eav
        self.dec = {}
        self._obj_index = self.update_index(0)
        self._attr_index = self.update_index(1)

    @classmethod
    def from_array(cls, array):
        return cls(cls.convert_to_eav(array))

    @property
    def dec(self):
        return self.dec

    @dec.setter
    def dec(self, value):
        self.dec = value

    # TODO: convert to format with dtype.names

    @staticmethod
    def convert_to_proper_format(array):
        ncols = len(array[0])
        formats = [('C' + str(i + 1), type(array[0][i])) for i in range(ncols)]
        new_array = map(lambda row: tuple(row), array)
        return np.array(new_array, dtype=formats)

    @staticmethod
    def convert_to_proper_array(array):
        new_array = map(lambda row: list(row), array)
        return np.array(new_array)

    @staticmethod
    def convert_to_eav(array):
        """
        this function convert numpy array to eav format (list of tuples (enitity, attribute, value))
        :param array: numpy array
        :return: list of eav tuples
        """
        if array.size:
            Eav.convert_to_proper_format(array)
            rows = range(array.shape[0])
            colnames = array.dtype.names
            list_of_eav = ([(r, c, array[c][r]) for c in colnames] for r in rows)
            return reduce(operator.add, list_of_eav)
        return []

    def convert_to_array(self):
        """
        this function convert eav format (list of tuples (entity, attribute, value)) to numpy array
        :param eav: array list of eav tuples
        :return: numpy
        """
        if self.eav:
            rows_size = max([x[0] for x in self.eav])
            cols_size = len(self.eav) / (rows_size + 1)
            formats = sorted(list(set([(x[1], float) for x in self.eav])))
            array = np.array([tuple([0] * (cols_size))] * (rows_size + 1),
                             dtype=formats)
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

    def get_obj_attrs(self, obj):
        return [self.eav[ind][2] for ind in self._obj_index[obj]]

    def get_obj_count(self):
        objects = [x[0] for x in self.eav]
        return len(set(objects))

    def get_attr_count(self):
        objects = [x[1] for x in self.eav]
        return len(set(objects))

    def sort(self, sc=None):
        if sc is not None:
            eav_rdd = sc.parallelize(self.eav)
            self.eav = eav_rdd.map(lambda x: ((x[1], x[2], x[0]), 1)).sortByKey()\
                .map(lambda (k, v): (k[2], k[0], k[1])).collect()
        else:
            self.eav = sorted(self.eav, key=lambda x: (x[1], x[2], x[0]))
        self.update_index(0)
        self.update_index(1)

    @staticmethod
    def _compare(iterator):
        yield sorted(iterator, key=lambda x: (x[1], x[2], x[0]))

    def merge_sort(self, sc):
        num_chunks = 10
        eav_rdd_part = sc.parallelize(self.eav, num_chunks)
        self.eav = eav_rdd_part.mapPartitions(Eav._compare)\
            .reduce(lambda x, y: sorted(x + y, key=lambda x: (x[1], x[2], x[0])))
        self.update_index(0)
        self.update_index(1)

    def is_consistent(self):
        same_dec = [(ob1, ob2) for ob1, dec1 in self.dec.iteritems() for ob2, dec2 in self.dec.iteritems()
                    if not dec1 == dec2 and not ob1 == ob2]
        for (ob1, ob2) in same_dec:
            if self.get_obj_attrs(ob1) == self.get_obj_attrs(ob2):
                return False
        return True

if __name__ == "__main__":
    None
