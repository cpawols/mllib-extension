"""
TODO
"""
import numpy as np

from settings import Configuration
from random import randint

import reduct_feature_selection


class AbstractionClass:
    def __init__(self, table):
        self.table = table

    def _preprae_table(self, attr=None):
        """
        Ostatni jest numer obiektu
        :param table:
        :return:
        """
        if attr is not None:
            new_table = self.table[:, attr + [self.table.shape[1] - 1]]
        else:
            new_table = self.table

        return [tuple(attributes[:-1]) + (i,) for i, attributes in enumerate(new_table)]

    def get_abstraction_class(self, chunk_number, attr):
        """
        TODO
        :param chunk_number:
        :return:
        """
        system = self._preprae_table(attr)
        eav_rdd = Configuration.sc.parallelize(system, chunk_number)
        agregatted = eav_rdd.map(lambda x: (x[:-1], [x[-1]])).reduceByKey(self.r)
        abstraction_class = []
        for tuples in agregatted.collect():
            abstraction_class.append(tuples[-1])
        return abstraction_class

    def get_possitive_area(self, chunk_number, attr=None):
        """
        TODO
        :param chunk_number:
        :param attr:
        :return:
        """
        abstraction_class = self.get_abstraction_class(chunk_number, attr=attr)
        pos = []

        for ab in abstraction_class:
            a=set()
            for ob in ab:
                a.add(self.table[ob][-1])
            if len(a) == 1:
                pos = pos + ab
        return pos


    def r(selfx,x,y):
        return x + y


if __name__ == "__main__":
    table = np.array([
        [1, 1, 0, 0, 0],
        [1, 1, 0, 1, 0],
        [2, 1, 0, 0, 1],
        [0, 2, 0, 0, 1],
        [0, 0, 1, 0, 1],
        [0, 0, 1, 1, 0],
        [2, 0, 1, 1, 1],
        [1, 2, 0, 0, 0],
        [1, 0, 1, 0, 1],
        [0, 2, 1, 0, 1],
        [1, 2, 1, 1, 1],
        [2, 2, 0, 1, 1],
        [2, 2, 0, 1, 1]

    ])
    #t = np.array([[randint(1,10) for _ in range(20000)] for __ in range(1000)])
    is_consistent = AbstractionClass(table)
    print is_consistent.get_possitive_area(4)
    # print is_consistent.get_abstraction_class(4, attr=range(0, 1))




# TODO Testy
# TODO Dokumentacja
# TODO Refactoring code
