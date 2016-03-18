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
        agregatted = eav_rdd.map(lambda x: (x[:-1], [x[-1]])).reduceByKey(lambda x,y : x + y)
        abstraction_class = []
        for tuples in agregatted.collect():
            abstraction_class.append(tuples[-1])
        return abstraction_class


    def prep_tab(self, attr):
        if attr is not None:
            new_table = self.table[:, attr + [self.table.shape[1] - 1]]
        else:
            new_table = self.table

        return [tuple(attributes[:-1]) + (attributes[-1],) + (i,) for i, attributes in enumerate(new_table)]

    def get_pos(self, chunk_number, attr=None):
        """

        :param chunk_number:
        :param attr:
        :return:
        """

        system = self.prep_tab(attr)
        eav_rdd = Configuration.sc.parallelize(system, chunk_number)
        agr = eav_rdd.map(lambda x: (x[:-2], [x[-2], [x[-1]]])).reduceByKey(self.r2)
        q = []
        print agr.collect()
        for el in agr.collect():
            add = True
            for e in el[1][1]:
                if e == -1:
                    add = False
            if add is True:
                q += el[1][1]
        return q

    def r2(self, x, y):
        if x[0] == y[0]:
            return [x[0], x[1] + y[1]]
        return [x[0], [-1]]


    def get_possitive_area(self, chunk_number, attr=None):
        """
        TODO
        :param chunk_number:
        :param attr:
        :return:
        """
        abstraction_class = self.get_abstraction_class(chunk_number, attr=attr)
        print abstraction_class
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
    a = is_consistent.get_possitive_area(4,[1,3])
    print a
    print ''
    b =is_consistent.get_pos(2, [1,3])
    print b
    print set(a)==set(b)



    # print is_consistent.get_abstraction_class(4, attr=range(0, 1))




# TODO Testy
# TODO Dokumentacja
# TODO Refactoring code
# TODO cleaning code
