"""
TODO
"""
import numpy as np

from settings import Configuration
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

    def check_consistent(self, chunk_number, attr):
        """
        TODO
        :param chunk_number:
        :return:
        """
        system = self._preprae_table(attr)
        eav_rdd = Configuration.sc.parallelize(system, chunk_number)
        agregatted = eav_rdd.map(lambda x: (x[:-1], [x[-1]])).reduceByKey(lambda x, y: x + y)
        abstraction_class = []
        for tuples in agregatted.collect():
            abstraction_class.append(tuples[-1])
        return abstraction_class


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

    is_consistent = AbstractionClass(table)
    print is_consistent.check_consistent(3, attr=[0, 1])


# TODO Testy
# TODO Dokumentacja
# TODO Refactoring code
