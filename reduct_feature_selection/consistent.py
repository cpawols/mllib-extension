"""
TODO
"""
import numpy as np

from settings import Configuration
import reduct_feature_selection


class Consistent:

    CONTRADICTION = 'contradiction'

    def __init__(self, table):
        self.table = table

    def _preprae_table(self, table):
        """
        TODO
        :param table:
        :return:
        """
        return [tuple(attributes) for attributes in table]

    def reducee(self, x, y):
        """
        TODO
        :param x:
        :param y:
        :return:
        """
        if x == y:
            return x
        else:
            return self.CONTRADICTION

    def check_consistent(self, chunk_number):
        """
        TODO
        :param chunk_number:
        :return:
        """
        system = self._preprae_table(self.table)
        eav_rdd = Configuration.sc.parallelize(system, chunk_number)
        agregatted = eav_rdd.map(lambda x: (x[:-1], x[-1])).reduceByKey(self.reducee)
        for tuple in agregatted.collect():
            if tuple[-1] == self.CONTRADICTION:
                return False
        return True


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
    ])

    is_consistent = Consistent(table)
    print is_consistent.check_consistent(10)



# TODO Dokumentacja
# TODO Testy
# TODO tutorial
#TODO refactoring
