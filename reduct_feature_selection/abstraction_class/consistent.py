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
        print 'reduce', x, y
        if x == y:
            return x
        else:
            return self.CONTRADICTION

    def check_consistent(self, chunk_number=10):
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

    def check_cos(self):
        dictionary = {}
        for row in self.table:
            if tuple(row[:-1]) in dictionary:
                dictionary[tuple(row[:-1])].append(row[-1])
            else:
                dictionary[tuple(row[:-1])] = [row[-1]]
        for values in dictionary.values():
            if len(set(values)) > 1:
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
        [2,2,0,1,1]
    ])



    is_consistent = Consistent(table)
    print is_consistent.check_consistent()
    print is_consistent.check_cos()



# TODO Dokumentacja
# TODO Testy
# TODO tutorial
#TODO refactoring
