import numpy as np

from reduct_feature_selection.abstraction_class.abstraction_class import AbstractionClass
from reduct_feature_selection.abstraction_class.consistent import Consistent


class ReduceInconsistentTable:
    """
    TODO
    """

    def __init__(self, table):
        self.table = table

    def reduce_table(self):
        """

        :return:
        """
        #consistent = Consistent(self.table)
        #if consistent.check_consistent():
        #    return self.table
        #else:
        return self._reduce()

    def _reduce(self):
        """

        :return:
        """

        abstraction_class = AbstractionClass(self.table)
        abstraction_class = abstraction_class.get_abstraction_class()
        reduced_table = []
        for abstraction in abstraction_class:
            decision = []
            for objects in abstraction:
                decision.append(self.table[objects][-1])
            reduced_table.append(tuple(self.table[objects][:-1]) + (tuple(tuple(set(decision))),))
        return reduced_table


if __name__ == "__main__":
    table_to_reduce = np.array([[1, 1, 1, 1], [1, 1, 1, 2], [1, 1, 0, 1], [1, 1, 0, 1], [2, 2, 1, 1]])
    a = ReduceInconsistentTable(table_to_reduce)
    print a.reduce_table()
