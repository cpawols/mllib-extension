import numpy as np
from unittest import TestCase

from reduct_feature_selection.abstraction_class.aproximation_class_set import SetAbstractionClass
from reduct_feature_selection.abstraction_class.reduce_inconsistent_table import ReduceInconsistentTable


class ReduceInconsistentTableTest(TestCase):
    def test_reduce_inconsistent_table(self):
        table = np.array([
            [1, 1, 0, 0, 1],
            [1, 1, 0, 0, 2],
            [0, 0, 1, 1, 2],
            [0, 1, 1, 0, 1],
            [1, 0, 0, 1, 2],
            [1, 1, 1, 0, 3],
            [1, 1, 1, 0, 2],
            [1, 1, 1, 0, 1],
            [1, 1, 1, 0, 1]
        ])

        reducer = ReduceInconsistentTable(table)
        reduced_table = reducer.reduce_table()
        expected_table = [(1, 1, 0, 0, (1, 2)), (0, 1, 1, 0, (1,)), (1, 0, 0, 1, (2,)), (0, 0, 1, 1, (2,)),
                          (1, 1, 1, 0, (1, 2, 3))]
        self.assertEqual(expected_table, reduced_table)

    def test_reduce_consistent_table(self):
        table = np.array([
            [1, 1, 0, 0, 1],
            [1, 1, 0, 0, 1],
        ])
        reducer = ReduceInconsistentTable(table)
        reduced_table = reducer.reduce_table()
        expected_table = [(1, 1, 0, 0, (1,))]
        self.assertEqual(expected_table, reduced_table)

    def test_rewrite_table(self):
        table_to_rewrite = [(1, 1, 0, 0, (1, 2)), (0, 1, 1, 0, (1,)), (1, 0, 0, 1, (2,)), (0, 0, 1, 1, (2,)),
                            (1, 1, 1, 0, (1, 2, 3))]

        rewriete_table = SetAbstractionClass.rewrite_matrix(table_to_rewrite, [1, 2])
        print rewriete_table
        expected_table = np.array([
            [1, 1, 0, 0, 1],
            [0, 1, 1, 0, 1],
            [1, 0, 0, 1, 1],
            [0, 0, 1, 1, 1],
            [1, 1, 1, 0, -1]
        ])
        np.testing.assert_array_equal(rewriete_table, expected_table)
