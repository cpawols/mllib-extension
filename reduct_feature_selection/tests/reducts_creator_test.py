import numpy as np
from unittest import TestCase

from reduct_feature_selection.reducts.compute_reducts import ReductsCreator
from reduct_feature_selection.reducts.distinguish_matrix import DistniguishMatrixCreator


class ReductCreatorTest(TestCase):
    def test_reduct_creator(self):
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
            [2, 2, 0, 1, 1]
        ])

        ds = DistniguishMatrixCreator(table)
        ds = ds.create_distinguish_table()

        reduct_generator = ReductsCreator(table, 2, 1, 3, sorted(ds.keys()))
        list_of_upreducts = reduct_generator.compute_up_reduct(ds)
        reducts = reduct_generator.check_if_reduct(list_of_upreducts)
        self.assertEqual(sorted(reducts), sorted([[0, 1, 3], [0, 2, 3]]))
