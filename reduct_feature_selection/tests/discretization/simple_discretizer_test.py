"""Test for simple discretizer class"""
import numpy as np
from unittest import TestCase

from reduct_feature_selection.discretization.simple_discretizer import SimpleDiscretizer


class SimpleDiscretizerTest(TestCase):

    def test_discretize(self):
        table = np.array([(0, 1, 7),
                          (4, 5, 8),
                          (1, 2, 3),
                          (3, 8, 9),
                          (0, 1, 7),
                          (4, 5, 8),
                          (1, 2, 3),
                          (3, 8, 9)])

        dec = [0, 1, 0, 1, 0, 1, 0, 1]
        discretizer = SimpleDiscretizer(table, ['C1', 'C2', 'C3'], dec, 3)
        discretized_table = discretizer.discretize()
        expected_table = np.array([[0., 0., 0.],
                                   [2., 1., 1.],
                                   [0., 0., 0.],
                                   [1., 2., 2.],
                                   [0., 0., 1.],
                                   [2., 1., 1.],
                                   [1., 1., 0.],
                                   [1., 2., 2.]])
        eq_list = [list(expected_table[i]) == list(discretized_table[i]) for i in range(8)]
        self.assertTrue(all(eq_list))

