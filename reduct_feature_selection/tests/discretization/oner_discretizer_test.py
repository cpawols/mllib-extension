"""Test for oner discretizer class"""
import numpy as np
from unittest import TestCase

from reduct_feature_selection.discretization.OneR_discretizer import OneRDiscretizer


class SimpleDiscretizerTest(TestCase):
    def test_discretize(self):
        table = np.array([(1, 7),
                          (1, 8),
                          (1, 3),
                          (1, 9),
                          (1, 1),
                          (1, 2),
                          (1, 5),
                          (1, 10)])
        dec = [0, 1, 1, 1, 0, 0, 1, 1]
        attrs_list = ['C1', 'C2']
        discretizer = OneRDiscretizer(table, attrs_list, dec, 2)
        discretized_table = discretizer.discretize()
        expected_table = np.array([[0., 1.],
                                   [0., 1.],
                                   [0., 0.],
                                   [1., 2.],
                                   [1., 0.],
                                   [1., 0.],
                                   [2., 1.],
                                   [2., 2.]])
        a = list(expected_table)
        eq_list = [list(expected_table[i]) == list(discretized_table[i]) for i in range(8)]
        self.assertTrue(all(eq_list))
