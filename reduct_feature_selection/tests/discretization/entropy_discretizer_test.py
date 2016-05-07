"""Test for simple discretizer class"""
import numpy as np
from unittest import TestCase

from reduct_feature_selection.discretization.entropy_discretizer import EntropyDiscretizer


class EntropyDiscretizerTest(TestCase):

    def test_discretize(self):
        table = np.array([(1, 1),
                      (1, 2),
                      (1, 3),
                      (1, 4),
                      (1, 5),
                      (1, 6),
                      (1, 7),
                      (1, 8),
                      (1, 9),
                      (1, 10),
                      (1, 11),
                      (1, 12),
                      (1, 13),
                      (1, 14),
                      (1, 15),
                      (1, 16),
                      (1, 17),
                      (1, 18),
                      (1, 19),
                      (1, 20)])
        dec_1 = [1,1,1,1,1,0,0,0,0,0,2,2,2,2,2,3,3,3,3,3]
        dec_2 = [0,1,0,1,0,1,0,1,0,1,2,2,2,2,2,3,3,3,3,3]
        discretizer = EntropyDiscretizer(table, ['C1', 'C2'], dec_1)
        discretized_table1 = discretizer.discretize()
        discretizer.set_dec(dec_2)
        discretized_table2 = discretizer.discretize()
        expected_table1 = np.array([(0.0, 0.0),
                                   (0.0, 0.0),
                                   (0.0, 0.0),
                                   (0.0, 0.0),
                                   (0.0, 0.0),
                                   (1.0, 1.0),
                                   (1.0, 1.0),
                                   (1.0, 1.0),
                                   (1.0, 1.0),
                                   (1.0, 1.0),
                                   (2.0, 2.0),
                                   (2.0, 2.0),
                                   (2.0, 2.0),
                                   (2.0, 2.0),
                                   (2.0, 2.0),
                                   (3.0, 3.0),
                                   (3.0, 3.0),
                                   (3.0, 3.0),
                                   (3.0, 3.0),
                                   (3.0, 3.0)])
        expected_table2 = np.array([(0.0, 0.0),
                                    (1.0, 1.0),
                                    (2.0, 2.0),
                                    (3.0, 3.0),
                                    (4.0, 4.0),
                                    (5.0, 5.0),
                                    (5.0, 5.0),
                                    (5.0, 5.0),
                                    (5.0, 5.0),
                                    (5.0, 5.0),
                                    (6.0, 6.0),
                                    (6.0, 6.0),
                                    (6.0, 6.0),
                                    (6.0, 6.0),
                                    (6.0, 6.0),
                                    (7.0, 7.0),
                                    (7.0, 7.0),
                                    (7.0, 7.0),
                                    (7.0, 7.0),
                                    (7.0, 7.0)])
        eq_list1 = [list(expected_table1[i]) == list(discretized_table1[i]) for i in range(20)]
        eq_list2 = [list(expected_table2[i]) == list(discretized_table2[i]) for i in range(20)]
        self.assertTrue(all(eq_list1))
        self.assertTrue(all(eq_list2))
