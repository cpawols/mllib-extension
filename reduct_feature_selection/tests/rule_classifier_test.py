from unittest import TestCase
import numpy as np

from reduct_feature_selection.rule_classifier.compute_ind_class import compute_ind_class, compute_rules


class TestRuleClassifier(TestCase):
    def test_compute_ind_class(self):
        array = np.array([[1, 2, 3, 4, 1], [6, 7, 3, 4, 0], [2, 3, 3, 4, 1], [1, 2, 2, 3, 0],
                          [1, 2, 2, 3, 0], [1, 2, 2, 3, 0], [1, 2, 2, 3, 0], [6, 7, 3, 4, 0],
                          [1, 2, 2, 3, 1], [1, 2, 2, 3, 1]])
        expected_result = {"partitions": [((2, 3), 6), ((3, 4), 4)],
                           "rules": [((2, 3, 0), 4), ((2, 3, 1), 2), ((3, 4, 0), 2), ((3, 4, 1), 2)]}
        result = compute_ind_class(array, [2, 3], 2)
        self.assertEqual(expected_result, result)

    def test_compute_rules(self):
        array1 = np.array([[1, 2, 3, 4, 1], [6, 7, 3, 4, 0], [2, 3, 3, 4, 1], [1, 2, 2, 3, 0],
                           [1, 2, 2, 3, 0], [1, 2, 2, 3, 0], [1, 2, 2, 3, 0], [6, 7, 3, 4, 0],
                           [1, 2, 2, 3, 1], [1, 2, 2, 3, 1]])
        array2 = np.array([[1, 2, 3, 4, 1], [6, 7, 3, 4, 0], [2, 3, 3, 4, 0], [1, 2, 2, 3, 0],
                           [1, 2, 2, 3, 0], [1, 2, 2, 3, 0], [1, 2, 2, 3, 0], [6, 7, 3, 4, 0],
                           [1, 2, 2, 3, 1], [1, 2, 2, 3, 1]])
        expected_result1 = [(2, 3, 0)]
        expected_result2 = [(2, 3, 0), (3, 4, 0)]
        result1 = compute_rules(array1, [2, 3], 0.5, 2)
        result2 = compute_rules(array2, [2, 3], 0.5, 2)
        self.assertEqual(expected_result1, result1)
        self.assertEqual(expected_result2, result2)
