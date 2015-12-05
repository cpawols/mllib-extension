from unittest import TestCase
import numpy as np

from reduct_feature_selection.rule_classifier.compute_ind_class import compute_ind_class


class TestRuleClassifier(TestCase):
    def test_compute_ind_class(self):
        array = np.array([[1, 2, 3, 4, 1], [6, 7, 3, 4, 0], [2, 3, 3, 4, 1], [1, 2, 2, 3, 0]])
        expected_result = [((3, 4, 1), 2), ((3, 4, 0), 1), ((2, 3, 0), 1)]
        result = compute_ind_class(array, [2, 3], 2)
        self.assertEqual(expected_result, result)
