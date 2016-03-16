"""Test for distinguish table class"""
import numpy as np
from unittest import TestCase

from reduct_feature_selection.abstraction_class import AbstractionClass


class AbstractionClassTest(TestCase):
    def test_son_table(self):
        table = np.array(
            [[1, 1, 0, 0],
             [1, 1, 2, 1],
             [1, 1, 3, 1],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0, 1, 2, 1],
             [0, 1, 1, 1],
             [0, 0, 2, 0]])
        real_result = AbstractionClass(table)
        real_result = real_result.check_consistent(3, [0, 1])
        expected_result = [[0, 1, 2], [3, 5, 6], [4, 7]]
        self.assertEqual(expected_result, real_result)

    def test_another_son_table(self):
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
        real_result = AbstractionClass(table)
        real_result = real_result.check_consistent(3, [0, 1])
        expected_result = [[7, 10], [6], [8], [3, 9], [0, 1], [4, 5], [2], [11]]
        self.assertEqual(expected_result, real_result)

    def test_one_class(self):
        table = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        real_result = AbstractionClass(table)
        real_result = real_result.check_consistent(2, [0, 1])
        self.assertEqual(real_result, [[0, 1, 2]])
