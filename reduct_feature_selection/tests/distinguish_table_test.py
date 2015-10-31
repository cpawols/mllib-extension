"""Test for distinguish table class"""
from unittest import TestCase
import numpy as np

from distinguish_table.make_distinguish_table import DistinguishTable


class TestCsvReaderTest(TestCase):
    def test_simple_distinguish_table(self):
        input_matrix = np.array([[1, 2, 3], [4, 5, 6]])
        res = DistinguishTable.compute_distinguish_matrix(input_matrix)
        expected_output = [[set(), {0, 1}], [{0, 1}, set()]]
        self.assertEqual(res, expected_output)

    def test_two_objects(self):
        input_data = np.array([[0, 1, 0, 1], [1, 1, 0, 0]])
        expected_output = [[set(), {0}], [{0}, set()]]
        actual_output = DistinguishTable.compute_distinguish_matrix(input_data)
        self.assertEqual(DistinguishTable.compute_distinguish_matrix(input_data),
                         expected_output)

    def test_four_objects(self):
        input_data = np.array([[1, 1, 0, 1, 1], [0, 1, 0, 1, 0],
                               [1, 1, 1, 0, 1], [1, 0, 0, 1, 0]])
        actual_output = DistinguishTable.compute_distinguish_matrix(input_data)
        expected_output = [[set() for _ in range(4)] for _ in range(4)]
        expected_output[0][1] = {0}
        expected_output[1][0] = {0}
        expected_output[0][3] = {1}
        expected_output[3][0] = {1}
        expected_output[1][2] = {0, 2, 3}
        expected_output[2][1] = {0, 2, 3}
        expected_output[2][3] = {1, 2, 3}
        expected_output[3][2] = {1, 2, 3}
        self.assertEqual(expected_output, actual_output)

    def test_empty_decision_table(self):
        input_data = np.array([])
        with self.assertRaises(ValueError):
            DistinguishTable.compute_distinguish_matrix(input_data)

    def test_all_object_with_the_same_decision(self):
        input_data = np.array([[1, 0, 1, 2, 1], [1, 1, 1, 1, 1],
                               [0, 0, 0, 0, 1], [9, 9, 9, 2, 1]])
        actual_output = DistinguishTable.compute_distinguish_matrix(input_data)
        expected_output = [[set() for _ in range(4)] for _ in range(4)]
        self.assertEqual(actual_output, expected_output)
