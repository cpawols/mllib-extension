"""Test for distinguish table class"""
import numpy as np
from unittest import TestCase

from commons.tables.make_distinguish_table import DistinguishTable


class TestCsvReaderTest(TestCase):
    def test_simple_distinguish_table(self):
        input_matrix = np.array([[1, 2, 3], [4, 5, 6]])
        res = DistinguishTable.compute_distinguish_matrix(input_matrix)
        frequency_of_attributes = (
            DistinguishTable.compute_frequency_of_attribute(res))
        expected_output = [[set(), {0, 1}], [{0, 1}, set()]]
        self.assertEqual(frequency_of_attributes, ([(0.0, 2), (1.0, 2)]))
        self.assertEqual(res, expected_output)

    def test_two_objects(self):
        input_data = np.array([[0, 1, 0, 1], [1, 1, 0, 0]])
        expected_output = [[set(), {0}], [{0}, set()]]
        actual_output = DistinguishTable.compute_distinguish_matrix(input_data)
        frequency_of_attribute = (
            DistinguishTable.compute_frequency_of_attribute(actual_output))
        self.assertEqual(actual_output,
                         expected_output)
        self.assertEqual(frequency_of_attribute, ([(0, 2)]))

    def test_four_objects(self):
        input_data = np.array([[1, 1, 0, 1, 1], [0, 1, 0, 1, 0],
                               [1, 1, 1, 0, 1], [1, 0, 0, 1, 0]])
        actual_output = DistinguishTable.compute_distinguish_matrix(input_data)
        frequency_of_attributes = (
            DistinguishTable.compute_frequency_of_attribute(actual_output))

        expected_output = [[set() for _ in range(4)] for _ in range(4)]
        expected_output[0][1] = {0}
        expected_output[1][0] = {0}
        expected_output[0][3] = {1}
        expected_output[3][0] = {1}
        expected_output[1][2] = {0, 2, 3}
        expected_output[2][1] = {0, 2, 3}
        expected_output[2][3] = {1, 2, 3}
        expected_output[3][2] = {1, 2, 3}

        self.assertEqual(frequency_of_attributes,
                         ([(0.0, 4), (1.0, 4), (2.0, 4), (3.0, 4)]))
        self.assertEqual(expected_output, actual_output)

    def test_empty_decision_table(self):
        input_data = np.array([])
        with self.assertRaises(ValueError):
            DistinguishTable.compute_distinguish_matrix(input_data)

    def test_all_object_with_the_same_decision(self):
        input_data = np.array([[1, 0, 1, 2, 1], [1, 1, 1, 1, 1],
                               [0, 0, 0, 0, 1], [9, 9, 9, 2, 1]])
        actual_output = DistinguishTable.compute_distinguish_matrix(input_data)
        frequency_of_attributes = (
            DistinguishTable.compute_frequency_of_attribute(actual_output))
        expected_output = [[set() for _ in range(4)] for _ in range(4)]

        self.assertEqual(frequency_of_attributes, [])
        self.assertEqual(actual_output, expected_output)
