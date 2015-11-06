from numpy.testing import assert_array_equal
import os
from unittest import TestCase
from reduct_feature_selection.commons.cvs_reader import CSVReader
import numpy as np


class TestCsvReaderTest(TestCase):
    def test_read_csv_file(self):
        matrix = CSVReader.read_csv(os.path.expanduser('~')
                                    + "/.mllib-extension/data/csv_test_file.csv")

        expected_output = np.array([[1, 2, 3, 4], [5, 5, 4, 5], [1, 1, 4, 5]])
        assert_array_equal(matrix, expected_output)

    def test_read_csv_float_value(self):
        matrix = CSVReader.read_csv(os.path.expanduser('~')
                                    + "/.mllib-extension/data/csv_test_float_numbers.csv")
        expected_output = np.array([[1.1, 3.1, 3.3], [1.2, 0.9, 3.111]])
        assert_array_equal(matrix, expected_output)

    def test_read_csv_simple_file_number_of_lines_with_heder_one(self):
        matrix = CSVReader.read_csv(os.path.expanduser('~') + "/.mllib-extension/data/csv_test_file.csv",
                                    number_of_header_lines=1)
        expected_output = np.array([[5, 5, 4, 5], [1, 1, 4, 5]])
        assert_array_equal(matrix, expected_output)

    def test_aaaa(self):
        with self.assertRaises(ValueError, mgs="File does not exist!"):
            CSVReader.read_csv(os.path.expanduser('~') + "/.mllib-extension/data/csv_test_file314.csv")
