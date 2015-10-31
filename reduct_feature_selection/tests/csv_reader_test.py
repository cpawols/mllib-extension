from numpy.testing import assert_array_equal
from unittest import TestCase
from reduct_feature_selection.commons.cvs_reader import CSVReader
import numpy as np


class TestCsvReaderTest(TestCase):
    def test_read_csv_file(self):
        matrix = CSVReader.read_csv("/home/pawols/Develop/Mgr/mllib-extension/data/csv_test_file.csv")
        expected_output = np.array([[1, 2, 3, 4], [5, 5, 4, 5], [1, 1, 4, 5]])
        assert_array_equal(matrix, expected_output)

    def test_read_csv_float_value(self):
        matrix = CSVReader.read_csv("/home/pawols/Develop/Mgr/mllib-extension/data/csv_test_float_numbers.csv")
        expected_output = np.array([[1.1, 3.1, 3.3], [1.2, 0.9, 3.111]])
        assert_array_equal(matrix, expected_output)
