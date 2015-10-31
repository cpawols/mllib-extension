from unittest import TestCase
import numpy as np
from reduct_feature_selection.commons.array_eav_converter import EavFormatter

__author__ = 'krzysztof'


class TestEavFormatter(TestCase):
    def test_to_array_format(self):
        array_example = np.array([(0, 1), (4, 5)], dtype=[('x', int), ('y', float)])
        expected_ouput = [(0, 'x', 0), (0, 'y', 1), (1, 'x', 4), (1, 'y', 5)]
        real_output = EavFormatter.from_array(array_example)
        self.assertEqual(expected_ouput, real_output, "wrong eav conversion")

    def test_to_eav_format(self):
        eav_example = [(0, 'x', 0), (0, 'y', 1), (1, 'x', 4), (1, 'y', 5)]
        expected_ouput = np.array([(0, 1), (4, 5)], dtype=[('x', float), ('y', float)])
        real_output = EavFormatter.to_array(eav_example)
        self.assertTrue(np.array_equal(expected_ouput, real_output), "wrong array conversion")
