"""
Test for EavConverter class
"""
from unittest import TestCase
import numpy as np
from reduct_feature_selection.commons.array_eav_converter import EavConverter

__author__ = 'krzysztof'


class TestEavConverter(TestCase):
    def test_to_eav_convert(self):
        array_example = np.array([(0, 1), (4, 5)], dtype=[('x', int), ('y', float)])
        expected_output = [(0, 'x', 0), (0, 'y', 1), (1, 'x', 4), (1, 'y', 5)]
        real_output = EavConverter.convert_to_eav(array_example)
        self.assertEqual(expected_output, real_output,
                         msg="wrong conversion from array to eav, tables are not equal")

    def test_to_array_convert(self):
        eav_example = [(0, 'x', 0), (0, 'y', 1), (1, 'x', 4), (1, 'y', 5)]
        expected_output = np.array([(0, 1), (4, 5)], dtype=[('x', float), ('y', float)])
        real_output = EavConverter.convert_to_array(eav_example)
        self.assertTrue(np.array_equal(expected_output, real_output),
                        msg="wrong conversion from eav to array, tables are not equal")

    def test_conversion_involution(self):
        eav_example = [(0, 'x', 0), (0, 'y', 1), (1, 'x', 4), (1, 'y', 5)]
        real_output = EavConverter.convert_to_eav(EavConverter.convert_to_array(eav_example))
        self.assertEqual(eav_example, real_output,
                         msg="error: conversions must be involution")

    def test_empty_conversion(self):
        eav_example = []
        expected_output = np.array([])
        real_output = EavConverter.convert_to_array(eav_example)
        self.assertTrue(np.array_equal(expected_output, real_output),
                        msg="wrong conversion from empty eav")

    def test_wrong_array_format_exception(self):
        with self.assertRaises(TypeError, msg="don't raise wrong format exception"):
            array_example = np.array([[0, 1], [4, 5]])
            EavConverter.convert_to_eav(array_example)
