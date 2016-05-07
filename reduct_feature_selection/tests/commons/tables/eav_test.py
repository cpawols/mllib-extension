"""
Test for EavConverter class
"""
import numpy as np
from unittest import TestCase

from reduct_feature_selection.commons.tables.eav import Eav
from pyspark import SparkContext, SparkConf

__author__ = 'krzysztof'


class TestEav(TestCase):
    def test_convert_to_proper_format(self):
        array_example = np.array([[1, 2, 3], [3, 4, 5]])
        expected_output = np.array([(1, 2, 3), (3, 4, 5)], dtype=[('C1', int), ('C2', int), ('C3', int)])
        output = Eav.convert_to_proper_format(array_example)
        self.assertEqual(list(expected_output), list(output))
        self.assertEqual(expected_output.dtype, output.dtype)

    def test_to_eav_convert(self):
        array_example = np.array([(0, 1), (4, 5)], dtype=[('x', int), ('y', float)])
        expected_output = [(0, 'x', 0), (0, 'y', 1), (1, 'x', 4), (1, 'y', 5)]
        real_output = Eav.from_array(array_example).eav
        self.assertEqual(expected_output, real_output,
                         msg="wrong conversion from array to eav, tables are not equal")

    def test_to_array_convert(self):
        eav_example = [(0, 'x', 0), (0, 'y', 1), (1, 'x', 4), (1, 'y', 5)]
        expected_output = np.array([(0, 1), (4, 5)], dtype=[('x', float), ('y', float)])
        real_output = Eav(eav_example).convert_to_array()
        self.assertTrue(np.array_equal(expected_output, real_output),
                        msg="wrong conversion from eav to array, tables are not equal")

    def test_conversion_involution(self):
        eav_example = [(0, 'x', 0), (0, 'y', 1), (1, 'x', 4), (1, 'y', 5)]
        real_output = Eav.from_array(Eav(eav_example).convert_to_array()).eav
        self.assertEqual(eav_example, real_output,
                         msg="error: conversions must be involution")

    def test_empty_conversion(self):
        eav_example = []
        expected_output = np.array([])
        real_output = Eav(eav_example).convert_to_array()
        self.assertTrue(np.array_equal(expected_output, real_output),
                        msg="wrong conversion from empty eav")

    def test_wrong_array_format_exception(self):
        with self.assertRaises(TypeError, msg="don't raise wrong format exception"):
            array_example = np.array([[0, 1], [4, 5]])
            Eav.from_array(array_example)

    def test_update_index(self):
        array_example = np.array([(0, 1), (4, 5)], dtype=[('x', int), ('y', float)])
        expected_obj_index = {0: [0, 1], 1: [2, 3]}
        expected_attr_index = {'x': [0, 2], 'y': [1, 3]}
        eav = Eav.from_array(array_example)
        self.assertEqual(expected_obj_index, eav.update_index(0))
        self.assertEqual(expected_attr_index, eav.update_index(1))

    def test_sort(self):
        eav_example = [(0, 'x', 5), (0, 'y', 8), (1, 'x', 4), (1, 'y', 5), (1, 'z', 2), (0, 'z', 3)]
        expected_output = [(1, 'x', 4), (0, 'x', 5), (1, 'y', 5), (0, 'y', 8), (1, 'z', 2), (0, 'z', 3)]
        sort_eav = Eav(eav_example)
        sort_eav.sort()
        self.assertEqual(expected_output, sort_eav.eav)

    def test_merge_sort(self):
        conf = (SparkConf().setMaster("spark://localhost:7077").setAppName("entropy"))
        sc = SparkContext(conf=conf)
        eav_example = [(0, 'x', 5), (0, 'y', 8), (1, 'x', 4), (1, 'y', 5), (1, 'z', 2), (0, 'z', 3)]
        expected_output = [(1, 'x', 4), (0, 'x', 5), (1, 'y', 5), (0, 'y', 8), (1, 'z', 2), (0, 'z', 3)]
        sort_eav = Eav(eav_example)
        sort_eav.merge_sort(sc)
        self.assertEqual(expected_output, sort_eav.eav)

    def test_consistent(self):
        eav_example_consistent = Eav([(0, 'x', 0), (0, 'y', 1), (1, 'x', 4), (1, 'y', 5)])
        eav_example_consistent.dec = {0: 1, 1: 0}
        eav_example_inconsistent = Eav([(0, 'x', 0), (0, 'y', 1), (1, 'x', 0), (1, 'y', 1)])
        eav_example_inconsistent.dec = {0: 1, 1: 0}
        self.assertTrue(not eav_example_inconsistent.is_consistent())
        self.assertTrue(eav_example_consistent.is_consistent())





