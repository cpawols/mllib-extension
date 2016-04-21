"""
Test for HyperplaneExtractor class
"""
import numpy as np
from unittest import TestCase

from reduct_feature_selection.feature_extractions.hyperplane_extractors import HyperplaneExtractor

__author__ = 'krzysztof'


class TestHyperplaneExtractor(TestCase):
    def test_get_unconsistent_reg_random_table1(self):
        dec = [1,0,1,0,1,0]
        extracted_table = [[1,0,0,1,0,0], [0,1,0,0,0,0]]
        he = HyperplaneExtractor(np.zeros(1), [], dec, 0.1)
        regs = he._get_unconsistent_reg(extracted_table)
        expected_regs = [[0,3], [2,4,5]]
        self.assertEqual(sorted(regs), sorted(expected_regs))

    def test_get_unconsistent_reg_random_table2(self):
        dec = [1,0,2,0,1,2,2]
        extracted_table = [[1,0,0,1,0,2,2], [0,1,0,0,0,2,2]]
        he = HyperplaneExtractor(np.zeros(1), [], dec, 0.1)
        regs = he._get_unconsistent_reg(extracted_table)
        expected_regs = [[0,3], [2,4]]
        self.assertEqual(sorted(regs), sorted(expected_regs))

    def test_get_unconsistent_reg_empty_table(self):
        dec = [1,0,1,0,1,0]
        extracted_table = []
        he = HyperplaneExtractor(np.zeros(1), [], dec, 0.1)
        regs = he._get_unconsistent_reg(extracted_table)
        expected_regs = [[0,1,2,3,4,5]]
        self.assertEqual(regs, expected_regs)