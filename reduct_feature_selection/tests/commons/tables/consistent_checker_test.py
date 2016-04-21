"""
Test for ConsistentChecker class
"""
import numpy as np
from unittest import TestCase

from reduct_feature_selection.commons.tables.consistent_checker import ConsistentChecker

__author__ = 'krzysztof'


class TestConsistentChecker(TestCase):
    def test_count_unconsistent_groups_random_table1(self):
        dec = [1,0,1,0,1,0]
        extracted_table = [[1,0,0,1,0,0], [0,1,0,0,0,0]]
        groups = ConsistentChecker.count_unconsistent_groups(extracted_table, dec)
        expected_groups = [[0,3], [2,4,5]]
        self.assertEqual(sorted(groups), sorted(expected_groups))
        self.assertFalse(ConsistentChecker.is_consistent(extracted_table, dec))

    def test_count_unconsistent_groups_random_table2(self):
        dec = [1,0,2,0,1,2,2]
        extracted_table = [[1,0,0,1,0,2,2], [0,1,0,0,0,2,2]]
        groups = ConsistentChecker.count_unconsistent_groups(extracted_table, dec)
        expected_groups = [[0,3], [2,4]]
        self.assertEqual(sorted(groups), sorted(expected_groups))
        self.assertFalse(ConsistentChecker.is_consistent(extracted_table, dec))

    def test_count_unconsistent_groups_empty_table(self):
        dec = [1,0,1,0,1,0]
        extracted_table = []
        groups = ConsistentChecker.count_unconsistent_groups(extracted_table, dec)
        expected_groups = [[0,1,2,3,4,5]]
        self.assertEqual(groups, expected_groups)
        self.assertTrue(ConsistentChecker.is_consistent(extracted_table, dec))