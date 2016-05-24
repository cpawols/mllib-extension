"""
Test for ConsistentChecker class
"""
import numpy as np
from unittest import TestCase

from reduct_feature_selection.commons.tables.consistent_checker import ConsistentChecker

from pyspark import SparkContext, SparkConf

__author__ = 'krzysztof'


# TODO: tests with spark context
class TestConsistentChecker(TestCase):

    def __init__(self, test):
        super(TestConsistentChecker, self).__init__(test)
        conf = (SparkConf().setMaster("spark://localhost:7077").setAppName("test_consistent"))
        self.sc = SparkContext(conf=conf)

    def test_count_unconsistent_groups_random_table1(self):
        dec = [1,0,1,0,1,0]
        extracted_table = [[1,0,0,1,0,0], [0,1,0,0,0,0]]
        groups = ConsistentChecker.count_unconsistent_groups(extracted_table, dec, self.sc)
        groups_not_sc = ConsistentChecker.count_unconsistent_groups(extracted_table, dec)
        sorted_groups = sorted(map(lambda x: sorted(x), groups))
        sorted_groups_not_sc = sorted(map(lambda x: sorted(x), groups_not_sc))
        expected_groups = [[0,3], [2,4,5]]
        self.assertEqual(sorted_groups_not_sc, sorted_groups)
        self.assertEqual(expected_groups, sorted_groups)
        self.assertFalse(ConsistentChecker.is_consistent(extracted_table, dec, self.sc))

    # def test_count_unconsistent_groups_random_table2(self):
    #     dec = [1,0,2,0,1,2,2]
    #     extracted_table = [[1,0,0,1,0,2,2], [0,1,0,0,0,2,2]]
    #     groups = ConsistentChecker.count_unconsistent_groups(extracted_table, dec, sc)
    #     sorted_groups = sorted(map(lambda x: sorted(x), groups))
    #     expected_groups = [[0,3], [2,4]]
    #     self.assertEqual(sorted_groups, expected_groups)
    #     self.assertFalse(ConsistentChecker.is_consistent(extracted_table, dec, sc))
    #
    # def test_count_unconsistent_groups_empty_table(self):
    #     dec = [1,0,1,0,1,0]
    #     extracted_table = []
    #     groups = ConsistentChecker.count_unconsistent_groups(extracted_table, dec, self.sc)
    #     sorted_groups = sorted(map(lambda x: sorted(x), groups))
    #     expected_groups = [[0,1,2,3,4,5]]
    #     self.assertEqual(sorted_groups, expected_groups)
    #     self.assertTrue(ConsistentChecker.is_consistent(extracted_table, dec, self.sc))
