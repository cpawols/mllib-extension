"""
Test for HyperplaneExtractor class
"""
import numpy as np
from unittest import TestCase

from reduct_feature_selection.feature_extractions.doubtful_points_strategies.min_dist_doubtful_points_strategy import \
    MinDistDoubtfulPointsStrategy
from reduct_feature_selection.feature_extractions.hyperplane_extractors import HyperplaneExtractor

from pyspark import SparkContext, SparkConf
__author__ = 'krzysztof'


class TestHyperplaneExtractor(TestCase):

    def test_extract(self):
        conf = (SparkConf().setMaster("spark://localhost:7077").setAppName("entropy"))
        sc = SparkContext(conf=conf)
        table = np.array([(0, 1, 7, 1, 1, 1),
                      (4, 5, 8, 2, 3, 4),
                      (1, 2, 3, 5, 6, 2),
                      (3, 8, 9, 2, 3, 5),
                      (0, 1, 7, 6, 7, 1),
                      (4, 5, 8, 5, 4, 1),
                      (1, 2, 3, 8, 9, 1),
                      (10, 2, 3, 8, 9, 1),
                      (400, 2, 3, 8, 9, 1),
                      (7, 2, 3, 8, 9, 1),
                      (900, 2, 3, 8, 9, 1),
                      (3, 8, 9, 5, 1, 2)])
        dec = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        attrs_list = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']
        md = MinDistDoubtfulPointsStrategy(table, dec, 3)
        extractor = HyperplaneExtractor(table, attrs_list, dec, md, 60)
        extracted_table = extractor.extract(sc)
        expected_table = np.array([[1, 1],
                             [0, 1],
                             [1, 1],
                             [0, 1],
                             [1, 1],
                             [0, 1],
                             [0, 0],
                             [0, 1],
                             [1, 1],
                             [0, 1],
                             [1, 1],
                             [0, 1]])
        eq_list = [list(expected_table[i]) == list(extracted_table[i]) for i in range(12)]
        self.assertTrue(all(eq_list))
