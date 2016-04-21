"""
Test for HyperplaneExtractor class
"""
import numpy as np
from unittest import TestCase

from reduct_feature_selection.feature_extractions.doubtful_points_strategies.min_dist_doubtful_points_strategy import \
    MinDistDoubtfulPointsStrategy

__author__ = 'krzysztof'


class TestMinDistDoubtfulPointsStrategy(TestCase):

    def test_strategy(self):
        table = np.array([(-1, -1),
                          (-1, 1),
                          (1, 1),
                          (1, -1),
                          (2, -1)],
                          dtype=[('x', float), ('y', float)])
        dec = [0, 0, 0, 1, 1]
        objects1 = range(5)
        objects2 = [2, 3, 4]
        objects3 = [0, 1, 2]
        min_dist_str1 = MinDistDoubtfulPointsStrategy(table, dec, 10)
        min_dist_str2 = MinDistDoubtfulPointsStrategy(table, dec, 2)
        min_dist_str3 = MinDistDoubtfulPointsStrategy(table, dec, 1)

        self.assertEqual(min_dist_str1.decision(objects1), 0)
        self.assertEqual(min_dist_str1.decision(objects2), 1)
        self.assertEqual(min_dist_str2.decision(objects1), None)
        self.assertEqual(min_dist_str3.decision(objects3), 0)
