"""
Test for HyperplaneExtractor class
"""
import numpy as np
from unittest import TestCase

from reduct_feature_selection.feature_extractions.doubtful_points_strategies.base_doubtful_points_strategy import BaseDoubtfulPointsStrategy

__author__ = 'krzysztof'


class TestBaseDoubtfulPointsStrategy(TestCase):

    def test_strategy(self):
        table = np.array([(-1, -1),
                          (-1, 1),
                          (1, 1),
                          (1, -1),
                          (2, -1)],
                          dtype=[('x', float), ('y', float)])
        dec = [0, 0, 0, 1, 1]
        objects1 = range(5)
        objects2 = [3, 4]
        base_str = BaseDoubtfulPointsStrategy(table, dec)
        self.assertEqual(base_str.decision(objects1), None)
        self.assertEqual(base_str.decision(objects2), 1)
