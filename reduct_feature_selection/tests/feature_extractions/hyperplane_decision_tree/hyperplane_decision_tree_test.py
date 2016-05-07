"""
Test for HyperplaneDecisionTree class
"""
import numpy as np
from unittest import TestCase

from reduct_feature_selection.feature_extractions.doubtful_points_strategies.min_dist_doubtful_points_strategy import \
    MinDistDoubtfulPointsStrategy
from reduct_feature_selection.feature_extractions.hyperplane_extractors import HyperplaneExtractor

from pyspark import SparkContext, SparkConf
__author__ = 'krzysztof'


class TestHyperplaneDecisionTree(TestCase):

    def test_count_decision_tree(self):
        conf = (SparkConf().setMaster("spark://localhost:7077").setAppName("extractor test"))
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
        dec1 = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        dec2 = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0]
        attrs_list = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']

        md1 = MinDistDoubtfulPointsStrategy(table, dec1, 3)
        extractor1 = HyperplaneExtractor(table, attrs_list, dec1, md1, 60)
        dec_tree1 = extractor1.count_decision_tree(range(12), sc)
        results1 = map(lambda x: dec_tree1.predict(list(x)), table)

        md2 = MinDistDoubtfulPointsStrategy(table, dec2, 3)
        extractor2 = HyperplaneExtractor(table, attrs_list, dec2, md2, 60)
        dec_tree2 = extractor2.count_decision_tree(range(12), sc)
        results2 = map(lambda x: dec_tree2.predict(list(x)), table)

        svm_dec_tree = extractor1.count_decision_tree(range(12), sc, svm=True)
        results_svm = map(lambda x: svm_dec_tree.predict(list(x), svm=True), table)

        self.assertEqual(dec1, results1)
        self.assertEqual(dec2, results2)
        self.assertEqual(dec1, results_svm)

