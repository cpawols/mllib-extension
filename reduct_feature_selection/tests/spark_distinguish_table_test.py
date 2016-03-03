import numpy as np
from numpy.testing import assert_array_equal
from unittest import TestCase

from pyspark import SparkConf

from commons.spark.tables.distinguish_table import DistinguishTable


class TestDistinguishTable(TestCase):
    def test_convert_to_tuple_of_list(self):
        decision_system = np.array([[1, 1, 0, 1], [0, 1, 0, 0], [1, 1, 1, 1]])
        a = DistinguishTable(decision_system)
        a._convert_to_list_of_tuples(a.decision_system)
        self.assertEqual(a._convert_to_list_of_tuples(a.decision_system),
                         [(1, 1, 0, 1, 0), (0, 1, 0, 0, 1), (1, 1, 1, 1, 2)])

    def test_trnapose_matrix(self):
        decision_system = np.array([[1, 1], [0, 0]])
        a = DistinguishTable(decision_system)
        assert_array_equal(a._transopse_matrix(a.decision_system), np.array([[1, 0], [1, 0]]))

    def test_remove_decision_column(self):
        decision_system = np.array([[1, 1, 0, 1], [0, 1, 0, 0], [1, 1, 1, 1]])
        a = DistinguishTable(decision_system)
        a.decision_system = a._remove_decision_column(a.decision_system)
        self.assertEqual(a._convert_to_list_of_tuples(a.decision_system), [(1, 1, 0, 0), (0, 1, 0, 1), (1, 1, 1, 2)])

    def test_make_table(self):
        decision_system = np.array(
            [[0, 1, 1, 0, 1, 1], [1, 0, 1, 0, 1, 0], [0, 0, 0, 0, 0, 1], [1, 1, 1, 1, 1, 1], [1, 0, 1, 0, 1, 0]])
        a = DistinguishTable(decision_system)
        expected_result = {(0, 1): [0, 1], (1, 2): [0, 2, 4], (1, 3): [1, 3],
                           (0, 4): [0, 1], (3, 4): [1, 3], (2, 4): [0, 2, 4]}
        system, decision = a._prepare_data_make_distinguish_table()[0], a._prepare_data_make_distinguish_table()[1]
        self.assertEqual(expected_result, next(a.make_table(system, decision)))

    def test_all_object_the_same_decision(self):
        decision_system = np.array([[0, 0, 1, 1], [1, 1, 1, 1]])
        a = DistinguishTable(decision_system)
        system, decision = a._prepare_data_make_distinguish_table()[0], a._prepare_data_make_distinguish_table()[1]
        self.assertEqual({}, next(a.make_table(system, decision)))

    def test_distributed_distinguish_table_one_chunk_test(self):
        pass
