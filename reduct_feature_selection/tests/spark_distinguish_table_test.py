import numpy as np
from numpy.testing import assert_array_equal
from unittest import TestCase

from reduct_feature_selection.commons.tables.distinguish_table import DistinguishTable


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
        self.assertEqual(expected_result, (a.make_table(system, decision)))

    def test_all_object_the_same_decision(self):
        decision_system = np.array([[0, 0, 1, 1], [1, 1, 1, 1]])
        a = DistinguishTable(decision_system)
        system, decision = a._prepare_data_make_distinguish_table()[0], a._prepare_data_make_distinguish_table()[1]
        self.assertEqual({}, (a.make_table(system, decision)))

    def test_distributed_distinguish_table_one_chunk_test(self):
        decision_system = np.array([[1, 0, 2, 1], [0, 1, 2, 0], [1, 1, 1, 1], [3, 3, 3, 1], [2, 1, 0, 0]])
        A = DistinguishTable(decision_system)
        x, dec = A._prepare_data_make_distinguish_table()
        result = A.make_table(x, decisions=dec)
        expected_result = {(0, 1): [0, 1], (1, 2): [0, 2], (1, 3): [0, 1, 2], (0, 4): [0, 1, 2], (3, 4): [0, 1, 2],
                           (2, 4): [0, 2]}
        self.assertEqual(result, expected_result)

    def test_compute_implicant_test(self):
        decision_system = np.array([[1, 0, 2, 1], [1, 1, 2, 0], [1, 1, 1, 1], [3, 3, 3, 1], [2, 1, 0, 0]])
        A = DistinguishTable(decision_system)
        dist_table = {(0, 1): [1], (1, 2): [2], (1, 3): [0, 1, 2], (0, 4): [0, 1, 2], (3, 4): [0, 1, 2], (2, 4): [0, 2]}
        expected_result = {0: [1], 1: [1, 2], 2: [2], 3: [2], 4: [2]}
        self.assertEqual(expected_result, (A._compute_implicants(dist_table, A.first_heuristic_method)))

    def test_stages_test(self):
        decision_system = np.array([[0, 1, 1, 1], [1, 1, 0, 1], [0, 1, 0, 0]])
        A = DistinguishTable(decision_system)
        system, decision = A._prepare_data_make_distinguish_table()

        real_distinguish_table = A.make_table(system, decision)
        expected_distigusih_table = {(0, 2): [2], (1, 2): [0]}
        self.assertEqual(real_distinguish_table, expected_distigusih_table)

        # TODO first heuristic method test

        real_implicants = A._compute_implicants(real_distinguish_table, A.first_heuristic_method)
        expected_implicants = {0: [2], 1: [0], 2: [0, 2]}
        self.assertEqual(real_implicants, expected_implicants)

        real_rules = A.generate_rules_from_implicants(real_implicants, A.decision_system)
        expected_rules = {(2, 0): [1, 1], (0, 1): [1, 1], (0, 2, 2): [0, 0, 0]}
        self.assertEqual(real_rules, expected_rules)

        accepted_rules = A.validate_rules(rules=real_rules, validation_function=A.validation_function_f1,
                                          original_decision_system=A.decision_system)
        self.assertEqual(real_rules, next(accepted_rules))

    def test_1(self):
        dec = np.array([[0,1,1,0,1],[1,1,0,1,0]])
        A = DistinguishTable(dec)
        system, decision = A._prepare_data_make_distinguish_table()

        real_distinguish_table = A.make_table(system, decision)
        print real_distinguish_table


if __name__ == "__main__":
    unittest.main()
