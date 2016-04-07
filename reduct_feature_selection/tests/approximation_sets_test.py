import numpy as np
from unittest import TestCase

from reduct_feature_selection.abstraction_class.aproximation_class_set import SetAbstractionClass


class ApproximationTest(TestCase):
    def test_upper_approximation(self):
        table = np.array([
            [1, 1, 0, 0, 1],
            [1, 1, 0, 0, 2],
            [0, 0, 1, 1, 2],
            [0, 1, 1, 0, 1],
            [1, 0, 0, 1, 2],
            [1, 1, 1, 0, 3],
            [1, 1, 1, 0, 2],
            [1, 1, 1, 0, 1],
            [1, 1, 1, 0, 1]
        ])
        approximation = SetAbstractionClass(table)
        abstraction_class = SetAbstractionClass.get_abstraction_class_stand_alone(table)
        expected_upper_approximation = [[0, 1], [3], [5, 6, 7, 8], [4], [2]]
        self.assertEqual(expected_upper_approximation,
                         approximation._compute_upper_approximation([1, 2], abstraction_class, table))

        expected_lower_approximation = [[0, 1], [3], [4], [2]]
        self.assertEqual(expected_lower_approximation,
                         approximation._compute_lower_approximation([1, 2], abstraction_class, table))

        decision_distribution = approximation._get_decision_distribution()
        self.assertEqual(9 / (8 * 1.0), approximation._compute_pl(expected_upper_approximation,
                                                                  decision_distribution, [1, 2]))
        self.assertEqual(5 / (8 * 1.0),
                         approximation._compute_belief(expected_lower_approximation, decision_distribution, [1, 2]))
        self.assertEqual([0.5, [1, 2]], approximation.compute_approximation([1, 2], abstraction_class, table))

        expected_approximation_one_three = [0.2, 1.4, [1, 3], [[3]], [[0, 1], [3], [5, 6, 7, 8]]]
        self.assertEqual(expected_approximation_one_three,
                         approximation.compute_approximation([1, 3], abstraction_class, table, approximation_list=True))
