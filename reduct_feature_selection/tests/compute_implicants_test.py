"""Test for distinguish table class"""
import numpy as np
from unittest import TestCase

from commons.rules.implicants_rules import Implicants
from commons.tables.make_distinguish_table import DistinguishTable

class TestComputeImplicants(TestCase):

    def test_one_object(self):
        decision_system = np.array([[1, 1, 1, 1], [0, 0, 1, 0]])
        distinguish_table = DistinguishTable.compute_distinguish_matrix(decision_system)
        frequency_of_attribute = DistinguishTable.compute_frequency_of_attribute(distinguish_table)
        implicants = Implicants.compute_implicants(distinguish_table, frequency_of_attribute)

        rules = Implicants.generate_rules(decision_system, implicants)
        expected_rules = {(1.0, 0): [[1], 1], (1.0, 1): [[0], 0]}
        self.assertEqual(expected_rules, rules)


