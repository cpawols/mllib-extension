import numpy as np
from collections import Counter
from unittest import TestCase

from reduct_feature_selection.abstraction_class.generate_rules import GenerateRules


class TestDistinguishTable(TestCase):
    def test_distinguish_row(self):
        table = np.array([[1, 0, 1, 1, 0, 1],
                          [0, 0, 1, 1, 1, 0],
                          [0, 1, 1, 0, 1, 0]
                          ])
        rule_generator_instance = GenerateRules()
        generated_rules_1 = rule_generator_instance.generate_distinguish_row(0, table)
        generated_rules_2 = rule_generator_instance.generate_distinguish_row(1, table)
        generated_rules_3 = rule_generator_instance.generate_distinguish_row(2, table)
        self.assertEqual(generated_rules_1, [[0, 4], [0, 1, 3, 4]])
        self.assertEqual(generated_rules_2, [[0, 4]])
        self.assertEqual(generated_rules_3, [[0, 1, 3, 4]])

    def test_attributes_frequency(self):
        table = np.array([[1, 0, 1, 1, 0, 1],
                          [0, 0, 1, 1, 1, 0],
                          [0, 1, 1, 0, 1, 0]
                          ])
        rule_generator_instance = GenerateRules()
        generated_distinct_row = rule_generator_instance.generate_distinguish_row(0, table)
        attribute_frequency = rule_generator_instance._get_frequency_distinguish_row(generated_distinct_row)
        self.assertEqual(attribute_frequency, Counter({0: 2, 1: 1, 3: 1, 4: 2}))

    def test_generate_implicants(self):
        table = np.array([[1, 1, 0, 1, 1],
                          [0, 1, 0, 1, 0],
                          [1, 1, 1, 0, 1],
                          [1, 0, 0, 1, 0]])
        a = GenerateRules()
        dis_3 = a.generate_distinguish_row(3, table)
        dis_2 = a.generate_distinguish_row(2, table)
        dis_1 = a.generate_distinguish_row(1, table)
        dis_0 = a.generate_distinguish_row(0, table)
        implicants = a.build_implicant(dis_3)

        self.assertEqual(implicants, [[1], [1, 2], [1, 3]])
        self.assertEqual(a.build_implicant(dis_2), [[2], [3], [0, 1]])
        self.assertEqual(a.build_implicant(dis_1), [[0], [0, 2], [0, 3]])
        self.assertEqual(a.build_implicant(dis_0), [[0, 1]])

    def test_generate_rules(self):
        table = np.array([[1, 1, 0],
                          [0, 1, 1],
                          [0, 0, 1],
                          [1, 0, 0]
                          ])
        rules_generator = GenerateRules()
        expected_rules = [{(0,): [1, 0]}, {(0, 1): [1, 1, 0]}, {(0,): [0, 1]}, {(0, 1): [0, 1, 1]}, {(0,): [0, 1]},
                          {(0, 1): [0, 0, 1]}, {(0,): [1, 0]}, {(0, 1): [1, 0, 0]}]
        self.assertEqual(expected_rules, rules_generator.generate_all_rules(table))
        len_of_cuted_rules = len(rules_generator.generate_all_rules(table, cut_rules=True, treshold=0.0))
        self.assertEqual(16, len_of_cuted_rules)
