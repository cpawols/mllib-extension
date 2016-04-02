"""
TODO
"""

import numpy as np
from collections import Counter

import itertools
from operator import add


class GenerateRules:
    @staticmethod
    def generate_all_rules(br_decision_table):
        """TODO"""
        objects_number = br_decision_table.shape[0]
        rules = []
        for i in range(objects_number):
            rules.append(GenerateRules.engine(i, br_decision_table))
        return reduce(add,rules)

    @staticmethod
    def engine(row_number, br_table=None):
        dis = GenerateRules.generate_distinguish_row(row_number, br_table)
        implicants = GenerateRules.build_implicant(dis, br_table)
        return GenerateRules.build_rules(implicants, row_number, br_table)

    @staticmethod
    def generate_distinguish_row(object_number, br_decision_table):
        """TODO"""
        distinguish_row = []
        given_object = br_decision_table[object_number]
        object_decision = given_object[-1]
        for i, current_object in enumerate(br_decision_table):
            if len(current_object) != len(given_object):
                raise ValueError("Number of attributes in two objects are different")
            if object_number != i and current_object[-1] != object_decision:
                row_distinction = GenerateRules._get_different_rows_possitions(current_object, given_object)
                distinguish_row.append(row_distinction)
        return distinguish_row

    @staticmethod
    def _get_different_rows_possitions(current_object, given_object):
        row_distinction = []
        for attribute_number, (attribute_x, attribute_y) in enumerate(
                zip(current_object[:-1], given_object[:-1])):
            if attribute_x != attribute_y:
                row_distinction.append(attribute_number)
        return row_distinction

    @staticmethod
    def _check_length_of_distinguish_filed(distinguish_row):
        # TODO rename this method.
        one_elements_fields = filter(lambda x: len(x) == 1, distinguish_row)
        if len(one_elements_fields) == len(distinguish_row):
            return True
        return False

    @staticmethod
    def build_implicant(distiguish_row, object_number):
        attribute_frequency = GenerateRules._get_frequency_distinguish_row(distiguish_row).most_common()
        distiguish_row_copy = distiguish_row[:]
        implicants = []
        while GenerateRules._check_length_of_distinguish_filed(distiguish_row_copy) is not True:
            implicant = []
            to_coverage = len(distiguish_row_copy)
            for attribute_info in attribute_frequency:
                if to_coverage != 0:
                    for i, field in enumerate(distiguish_row_copy):
                        if attribute_info[0] in field:
                            if len(distiguish_row_copy[i]) != 1:
                                distiguish_row_copy[i].remove(attribute_info[0])
                            implicant.append(attribute_info[0])
                            to_coverage -= 1
            implicants.append(list(set(implicant)))
        implicants.append([e[0] for e in distiguish_row_copy])
        return implicants

    @staticmethod
    def _get_frequency_distinguish_row(distinguish_row):
        return Counter(attribute for element in distinguish_row for attribute in element)

    @staticmethod
    def build_rules(implicants, object_number, br_decision_table=None):
        """TODO"""
        rules = []
        for implicant in implicants:
            rules.append(GenerateRules.build_rule(implicant, object_number, br_decision_table))
        return rules

    @staticmethod
    def build_rule(implicant, object_number, br_decision_table=None):
        key = tuple(implicant)
        attributes_values = []
        for attribut_number in implicant:
            attributes_values.append(br_decision_table[object_number][attribut_number])
        attributes_values.append(br_decision_table[object_number][-1])
        return {key: attributes_values}

    @staticmethod
    def cut_rule(rule):
        """TODO"""
        attributes = set()
        #rule_size =
        print rule
        # for key in rule.keys():
        #     for attribute in key:
        #         attributes.add(attribute)

        # combinations = [ set(itertools.combinations(attributes, i)) for i in range(1,rule_size)]
        # print combinations



    @staticmethod
    def compute_coverage(rule, br_decision_table):
        """TODO"""
        coverage = 0
        for i, row in enumerate(br_decision_table):
            add = True
            for attributes, values in rule.items():
                for attribute, valuee in zip(attributes, values[:-1]):
                    if row[attribute] != valuee:
                        add = False
                        break
            if add is True:
                coverage += 1
        return coverage

    @staticmethod
    def compute_accuracy(rule, br_decision_table):
        """TODO"""
        pass



if __name__ == "__main__":
    table = np.array([[1, 0, 1, 1, 0, 1],
                      [0, 0, 1, 1, 1, 0],
                      [0, 1, 1, 0, 1, 0]
                      ])

    table = np.array([[1, 1, 0, 1, 1],
                      [0, 1, 0, 1, 0],
                      [1, 1, 1, 0, 1],
                      [1, 0, 0, 1, 0]])

    x = GenerateRules.generate_all_rules(table)
    print x

    GenerateRules.cut_rule(x[1])
