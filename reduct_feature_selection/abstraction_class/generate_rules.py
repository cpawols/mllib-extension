"""
TODO
"""

import itertools
import numpy as np
from collections import Counter
from operator import add
from random import randint
from sklearn.cross_validation import train_test_split
from sklearn.tree import tree

from reduct_feature_selection.abstraction_class.consistent import Consistent
from settings import Configuration


class GenerateRules:
    @staticmethod
    def generate_all_rules(br_decision_table, cut_rules=False, treshold=0.9, sc=None):
        """TODO"""


        if sc is None:
            objects_number = br_decision_table.shape[0]
            rules = []
            for i in range(objects_number):
                rules.append(GenerateRules.engine(i, br_decision_table, cut_rules, treshold))
            return reduce(add, rules)
        else:
            objects_number = br_decision_table.shape[0]

            objects_number_rdd = sc.parallelize(range(objects_number))

            rules = objects_number_rdd.map(lambda x: GenerateRules.engine(x, br_decision_table, cut_rules, treshold)).reduce(lambda x,y : x+y)
            return rules



    @staticmethod
    def engine(row_number, br_table=None, cut_rules=False, treshold=0.9):
        dis = GenerateRules.generate_distinguish_row(row_number, br_table)
        implicants = GenerateRules.build_implicant(dis)
        z = GenerateRules.build_rules(implicants, row_number, br_table, cut_rules, treshold)

        return z

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
    def build_implicant(distiguish_row):
        print 'start'
        attribute_frequency = GenerateRules._get_frequency_distinguish_row(distiguish_row).most_common()
        distiguish_row_copy = distiguish_row[:]
        implicants = []
        while GenerateRules._check_length_of_distinguish_filed(distiguish_row_copy) is not True:
            implicant = []
            to_coverage = len(distiguish_row_copy)
            coveraged = []
            for attribute_info in attribute_frequency:
                if to_coverage != 0:
                    for i, field in enumerate(distiguish_row_copy):
                        if attribute_info[0] in field and i not in coveraged:
                            if len(distiguish_row_copy[i]) != 1:
                                distiguish_row_copy[i].remove(attribute_info[0])
                            if attribute_info not in implicant:
                                implicant.append(attribute_info[0])
                            coveraged.append(i)
                            #if i not in coveraged:
                            to_coverage -= 1
            implicants.append(list(set(implicant)))
        implicants.append(list(set([e[0] for e in distiguish_row_copy])))
        print 'end'
        return implicants

    @staticmethod
    def _get_frequency_distinguish_row(distinguish_row):
        return Counter(attribute for element in distinguish_row for attribute in element)

    @staticmethod
    def build_rules(implicants, object_number, br_decision_table=None, cut_rules=False, treshold=0.9):
        """TODO"""
        rules = []
        for implicant in implicants:
            rules.append(GenerateRules.build_rule(implicant, object_number, br_decision_table))

        if cut_rules is True:
            cuted_rules = []
            for rule in rules:
                cuted_rules.append(GenerateRules.cut_rule(rule, treshold=treshold, br_decision_table=br_decision_table))
            for i in filter(lambda x: len(x) > 0, cuted_rules):
                rules.extend(i)
        return rules

    @staticmethod
    def build_rule(implicant, object_number, br_decision_table=None):
        key = tuple(implicant)
        attributes_values = []
        for attribut_number in implicant:
            attributes_values.append(br_decision_table[object_number][attribut_number])
        attributes_values.append(br_decision_table[object_number][-1])
        print 'koniec'
        return {key: attributes_values}

    @staticmethod
    def cut_rule(rule, treshold=0.9, br_decision_table=None, max_length_of_cut=7):
        """TODO"""
        attributes = set()
        accepted_rules = []
        rule_size = len(reduce(add, rule.values())) - 1

        if 1 < rule_size < max_length_of_cut:
            for key in rule.keys():
                for attribute in key:
                    attributes.add(attribute)

            combinations = reduce(add,
                                  list(list(itertools.combinations(list(attributes), i)) for i in range(1, rule_size)))
            for combination in combinations:
                new_attributes = tuple([e for e in rule.keys()[0] if e not in combination])

                to_remove = [i for i, e in enumerate(rule.keys()[0]) if e in combination]
                new_values = [e for i, e in enumerate(rule.values()[0]) if i not in to_remove]
                new_rule = {new_attributes: new_values}
                accuracy = GenerateRules.compute_accuracy(new_rule, br_decision_table)
                if accuracy > treshold:
                    if len(new_values) > 1:
                        accepted_rules.append(new_rule)
                        # print 'accepted with accuracy', accuracy, rule_size, len(new_values)-1
                        # else:
                        #     print 'rejected rule', new_rule, ' with accuracy', accuracy
        return accepted_rules

    @staticmethod
    def get_important_rules(rules, set_decision=None):
        """
        Gets only significant rules.
        :param rules:  rules
        :param set_decision: decision which are significant
        :return: significant rules
        """
        if set_decision is None:
            set_decision = [0, 1]

        out = []
        for rule in rules:
            if rule.values[-1] in set_decision:
                out.append(rule)
        return out


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
        coverage = 0
        correct_decision = 0
        for i, row in enumerate(br_decision_table):
            cover = True
            for attribute, value in zip(rule.keys()[0], rule.values()[0][:-1]):
                if row[attribute] != value:
                    cover = False
            if cover is True:
                coverage += 1
                if br_decision_table[i][-1] == rule.values()[0][-1]:
                    correct_decision += 1
        return correct_decision / (coverage * 1.0)


if __name__ == "__main__":
    pass
    # table = np.array([[1, 1, 1, 1, 0],
    #                   [1, 0, 0, 1, 1]
    #                   ])
    #
    # table = np.array([[ 1,  1 , 1,  1 , 0],
    #                  [ 1,  1 , 0 , 0, -1],
    #                  [ 0 , 1 , 1 , 0 , 1],
    #                  [ 1 , 0,  0 , 1 , 1],
    #                  [ 0 , 0 , 1 , 1 , 0],
    #                  [ 0 , 0 , 1 , 1 , 1],
    #                  [ 1  ,1,  1 , 0 , 1],
    #                 ])
    # # table = np.array([[1, 1, 0, 0, 1],
    # #                   [1, 1, 0, 0, 1],
    # #                   [0, 0, 1, 1, 1],
    # #                   [0, 1, 1, 0, 1],
    # #                   [1, 0, 0, 1, 1],
    # #                   [1, 0, 0, 1, 1],
    # #                   [1, 1, 1, 0, 0],
    # #                   [1, 1, 1, 0, 0],
    # #                   [1, 1, 1, 0, 0],
    # #                   [1, 1, 1, 0, 0]])
    # # table = np.array([[randint(0, 1) for _ in range(5)] for _ in range(10)])
    #
    # # np.savetxt("example_table.csv", table, delimiter=",")
    # # table = np.genfromtxt('example_table.csv', delimiter=',')
    # print GenerateRules.generate_all_rules(table, cut_rules=False, treshold=0.9)
    # import scipy.io as sio
    # x = sio.loadmat('/home/pawols/Develop/Mgr/mgr/BASEHOCK.mat')
    #
    # X_train, X_test, y_train, y_test = train_test_split(
    #      x['X'], x['Y'], test_size=0.3, random_state=42)
    #
    #
    # t#able = np.append(x['X'], x['Y'], axis=1)
    #
    # attributes_list = [sorted(
    #     list(np.random.choice(range(0, table.shape[1] - 1), randint(500, 500),
    #                           replace=False)))
    #                    for _ in range(15000)]
    #
    # object_list = [sorted(list(np.random.choice(range(0, table.shape[0]-1), randint(15,15)))) for _ in range(2000) ]
    #
    # res = []
    # t = 0
    # for i, j in zip(attributes_list, object_list):
    #     #print i, j
    #     i.append(table.shape[1]-1)
    #     is_consistent = Consistent(table[j, :][:, i])
    #     #print table[j, :][:, i]
    #     if is_consistent.check_cos() is True:
    #         xx = GenerateRules.generate_all_rules(table[j, :][:, i], cut_rules=True, treshold=0.9)
    #
    #         for element in xx:
    #             for atr in element.keys():
    #                 for qw in atr:
    #                     res.append(qw)
    #         #print res
    #     t += 1
    #
    #
    #
    # k_folds=10
    # res = Counter(res)
    # print res
    # for i in range(1, 1500, 3):
    #     sel = [e[0] for j, e in enumerate(res.most_common()) if j < i]
    #     avg_score = 0
    #     for c in range(k_folds):
    #         X_train, X_test, y_train, y_test = train_test_split(
    #             x['X'], x['Y'], test_size=0.33, random_state=42)
    #         clf = tree.DecisionTreeClassifier()
    #         clf.fit(X_train[:, sorted(sel)], y_train)
    #         #wq.append(clf.score(X_test[:, sorted(sel)], y_test))
    #         #var_all.append(wq[-1])
    #         avg_score += clf.score(X_test[:, sorted(sel)], y_test)
    #     #avg_score2 += 1.0 * avg_score / k_folds
    #     print i, 1.0 * avg_score / k_folds