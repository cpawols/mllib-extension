# coding=utf-8
"""
TODO
"""

import itertools
import numpy as np
import operator
from collections import Counter
from random import randint
from sklearn.cross_validation import train_test_split
from sklearn.tree import tree

from reduct_feature_selection.abstraction_class.abstraction_class import AbstractionClass
from reduct_feature_selection.abstraction_class.generate_rules import GenerateRules
from reduct_feature_selection.abstraction_class.reduce_inconsistent_table import ReduceInconsistentTable
from settings import Configuration


class SetAbstractionClass:
    INFINITY = 10000000

    def __init__(self, table):
        """
        TODO
        :param table: numpy array
        :return:
        """
        self.table = table

    def _prepare_table(self):
        """
        This function convert numpy array to list of tuple.
        One row is represented by one tuple.
        :return: List of tuples.
        """
        return [tuple(row) + (i,) for i, row in enumerate(self.table)]

    def _choose_subsets(self):
        """
        This method gets this subsets for which is computed approximation.
        :return:
        """
        pass

    def _compute_belief(self, lower_approximation, decision_distribution, decision_subset):
        """
        This function compute believe function for given subset.
        Believe function it's a ratio LowerApproximation(X_i \cup ... \cup X_n)
        to the |(X_1 \cup ... \cup X_n)|.
        Where X_1 and X_2 are a decisions.
        :return: Value of ratio for given subset
        """
        card_lower = sum(len(e) for e in lower_approximation)
        card_decision = self.compute_card_decision(decision_distribution, decision_subset)

        if card_decision != 0:
            return 1.0 * card_lower / card_decision
        else:
            return 10000000

    @staticmethod
    def compute_card_decision(decision_distribution, decision_subset):
        """
        TODO
        :param decision_distribution:
        :param decision_subset:
        :return:
        """
        card_decision = sum(value for key, value in decision_distribution.items() if key in decision_subset)
        return card_decision

    def _compute_pl(self, upper_approximation, decision_distribution, decision_subset):
        """
        Compute ratio UpperApproximation(X_i \cup ... \cup X_n)
        to the |(X_1 \cup ... \cup X_n)|.
        :return:
        """
        card_upper = sum(len(e) for e in upper_approximation)
        card_decision = self.compute_card_decision(decision_distribution, decision_subset)
        if card_decision != 0:
            return 1.0 * card_upper / card_decision
        else:
            return 10000000

    def compute_approximation(self, decision_subset, broadcast_abstraction_class, broadcast_decision_system,
                              approximation_list=False):
        """
        Compute
        :param broadcast_decision_system:
        :param broadcast_abstraction_class:
        :param decision_subset: subset of decisions.
        :return:
        """

        lower_approximation = self._compute_lower_approximation(
            decision_subset, broadcast_abstraction_class, broadcast_decision_system)

        upper_approximation = self._compute_upper_approximation(
            decision_subset, broadcast_abstraction_class, broadcast_decision_system)

        decision_distribution = self._get_decision_distribution()

        belief = self._compute_belief(lower_approximation, decision_distribution, decision_subset)
        pl = self._compute_pl(upper_approximation, decision_distribution, decision_subset)

        if approximation_list:
            return [belief, pl, decision_subset, lower_approximation, upper_approximation]
        else:
            return [abs(belief - pl), decision_subset]

    @staticmethod
    def _compute_upper_approximation(decision_subset, broadcast_abstraction_class, broadcast_decision_system):
        """
        Only one have to be.
        :param decision_subset:
        :return:
        """
        upper_approximation = []

        for abstraction_class in broadcast_abstraction_class:
            for object in abstraction_class:
                if (broadcast_decision_system[object][-1] in decision_subset and
                            abstraction_class not in upper_approximation):
                    upper_approximation.append(abstraction_class)
                    continue
        return upper_approximation

    @staticmethod
    def _compute_lower_approximation(decision_subset, broadcast_abstraction_class, broadcast_decision_system):
        """
        TODO
        :param decision_subset:
        :param broadcast_abstraction_class:
        :param broadcast_decision_system:
        :return:
        """
        lower_approximation = []
        for abstraction_class in broadcast_abstraction_class:  # usuwam .value
            add_class = True
            for object in abstraction_class:
                if broadcast_decision_system[object][-1] not in decision_subset:
                    add_class = False
            if add_class:
                lower_approximation.append(abstraction_class)
        return lower_approximation

    def _create_decision_subsets(self, range_of_subsets):
        """
        TODO at the moment choose all subsets to given range.
        :param range_of_subsets:
        :return:
        """
        decision_distribution = self._get_decision_distribution()
        subsets = []
        for k in range(0, range_of_subsets):
            subsets.append(
                list(itertools.combinations(range(min(decision_distribution), max(decision_distribution) + 1), k + 1)))
        return [list(e) for e in reduce(operator.add, subsets)]

    def _get_decision_distribution(self):
        """
        TODO
        :return:
        """
        return Counter(decision[-1] for decision in self.table)

    def get_abstraction_class(self):
        """
        TODO
        :return:
        """
        abstraction_class = AbstractionClass(self.table)
        abstraction_class = abstraction_class.get_abstraction_class()
        return abstraction_class

    def generate_subtable_indexes(self, subtables_number, min_range, max_range):
        """
        TODO
        :param subtables_number:
        :param min_range:
        :param max_range:
        :return:
        """
        subtables_indexes = []
        for _ in range(subtables_number):
            cardinality_of_subtable = randint(min_range, max_range)
            subtables_indexes.append([randint(0, self.table.shape[1] - 2) for _ in range(cardinality_of_subtable)])
        return subtables_indexes

    @staticmethod
    def get_abstraction_class_stand_alone(table):
        abstraction_class = AbstractionClass(table)
        return abstraction_class.standalone_abstraction_class()

    @staticmethod
    def generate_rules_for_approximation(res, table, take, subset_col_nums, cut_rules=False, treshold=0.9,
                                         weight=False):
        """
        TODO
        :param res:
        :param table:
        :param take:
        :param subset_col_nums:
        :param cut_rules:
        :param treshold:
        :param weight:
        :return:
        """

        counter = Counter()
        for r in res[:take]:
            table_tmp = SetAbstractionClass.rewrite_matrix(table, r[1])
            rules_for_approximation = GenerateRules.generate_all_rules(table_tmp, cut_rules=cut_rules,
                                                                       treshold=treshold)

            if weight is True:
                counter.update(
                    SetAbstractionClass.weight_attribute_importance(rules_for_approximation, table, subset_col_nums))
            else:
                counter.update(
                    subset_col_nums[e] for dictionary in rules_for_approximation for e in dictionary.keys()[0])

        return counter

    @staticmethod
    def weight_attribute_importance(rules_for_approximation, table, subset_col_nums):
        """
        TODO
        :param rules_for_approximation:
        :param table:
        :return:
        """
        counter = Counter()
        for rule in rules_for_approximation:
            coverage = GenerateRules.compute_coverage(rule, table)
            counter.update(subset_col_nums[attribute] for attribute in rule.keys()[0] for _ in range(coverage))
        return counter

    @staticmethod
    def set_decision(list_of_rows, row, decision):
        """

        :param list_of_rows:
        :param row:
        :param decision:
        :return:
        """
        row_without_decision = list(row[:-1])
        row_without_decision.append(decision)
        list_of_rows.append(row_without_decision)
        return list_of_rows

    @staticmethod
    def rewrite_matrix(table, decision_set):
        """

        :param table:
        :param decision_set:
        :return:
        """
        list_of_rows = []
        for row in table:
            all_in_decision_set = True
            at_least = False
            for decision in row[-1]:
                if decision not in decision_set:
                    all_in_decision_set = False
                if decision in decision_set:
                    at_least = True

            if all_in_decision_set is True:
                list_of_rows = SetAbstractionClass.set_decision(list_of_rows, row, 1)
            elif at_least is True and all_in_decision_set is False:
                list_of_rows = SetAbstractionClass.set_decision(list_of_rows, row, -1)
            else:
                list_of_rows = SetAbstractionClass.set_decision(list_of_rows, row, 0)

        return np.array([row for row in list_of_rows])

    def run_pipeline(self, subset_col_nums, subset_cardinality=2, take=5, cut_rules=False, treshold=0.9,
                     weight=False):
        """
        TODO - zmienic nazwe
        :param subset_col_nums:
        :param subset_cardinality:
        :param take:
        :return:
        """

        z = sorted(list(subset_col_nums))
        z.append(self.table.shape[1] - 1)
        t = self.table[:, z]
        abstraction_class = SetAbstractionClass.get_abstraction_class_stand_alone(t)
        decision_subset = self._create_decision_subsets(subset_cardinality)
        res = []
        for x in decision_subset:
            res.append(self.compute_approximation(x, abstraction_class, t))
        table = ReduceInconsistentTable(t)
        table = table.reduce_table()

        res = sorted(res)

        return SetAbstractionClass.generate_rules_for_approximation(res, table, take, subset_col_nums, cut_rules,
                                                                    treshold,
                                                                    weight=weight)

    def select_attributes(self, subtable_num, min_s, max_s, cut_rules=False, treshold=0.9,
                          weight=False):
        """
        Zrownolegalamy ze wzgledu na podtabele.
        :param subtable_num: liczba podtabeli jaki bedziemy losowac
        :param min_s: minimalna wielkosc podtabelo
        :param max_s: maksymalna wielkosc podtabeli
        :return:
        """
        subtable_attributes_numbers = [
            sorted(list(np.random.choice(self.table.shape[1] - 1, randint(min_s, max_s), replace=False)))
            for _ in range(subtable_num)]
        subtable_attr_num_rdd = Configuration.sc.parallelize(subtable_attributes_numbers)
        result = subtable_attr_num_rdd.map(
            lambda x: (1, self.run_pipeline(x, cut_rules=cut_rules, treshold=treshold,
                                            weight=weight))).reduceByKey(
            lambda x, y: x + y)
        # attribute_rank = result.collect()[0][1].most_common()
        return result.collect()

    @staticmethod
    def cut_attributes(attributes_rank):
        """
        TODO
        :param attributes_rank:
        :return:
        """
        value_of_model = float("inf")
        number_of_all_attributes = len(attributes_rank)
        sum_of_all_scores = sum(e[1] for e in attributes_rank)
        number_of_significant_attributes = 0

        for i in range(1, number_of_all_attributes + 1):
            ith_score = sum(e[1] for j, e in enumerate(attributes_rank) if j < i)
            fir = (1 - (1.0 * ith_score) / sum_of_all_scores) * (1 - (1.0 * ith_score) / sum_of_all_scores)
            sec = ((1.0 * i) / number_of_all_attributes) * ((1.0 * i) / number_of_all_attributes)
            fir += sec

            if fir < value_of_model:
                value_of_model = fir
                number_of_significant_attributes = i

        return sorted([e[0] for i, e in enumerate(attributes_rank) if i < number_of_significant_attributes])


if __name__ == "__main__":
    table = np.array(
        [
            [1, 1, 0, 0, 1],
            [1, 1, 0, 0, 3],
            [0, 0, 1, 1, 2],
            [0, 1, 1, 0, 1],
            [1, 0, 0, 1, 2],
            [1, 0, 0, 1, 2],
            [1, 1, 1, 1, 3],
            [1, 1, 1, 0, 2],
            [1, 1, 1, 0, 2],
            [1, 1, 1, 0, 1]

        ]
    )
    # table = np.array([
    #     [1, 0, 0, 0, 1],
    #     [0, 1, 0, 0, 0],
    #     [1, 1, 1, 1, 1],
    #     [0, 1, 1, 1, 0]
    # ])
    import scipy.io as sio

    x = sio.loadmat('/home/pawols/Develop/Mgr/mgr/BASEHOCK.mat')

    X_train, X_test, y_train, y_test = train_test_split(
        x['X'], x['Y'], test_size=0.2, random_state=42)
    # table = np.append(x['X'], x['Y'], axis=1)
    table = np.append(X_train, y_train, axis=1)
    table_v = np.append(X_test, y_test, axis=1)

    # table = np.array([[randint(0, 5) for _ in range(7000)] for _ in range(500)])
    # X_train, X_test = table[:400, :-1], table[400:, :-1]
    # y_train, y_test = table[:400, -1], table[400:, -1]

    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    print clf.score(X_test, y_test)

    a = SetAbstractionClass(table)
    res = a.select_attributes(4, 4, 30, cut_rules=True, treshold=0.8, weight=True)
    print res[0][1].most_common()
    for i in range(1, 150, 3):
        sel = [e[0] for j, e in enumerate(res[0][1].most_common()) if j < i]

        clf = tree.DecisionTreeClassifier()
        clf.fit(X_train[:, sorted(sel)], y_train)
        print len(sel), clf.score(X_test[:, sorted(sel)], y_test)

    selected = SetAbstractionClass.cut_attributes(res[0][1].most_common())

    clf = tree.DecisionTreeClassifier()

    clf.fit(X_train[:, sorted(selected)], y_train)
    print clf.score(X_test[:, sorted(selected)], y_test)
    print len(selected)
