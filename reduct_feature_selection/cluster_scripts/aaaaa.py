# coding=utf-8
"""
TODO
"""

import itertools
import numpy as np
import operator
import pickle
from collections import Counter
from random import randint
from sklearn.cross_validation import train_test_split
from sklearn.tree import tree

import time
from pyspark import SparkConf, SparkContext

class ReduceInconsistentTable:
    """
    TODO
    """

    def __init__(self, table):
        self.table = table

    def reduce_table(self):
        """

        :return:
        """
        #consistent = Consistent(self.table)
        #if consistent.check_consistent():
        #    return self.table
        #else:
        return self._reduce()

    def _reduce(self):
        """

        :return:
        """

        abstraction_class = AbstractionClass(self.table)
        abstraction_class = abstraction_class.standalone_abstraction_class()
        reduced_table = []
        for abstraction in abstraction_class:
            decision = []
            for objects in abstraction:
                decision.append(self.table[objects][-1])
            reduced_table.append(tuple(self.table[objects][:-1]) + (tuple(tuple(set(decision))),))
        return reduced_table


class GenerateRules:
    @staticmethod
    def generate_all_rules(br_decision_table, cut_rules=False, treshold=0.9):
        """TODO"""
        objects_number = br_decision_table.shape[0]
        rules = []
        for i in range(objects_number):
            rules.append(GenerateRules.engine(i, br_decision_table, cut_rules, treshold))
        return reduce(operator.add, rules)

    @staticmethod
    def engine(row_number, br_table=None, cut_rules=False, treshold=0.9):
        dis = GenerateRules.generate_distinguish_row(row_number, br_table)
        implicants = GenerateRules.build_implicant(dis)
        return GenerateRules.build_rules(implicants, row_number, br_table, cut_rules, treshold)

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
        return {key: attributes_values}

    @staticmethod
    def cut_rule(rule, treshold=0.9, br_decision_table=None, max_length_of_cut=7):
        """TODO"""
        attributes = set()
        accepted_rules = []
        rule_size = len(reduce(operator.add, rule.values())) - 1

        if 1 < rule_size < max_length_of_cut:
            for key in rule.keys():
                for attribute in key:
                    attributes.add(attribute)

            combinations = reduce(operator.add,
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



class AbstractionClass:
    def __init__(self, table):
        self.table = table

    def _preprae_table(self, attr=None):
        """
        Ostatni jest numer obiektu
        :param table:
        :return:
        """
        if attr is not None:
            new_table = self.table[:, attr + [self.table.shape[1] - 1]]
        else:
            new_table = self.table

        return [tuple(attributes[:-1]) + (i,) for i, attributes in enumerate(new_table)]
    #
    # def get_abstraction_class(self, chunk_number=None, attr=None):
    #     """
    #     TODO
    #     :param chunk_number:
    #     :return:
    #     """
    #     system = self._preprae_table(attr)
    #     eav_rdd = sc.parallelize(system, chunk_number)
    #     agregatted = eav_rdd.map(lambda x: (x[:-1], [x[-1]])).reduceByKey(lambda x, y: x + y)
    #     abstraction_class = []
    #     for tuples in agregatted.collect():
    #         abstraction_class.append(tuples[-1])
    #     return abstraction_class


    def my_map(self, list_of_tuples):

        result = {}
        for tuple in list_of_tuples:
            if tuple[:-1] in result:
                result[tuple[:-1]].append(tuple[-1])
            else:
                result[tuple[:-1]] = [tuple[-1]]
        yield result

    def my_reduce(self, x, y):
        for key, value in x.items():
            if key in y:
                y[key] += value
            else:
                y[key] = value
        return y

    def standalone_abstraction_class(self):
        d = {}
        for i, row in enumerate(self.table):
            if tuple(row[:-1]) in d:
                d[tuple(row[:-1])].append(i)
            else:
                d[tuple(row[:-1])] = [i]
        return d.values()

    def prepare_table(self, attr):
        if attr is not None:
            new_table = self.table[:, attr + [self.table.shape[1] - 1]]
        else:
            new_table = self.table

        return [tuple(attributes[:-1]) + (attributes[-1],) + (i,) for i, attributes in enumerate(new_table)]

    # def get_pos(self, chunk_number, attr=None):
    #     """
    #
    #     :param chunk_number:
    #     :param attr:
    #     :return:
    #     """
    #
    #     system = self.prepare_table(attr)
    #     eav_rdd = Configuration.sc.parallelize(system, chunk_number)
    #     agr = eav_rdd.map(lambda x: (x[:-2], [x[-2], [x[-1]]])).reduceByKey(self.r2)
    #     q = []
    #     print agr.collect()
    #     for el in agr.collect():
    #         add = True
    #         for e in el[1][1]:
    #             if e == -1:
    #                 add = False
    #         if add is True:
    #             q += el[1][1]
    #     return q

    def r2(self, x, y):
        if x[0] == y[0]:
            return [x[0], x[1] + y[1]]
        return [x[0], [-1]]

    def get_positive_area(self, chunk_number, attr=None):
        """

        :param chunk_number:
        :param attr:
        :return:
        """
        abstraction_class = self.get_abstraction_class(chunk_number, attr=attr)
        pos = []
        for ab in abstraction_class:
            a = set()
            for ob in ab:
                a.add(self.table[ob][-1])
            if len(a) == 1:
                pos = pos + ab
        return pos


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

    def select_attributes(self,sc, subtable_num, min_s, max_s, cut_rules=False, treshold=0.9,
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
        subtable_attr_num_rdd = sc.parallelize(subtable_attributes_numbers)
        result = subtable_attr_num_rdd.map(
            lambda x: (1, self.run_pipeline(x, cut_rules=cut_rules, treshold=treshold,
                                            weight=weight))).reduceByKey(
            lambda x, y: x + y)

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

    table = np.append(X_train, y_train, axis=1)
    table_v = np.append(X_test, y_test, axis=1)


    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    print clf.score(X_test, y_test)

    a = SetAbstractionClass(table)

    conf = SparkConf().setAppName("dasds")
    sc = SparkContext(conf=conf)
    for i in range(1500, 2501, 500):
        suma = 0
        iteration = 0
        maximum  = 0
        print "Subtables number is equal = ", i
        start = time.time()
        res = a.select_attributes(sc,  i, 5, 15, cut_rules=True, treshold=0.8, weight=True)
        end = time.time()
        print 'Time for ', i, 'subtables',  end-start
        #print res[0][1]
        # s = '/home/pawols/Develop/Mgr/mllib-extension/results/sel_attr_rough/aproximation' + str(i) + '.pickle'
        # with open(str, 'wb') as handle:
        #     pickle.dump(res[0][1], handle)
        #end = time.time()

        # for i in range(1, 2500, 3):
        #     sel = [e[0] for j, e in enumerate(res[0][1].most_common()) if j < i]
        #
        #     clf = tree.DecisionTreeClassifier()
        #     clf.fit(X_train[:, sorted(sel)], y_train)
        #     q = clf.score(X_test[:, sorted(sel)], y_test)
        #     suma += q
        #     maximum = max(maximum, q)
        #     iteration += 1
        #     print len(sel), clf.score(X_test[:, sorted(sel)], y_test)
        # #
        # print 1.0*suma/iteration, maximum
        # selected = SetAbstractionClass.cut_attributes(res[0][1].most_common())
        # #
        # clf = tree.DecisionTreeClassifier()
        # #
        # clf.fit(X_train[:, sorted(selected)], y_train)
        # print clf.score(X_test[:, sorted(selected)], y_test)
    #print len(selected)
