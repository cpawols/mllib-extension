"""
TODO
"""

import numpy as np
import operator
from random import randint
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

from settings import Configuration
import reduct_feature_selection


class SmallAttributeSubsets:
    def __init__(self, table):
        self.table = table

    def _get_decisions_vector(self):
        """Returns decision list"""
        return list(self.table[:, -1])

    def _generate_attribute_subtables(self, number_of_subtables, minimal_cardinality_of_subtables,
                                      maximal_cardinality_of_subtables):
        """Generates list of attribute subtables"""
        if minimal_cardinality_of_subtables <= 0 or maximal_cardinality_of_subtables > self.table.shape[1]:
            raise ValueError("Incorrect values of subtables")

        list_of_subtables = []
        for _ in range(number_of_subtables):
            cardinality_of_subtable = randint(minimal_cardinality_of_subtables, maximal_cardinality_of_subtables)
            selected_attributes = [randint(0, self.table.shape[1] - 2) for _ in range(cardinality_of_subtable)]
            list_of_subtables.append([self.table[:, sorted(selected_attributes)], selected_attributes])

        return self._convert_list_of_subtables_to_list_of_list_of_tuples(list_of_subtables)

    def _convert_list_of_subtables_to_list_of_list_of_tuples(self, list_of_subtables):
        """Converts list of numpy arrays to list of list of tuples """
        result = []
        for subtable in list_of_subtables:
            result.append(list(tuple(row) for row in subtable[0]))
            result[-1].append(tuple(sorted(subtable[1])))
        return result

    @staticmethod
    def _convert_to_numpy(subtable):
        """Convert to numpy table"""
        return np.array([row for row in subtable[:-1]])

    def _map(self, subtable, target):
        """Map function"""
        data = self._convert_to_numpy(subtable)
        selected_attributes = subtable[-1]
        tree = DecisionTreeClassifier()
        clf = tree.fit(data, target)
        return [list(clf.feature_importances_), selected_attributes]

    def _get_average_of_attributes(self, list_with_results):
        """Compute average of attributes"""
        average = {}
        for result in list_with_results:
            for value, attriute in zip(result[0], result[1]):
                if attriute in average:
                    average[attriute] += value
                else:
                    average[attriute] = value
        return average

    def _get_averae2(self, list_with_results):
        average = {}
        for tables in list_with_results:
            for table in tables:
                for value, attribute in zip(table[0], table[1]):
                    if attribute in average:
                        average[attribute] += value
                    else:
                        average[attribute] = value
        return average

    def _map2(self, subtable_number, minimal_cardinality, maximal_cardinality, target):
        list_of_subtables = self._generate_attribute_subtables(subtable_number, minimal_cardinality,
                                                               maximal_cardinality)
        r = []
        for subtable in list_of_subtables:
            data = self._convert_to_numpy(subtable)
            selected_attributes = subtable[-1]
            tree = DecisionTreeClassifier()
            clf = tree.fit(data, target)
            r.append([list(clf.feature_importances_), selected_attributes])
        return r

    def engine(self, subtable_number, minimal_cardinality, maximal_cardinality, partition_number=6):
        """Engine of solution"""
        decisions = self._get_decisions_vector()
        list_of_subtables = self._generate_attribute_subtables(subtable_number, minimal_cardinality,
                                                               maximal_cardinality)
        rdd = Configuration.sc.parallelize(range(subtable_number))
        res = rdd.map(lambda x: self._map2(subtable_number, minimal_cardinality, maximal_cardinality, decisions))
        # system_rdd = Configuration.sc.parallelize(list_of_subtables)
        # res = system_rdd.map(lambda x: self._map(x, decisions))
        # print res.collect()
        #        print self._get_averae2(res.collect())
        a = self._get_averae2(res.collect())
        sorted_x = reversed(sorted(a.items(), key=operator.itemgetter(1)))
        return list(sorted_x)


if __name__ == "__main__":
    # table = np.array([[1, 2, 3, 1, 1],
    #                   [2, 1, 3, 4, 1],
    #                   [2, 3, 2, 1, 7]])
    table = np.array([[randint(0, 1000) for _ in range(700)] for _ in range(1000)])
    target = np.array([[randint(1, 4)] for _ in range(1000)])



    X_train, X_test, y_train, y_test = train_test_split(
        table, target, test_size=0.2, random_state=42)

    table = np.append(table, target, axis=1)
    tree = DecisionTreeClassifier()
    clf = tree.fit(X_train, y_train)
    print clf.score(X_test, y_test)


    #table = np.append(iris.data, np.array([[e] for e in iris.target]), axis=1)

    a = SmallAttributeSubsets(table)
    r = a.engine(20, 2, 4)

    num = 0.3 * len(r)
    sel = sorted([e[0] for i, e in enumerate(r) if i < num])

    x_train = X_train[:,sel]
    x_test = X_test[:,sel]


    tree2 = DecisionTreeClassifier()
    clf2 = tree2.fit(x_train, y_train)
    print clf2.score(x_test, y_test)

    # print a._convert_to_numpy([(1,2,2),(2,1,1),(2,3,3)])
    # x = a._generate_attribute_subtables(6, 1, 4)
    # print x
    # a._convert_list_of_subtables_to_list_of_list_of_tuples(x)
