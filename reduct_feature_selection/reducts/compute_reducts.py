# coding=utf-8
"""Compute reducts"""
import numpy as np
import time
from collections import Counter
from random import randint, shuffle
from sklearn.cross_validation import train_test_split
from sklearn.tree import tree

from reduct_feature_selection.abstraction_class.aproximation_class_set import SetAbstractionClass
from reduct_feature_selection.reducts.distinguish_matrix import DistniguishMatrixCreator
from settings import Configuration


class ReductsCreator:
    """ Reducts creator""
    """

    def __init__(self, initial_table, ind_original_relation, subtable_number=1, minimum=1, maximum=1, columns=None):
        self.table = initial_table
        self.subtable_number = subtable_number
        self.minimum = minimum
        self.maximum = maximum
        self.ind_original_relation = ind_original_relation
        self.columns = columns

    def choose_attibutes_for_subtables(self):
        """
        Returns number of attributes which will be chosen to subtables.
        :return: list of lists.
        """
        return [sorted(
            list(np.random.choice(range(0, self.table.shape[1]), randint(self.minimum, self.maximum), replace=False)))
                for _ in range(self.subtable_number)]

    @staticmethod
    def compute_up_reduct(distinguish_table, iter_number=10):
        """
        This function return reducts
        :param distinguish_table distinguish_table
        :return: list of reducts and upredcts.
        """
        list_of_reducts = []
        for _ in range(iter_number):
            """ Make it cleaver!"""
            local_distinguish_table = distinguish_table.copy()  # It's possible to use dict(distinguish_table)
            local_attributes_frequency = DistniguishMatrixCreator.get_attributes_frequency(distinguish_table)
            local_attributes_frequency = local_attributes_frequency.most_common()[:]
            shuffle(local_attributes_frequency)
            reduct = set()

            while local_distinguish_table != {}:
                actual_attr = local_attributes_frequency[0][0]
                del local_attributes_frequency[0]
                for key, value in local_distinguish_table.items():
                    if actual_attr in value:
                        reduct.add(actual_attr)
                        del local_distinguish_table[key]
            reduct = list(reduct)
            if reduct not in list_of_reducts:
                list_of_reducts.append(reduct)
        return list_of_reducts

    @staticmethod
    def indiscernibility_relation(table):
        """
        Returns list of pairs elements which are distinguish by table.
        :param table: numpy table.
        :return: list of tuples.
        """
        dis_rel = []
        for i, f_row in enumerate(table):
            for j, s_row in enumerate(table):
                if i < j and f_row[-1] != s_row[-1] and not (f_row[:-1] == s_row[:-1]).all():
                    dis_rel.append((i, j))
        return sorted(dis_rel)

    def check_if_reduct(self, decision_table, list_of_reducts):
        """
        Remove from list upreducts.
        :param list_of_reducts:
        :return:
        """

        attributes_number = decision_table.shape[1]
        reduct_list = []

        for reduct in list_of_reducts:

            attr_subtable = list(reduct)
            attr_subtable.append(attributes_number - 1)
            erase = []
            for attribute in reduct:
                # erase = []
                attr_table = filter(lambda x: x != attribute, attr_subtable)
                up_reduct_dist = DistniguishMatrixCreator.create_distingiush_matrix_sort_table(
                    decision_table[:, attr_table])
                for pair in up_reduct_dist:
                    if pair not in self.ind_original_relation:
                        continue
                    erase.append(False)
                    break

            if len(erase)  ==  len(reduct) and reduct not in reduct_list:
                reduct_list.append(reduct)
                # if up_reduct_dist == []:
                #     erase.append(False)




        # print reduct_list
        return reduct_list


class ComputeReductsForSubtables:
    def __init__(self, initial_table, sub_num, min, max):
        self.table = initial_table
        self.subtable_number = sub_num
        self.minimum = min
        self.maximum = max

    def generate_attributes_for_subtables(self):
        """
        Returns list of attributes.
        :return: list of attributes.
        """
        attributes_list = [sorted(
            list(np.random.choice(range(0, self.table.shape[1] - 1), randint(self.minimum, self.maximum),
                                  replace=False)))
                           for _ in range(self.subtable_number)]
        return [e[:] + [self.table.shape[1] - 1] for e in attributes_list]

    def generate_object_subtables(self):
        object_list = [sorted(
            list(np.random.choice(range(0, self.table.shape[0] - 1), randint(self.minimum, self.maximum),
                                  replace=False)))
                           for _ in range(200)]
        return object_list

    @staticmethod
    def compute_reducts_for_subtables(table, choosen_attributes):
        """
        Computes reducts for given table.
        :param table: decision table (numpy array).
        :return: list of list with reducts.
        """
        ds_creator = DistniguishMatrixCreator(table)
        ds_matrix = ds_creator.create_distinguish_table()
        reduct_generator = ReductsCreator(table, sorted(ds_matrix.keys()))
        list_of_reducts = reduct_generator.compute_up_reduct(ds_matrix)

        reducts = reduct_generator.check_if_reduct(table, list_of_reducts)
        return [[choosen_attributes[attr] for reduct in reducts for attr in reduct]]

    @staticmethod
    def compute_attribute_ranking_for_reducts(table, choosen_attributes):
        """
        Computes reducts for given table.
        :param table: decision table (numpy array).
        :return: list of list with reducts.
        """
        objects = list(np.random.choice(range(0, table.shape[0]-1 ),size=200, replace=False))
        table = table[objects, :]
        start = time.time()
        ds_creator = DistniguishMatrixCreator(table)
        ds_matrix = ds_creator.create_distinguish_table()
        print 'DistTableTime: ', time.time() - start

        reduct_generator = ReductsCreator(table, sorted(ds_matrix.keys()))
        start = time.time()
        list_of_reducts = reduct_generator.compute_up_reduct(ds_matrix)
        print 'UpreductTime: ', time.time() - start
        start = time.time()
        reducts = reduct_generator.check_if_reduct(table, list_of_reducts)
        print 'ReductUpReduct: ', time.time() - start
        return Counter(choosen_attributes[attr] for red in reducts for attr in red)

    def sparkdriver(self):
        """
        Computes reducts for subtables and returns ranking of attributes.
        :return: Counter.
        """
        x = self.generate_attributes_for_subtables()



        x_rdd = Configuration.sc.parallelize(x)
        # a = x_rdd \
        #     .map(lambda x: ComputeReductsForSubtables.compute_reducts_for_subtables(self.table[:, x], x)) \
        #     .reduce(lambda x, y: x + y)
        a = x_rdd \
            .map(lambda x: ComputeReductsForSubtables.compute_attribute_ranking_for_reducts(self.table[:, x], x)) \
            .reduce(lambda x, y: x + y)
        print len(a)
        return a


if __name__ == "__main__":
    # table = np.array([
    #     [1, 1, 0, 0, 0],  # 0
    #     [1, 1, 0, 1, 0],  # 1
    #     [2, 1, 0, 0, 1],  # 2
    #     [0, 2, 0, 0, 1],  # 3
    #     [0, 0, 1, 0, 1],  # 4
    #     [0, 0, 1, 1, 0],  # 5
    #     [2, 0, 1, 1, 1],  # 6
    #     [1, 2, 0, 0, 0],  # 7
    #     [1, 0, 1, 0, 1],  # 8
    #     [0, 2, 1, 0, 1],  # 9
    #     [1, 2, 1, 1, 1],  # 10
    #     [2, 2, 0, 1, 1],  # 11
    # ])

    table = np.array([
        [1, 1, 0, 0, 0],
        [1, 1, 0, 1, 0],
        [2, 1, 0, 0, 1],
        [0, 2, 0, 0, 1],
        [0, 0, 1, 0, 1],
        [0, 0, 1, 1, 0],
        [2, 0, 1, 1, 1],
        [1, 2, 0, 0, 0],
        [1, 0, 1, 0, 1],
        [0, 2, 1, 0, 1],
        [1, 2, 1, 1, 1],
        [2, 2, 0, 1, 1]
    ])

    import scipy.io as sio
    x = sio.loadmat('/home/pawols/Develop/Mgr/mgr/BASEHOCK.mat')
    # table = np.append(table['X'], table['Y'], axis=1)


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
    # table = np.array([[randint(0, 10) for _ in range(2300)] for _ in range(700)])
    begin = time.time()
    a = ComputeReductsForSubtables(table, 1000, 50, 150)
    res = a.sparkdriver()
    end = time.time()
    print 'total time: ', end - begin
    for i in range(1, 1500, 3):
        sel = [e[0] for j, e in enumerate(res.most_common()) if j < i]

        clf = tree.DecisionTreeClassifier()
        clf.fit(X_train[:, sorted(sel)], y_train)
        print len(sel), clf.score(X_test[:, sorted(sel)], y_test)

    selected = SetAbstractionClass.cut_attributes(res.most_common())

    clf = tree.DecisionTreeClassifier()

    clf.fit(X_train[:, sorted(selected)], y_train)
    print clf.score(X_test[:, sorted(selected)], y_test)
    print len(selected)




    # print table[np.lexsort(np.fliplr(table).T)]
    #
    # table = np.array([
    #     [1, 1, 1, 0, 1],
    #     [0, 1, 1, 2, 0],
    #     [1, 1, 0, 0, 0],
    #     [0, 1, 1, 0, 1],
    #     [0, 0, 0, 1, 1]
    # ])
    #
    # table = np.array([
    #     [0, 1, 1, 1],
    #     [0, 1, 0, 1],
    #     [1, 0, 0, 0],
    #     [1, 0, 0, 1],
    #     [1, 1, 0, 0]
    #
    # ])
    #
    # ds = DistniguishMatrixCreator(table)
    # ds = ds.create_distinguish_table()
    #
    # reduct_generator = ReductsCreator(table, 2, 1, 3, sorted(ds.keys()))
    #
    # list_of_reducts = reduct_generator.compute_up_reduct(ds)
    # print list_of_reducts
    # print reduct_generator.check_if_reduct(table, list_of_reducts)
    s = "aa"
