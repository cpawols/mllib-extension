# coding=utf-8
"""Compute reducts"""
import numpy as np
import scipy.io as sio
from collections import Counter
from math import ceil
from numpy.ma import ravel
from random import randint, shuffle
from sklearn.cross_validation import train_test_split
from sklearn.tree import tree

from reduct_feature_selection.reducts.distinguish_matrix import DistniguishMatrixCreator
from settings import Configuration

x = sio.loadmat('/home/pawols/Develop/Mgr/mgr/BASEHOCK.mat')
table = np.append(x['X'], x['Y'], axis=1)
broadcast_table = Configuration.sc.broadcast(table)


class ReductsCreator:
    """
    Reducts creator.
    This class creates a list of reducts.
    """

    def __init__(self,
                 initial_table,
                 ind_original_relation,
                 subtable_number=1,
                 minimum=1,
                 maximum=1,
                 columns=None,
                 attr_rank=None):

        self.table = initial_table
        self.subtable_number = subtable_number
        self.minimum = minimum
        self.maximum = maximum
        self.ind_original_relation = ind_original_relation
        self.columns = columns
        self.attibute_rank = attr_rank

    def choose_attibutes_for_subtables(self):
        """
        Returns attributes which will be chosen to subtables with decision attribute.
        :return: list of lists.
        """
        return [sorted(
            list(np.random.choice(range(0, self.table.shape[1]), randint(self.minimum, self.maximum), replace=False)))
                for _ in range(self.subtable_number)]

    @staticmethod
    def shuffle_attributes_group(attributes_group):
        """
        Returns shuffled ranging of attributes.
        :param attributes_group: list of list containing groups of attributes.
        :return: list with attribute ranking.
        """
        shuffeled_groups = []
        for group in attributes_group[:]:
            shuffle(group)
            shuffeled_groups.extend(group)
        return shuffeled_groups

    @staticmethod
    def get_attributes_group(ranking_of_attributes):
        """
        Shuffles attributes for 'Johnson' heuristic.
        :param ranking_of_attributes:
        :return:
        """
        shuffeled_attributes = ranking_of_attributes.most_common()[:]
        mean_difference = ReductsCreator.compute_mean_difference(shuffeled_attributes)
        print mean_difference
        return ReductsCreator.compute_attributes_grup(mean_difference, shuffeled_attributes)

    @staticmethod
    def compute_attributes_grup(mean_difference, shuffeled_attributes):
        """
        Computes attributes groups of significant attributes.
        :param mean_difference:
        :param shuffeled_attributes:
        :return: group of attributes.
        """
        result = []
        added_num = 0
        added = set()
        for i, attributes in enumerate(shuffeled_attributes):
            if shuffeled_attributes[i] not in added:
                tmp = [shuffeled_attributes[i]]
                added.add(shuffeled_attributes[i])
                added_num += 1
                for j in range(i + 1, len(shuffeled_attributes)):
                    if abs(tmp[-1][1] - shuffeled_attributes[j][1]) <= mean_difference:
                        tmp.append(shuffeled_attributes[j])
                        added.add(shuffeled_attributes[j])
                        added_num += 1
                    else:
                        break

                result.append(tmp)
                if added_num == len(shuffeled_attributes):
                    break
        return result

    @staticmethod
    def compute_mean_difference(most_common_attributes):
        """
        Computes mean difference of frequency occurrence of attributes.
        :param most_common_attributes: list of
        :return: mean of difference frequency.
        """
        end = 1
        difference = 0
        while end < len(most_common_attributes):
            difference += most_common_attributes[end - 1][1] - most_common_attributes[end][1]
            end += 1
        mean_difference = ceil(1. * difference / len(most_common_attributes))
        return mean_difference

    @staticmethod
    def compute_up_reduct(distinguish_table, ranking=None, iter_number=18):
        """
        This function return reducts
        :param distinguish_table distinguish_table
        :return: list of reducts and upredcts.
        """
        list_of_reducts = []
        for _ in range(iter_number):
            """ Make it cleaver!"""
            if ranking is None:
                local_distinguish_table = distinguish_table.copy()  # It's possible to use dict(distinguish_table)
                local_attributes_frequency = DistniguishMatrixCreator.get_attributes_frequency(distinguish_table)
                local_attributes_frequency = local_attributes_frequency.most_common()[:]
                shuffle(local_attributes_frequency)
            else:
                at_group = ReductsCreator.get_attributes_group(ranking)
                local_attributes_frequency = ReductsCreator.shuffle_attributes_group(at_group)
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
        Remove from upreducts form list of potentially reducts.
        :param list_of_reducts: list of potentially reducts (possible upreducts)
        :return: lis of reducts.
        """

        attributes_number = decision_table.shape[1]
        reduct_list = []

        for reduct in list_of_reducts:

            attr_subtable = list(reduct)
            attr_subtable.append(attributes_number - 1)
            erase = []
            for attribute in reduct:
                attr_table = filter(lambda x: x != attribute, attr_subtable)
                up_reduct_dist = DistniguishMatrixCreator.create_distingiush_matrix_sort_table(
                    decision_table[:, attr_table])
                for pair in up_reduct_dist:
                    if pair not in self.ind_original_relation:
                        continue
                    erase.append(False)
                    break
            if len(erase) == len(reduct) and reduct not in reduct_list:
                reduct_list.append(reduct)

        return reduct_list

    @staticmethod
    def renumerate_attributes_in_list_ofreducts(list_of_reducts, choosen_attributes):
        """
        Returns list of reduct with renumber attributes.
        :param list_of_reducts: list of reducts.
        :param choosen_attributes: list of original number of attributes.
        :return: list of reducts.
        """
        res = []
        for reduct in list_of_reducts:
            res.append([choosen_attributes[e] for e in reduct])
        return res


class ComputeReductsForSubtables:
    """
    Class which allows to compute reducts for given decision system.
    """

    def __init__(self, initial_table, sub_num, min, max):
        """
        Parameter of class.
        :param initial_table: decision table.
        :param sub_num:  number of subtables.
        :param min: minimal length of subtable.
        :param max: maximal length of subtable.
        """
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

    def generate_object_subtables(self, default_subtable_number=200):
        """
        Creates object subtables of given decision system.
        :return: list of object numbers.
        """
        object_list = [sorted(
            list(np.random.choice(range(0, self.table.shape[0] - 1), randint(self.minimum, self.maximum),
                                  replace=False)))
                       for _ in range(default_subtable_number)]
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
    def compute_attribute_ranking_for_reducts(table, choosen_attributes, num_objects=None):
        """
        Computes reducts for given table.
        :param table: decision table (numpy array).
        :return: list of list with reducts.
        """
        if num_objects is not None:
            objects = list(np.random.choice(range(0, table.shape[0] - 1), size=200, replace=False))
            table = table[objects, :]

        ds_creator = DistniguishMatrixCreator(table)
        ds_matrix = ds_creator.create_distinguish_table()

        reduct_generator = ReductsCreator(table, sorted(ds_matrix.keys()))
        list_of_reducts = reduct_generator.compute_up_reduct(ds_matrix)
        reducts = reduct_generator.check_if_reduct(table, list_of_reducts)
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
        return a


class SkowronDreamHeuristic:
    """
    TODO - full description of algorithm.
    """

    def __init__(self, decision_table, sub_number, min, max, object_subtable=False, size=200):
        """
        TODO Make it in the feature.
        :return:
        """
        self.table = decision_table
        self.sub_number = sub_number
        self.min = min
        self.max = max
        self.object_subtable = object_subtable
        self.object_subtables_size = size

    def generate_attributes_for_subtables(self):
        """
        Returns list of attributes.
        :return: list of attributes.
        """
        attributes_list = [sorted(
            list(np.random.choice(range(0, self.table.shape[1] - 1), randint(self.min, self.max),
                                  replace=False)))
                           for _ in range(self.sub_number)]

        object_list = [
            sorted(list(np.random.choice(range(0, self.table.shape[0]), self.object_subtables_size, replace=False))) for
            _ in
            range(self.sub_number)]

        attributes = [e[:] + [self.table.shape[1] - 1] for e in attributes_list]
        result = []
        for x, y in zip(object_list, attributes):
            result.append((x, y))
        return result

    @staticmethod
    def first_stage(table):
        """
        TODO
        :param table:
        :return:
        """
        # print broadcast_table.value[x[0],:][:,x[1]]
        ds_creator = DistniguishMatrixCreator(table)
        ds_table = ds_creator.create_distinguish_table()
        return Counter(e for attr in ds_table.values() for e in attr)

    def second_stage(self, table, ranking, choosen_attributes, up_reduct=False):
        """
        TODO
        :param table:
        :param ranking:
        :param choosen_attributes:
        :return:
        """
        ds_creator = DistniguishMatrixCreator(table)
        ds_table = ds_creator.create_distinguish_table()

        reduct_creator = ReductsCreator(table, sorted(ds_table.keys()), attr_rank=ranking)
        list_of_up_reducts = reduct_creator.compute_up_reduct(ds_table)

        if up_reduct is False:
            reducts = reduct_creator.check_if_reduct(table, list_of_up_reducts)
        else:
            reducts = list_of_up_reducts

        return ReductsCreator.renumerate_attributes_in_list_ofreducts(reducts,
                                                                      choosen_attributes=choosen_attributes)

    def spark_driver(self, up_reducts=False):
        """
        TODO
        :return:
        """

        x = self.generate_attributes_for_subtables()
        x_rdd = Configuration.sc.parallelize(x)

        ranking = x_rdd \
            .map(lambda x: SkowronDreamHeuristic.first_stage(broadcast_table.value[x[0], :][:, x[1]])) \
            .reduce(lambda x, y: x + y)

        reducts = x_rdd.map(lambda x: self.second_stage(broadcast_table.value[x[0], :][:, x[1]], ranking,
                                                        choosen_attributes=x[1], up_reduct=up_reducts)).reduce(
            lambda x, y: x + y)
        return reducts



def traiin_full_tree():
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(
        x['X'], x['Y'], test_size=0.2, random_state=42)
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    print clf.score(X_test, y_test)


def load_data():
    global x, table
    x = sio.loadmat('/home/pawols/Develop/Mgr/mgr/BASEHOCK.mat')
    table = np.append(x['X'], x['Y'], axis=1)


def scale_attributes():
    global e
    res = Counter(e for a in r for e in a)
    all_occurence = sum(res.values())
    tmp = res.most_common()
    res_scal = []
    for e in tmp:
        res_scal.append((e[0], 1.0 * e[1] / all_occurence))
    return res_scal

def score_knn():
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier()
    knn.fit(X_train[:, sorted(sel)], ravel(y_train))
    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                         metric_params=None, n_jobs=1, n_neighbors=5, p=2,
                         weights='uniform')
    print i, knn.score(X_test[:, sorted(sel)], y_test)


def score_tree():
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train[:, sorted(sel)], y_train)
    print len(sel), clf.score(X_test[:, sorted(sel)], y_test)


if __name__ == "__main__":

    load_data()
    traiin_full_tree()


    heuristic = SkowronDreamHeuristic(table, 40, 500, 4550, size=600)
    r = heuristic.spark_driver(up_reducts=True)

    res_scal = scale_attributes()

    for i in range(1, 1050, 3):
        sel = [e[0] for j, e in enumerate(res_scal) if j < i]
        # score_knn()
        score_tree()


    """
    26 minut 500, 4500, 150 ze skracaniem nadreduktów do reduktów
    Najlepsze ustawienia
    heuristic = SkowronDreamHeuristic(table, 200, 5, 50, size=1050)
    """