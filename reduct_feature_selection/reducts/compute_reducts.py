# coding=utf-8
"""Compute reducts"""
import numpy as np
from random import randint, shuffle

from reduct_feature_selection.reducts.distinguish_matrix import DistniguishMatrixCreator


class ReductsCreator:
    """ Reducts creator""
    """

    def __init__(self, initial_table, subtable_number, minimum, maximum, ind_original_relation):
        self.table = initial_table
        self.subtable_number = subtable_number
        self.minimum = minimum
        self.maximum = maximum
        self.ind_original_relation = ind_original_relation

    def choose_attibutes_for_subtables(self):
        """
        Returns number of attributes which will be chosen to subtables.
        :return: list of lists.
        """
        return [sorted(
            list(np.random.choice(range(0, self.table.shape[1]), randint(self.minimum, self.maximum), replace=False)))
                for _ in range(self.subtable_number)]

    @staticmethod
    def compute_up_reduct(distinguish_table):
        """
        This function return reducts
        :param distinguish_table distinguish_table
        :return: list of reducts and upredcts.
        """
        list_of_reducts = []
        for _ in range(1000):
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

    def check_if_reduct(self, list_of_reducts):
        """
        Remove from list upreducts.
        :param list_of_reducts:
        :return:
        """

        attributes_number = self.table.shape[1]
        reduct_list = []

        for reduct in list_of_reducts:
            is_reduct = True
            attr_subtable = list(reduct)
            attr_subtable.append(attributes_number - 1)
            for attribute in reduct:
                attr_table = filter(lambda x: x != attribute, attr_subtable)
                up_reduct_abstraction_class = ReductsCreator.indiscernibility_relation(self.table[:, attr_table])
                if up_reduct_abstraction_class == self.ind_original_relation:
                    is_reduct = False
            if is_reduct is True:
                reduct_list.append(reduct)
        return reduct_list


if __name__ == "__main__":
    pass
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

    # table = np.array([
    #     [1, 1, 0, 0, 0],
    #     [1, 1, 0, 1, 0],
    #     [2, 1, 0, 0, 1],
    #     [0, 2, 0, 0, 1],
    #     [0, 0, 1, 0, 1],
    #     [0, 0, 1, 1, 0],
    #     [2, 0, 1, 1, 1],
    #     [1, 2, 0, 0, 0],
    #     [1, 0, 1, 0, 1],
    #     [0, 2, 1, 0, 1],
    #     [1, 2, 1, 1, 1],
    #     [2, 2, 0, 1, 1]
    # ])
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

    # list_of_reducts = reduct_generator.compute_up_reduct(ds)
    # print list_of_reducts
    # print reduct_generator.check_if_reduct(list_of_reducts)
