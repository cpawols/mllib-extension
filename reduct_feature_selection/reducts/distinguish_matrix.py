# coding=utf-8
"""
Class computes distinguish table.
"""
import numpy as np
from collections import Counter


class DistniguishMatrixCreator:
    """
    TODO
    """

    def __init__(self, table_initial):
        self.table = table_initial

    def create_distinguish_table(self):
        """
        Returns dictionary which represents distinguish table.
        :return: dictionary.
        """
        result_dictionary = dict()
        for i, f_row in enumerate(self.table):
            for j, s_row in enumerate(self.table):
                if i != j and f_row[-1] != s_row[-1] and i < j:
                    result_dictionary.update(DistniguishMatrixCreator.process_objects(f_row, i, j, s_row))

        return result_dictionary

    @staticmethod
    def process_objects(f_row, i, j, s_row):
        """
        Creates distinguish row between two objects.
        :param f_row: row of original matrix
        :param i: number of  f_row
        :param j: number od s_row
        :param s_row:
        :return: distinguish row
        """
        result_dictionary = {}
        for attr_num in range(len(f_row) - 1):
            if f_row[attr_num] != s_row[attr_num]:
                if (i, j) not in result_dictionary:
                    result_dictionary[(i, j)] = [attr_num]
                else:
                    result_dictionary[(i, j)].append(attr_num)
        return result_dictionary

    @staticmethod
    def get_attributes_frequency(distinguish_matrix):
        """
        Returns counter with attributes frequency.
        :param distinguish_matrix: dictionary.
        :return: counter with attributes frequency.
        """
        return Counter(attr for attr_list in distinguish_matrix.values() for attr in attr_list)


if __name__ == "__main__":
    table = np.array([[1, 2, 3], [2, 1, 1], [1, 2, 1], [0, 0, 0], [0, 1, 0]])
    ds = DistniguishMatrixCreator(table)
    com = ds.create_distinguish_table()
