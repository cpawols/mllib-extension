"""
This class compute distinguish table using decision system in array form
"""
from collections import Counter
from numpy import append
import itertools
import operator


class DistinguishTable:
    def __init__(self, decision_system):
        self.decision_system = decision_system

    @staticmethod
    def _complete_distinguish_table(decision_column_number, distinguish_matrix,
                                    first_object, i, j, second_object):
        """
        This function compute which of attributes for to objects
        (with different decision) are different
        and put this attributes to distinguish matrix
        :param decision_column_number: number of decision column
        :param distinguish_matrix: distinguish matrix (array of sets)
        :param first_object: one of row decision system (object)
        :param i: number of first object
        :param j: number of second object
        :param second_object: one of row in decision system (object)
        :return: Nothing
        """
        different_elements = set()
        for k, (first_element, second_element) in (
                enumerate(zip(first_object, second_object))):
            if first_element != second_element and (k < decision_column_number):
                different_elements.add(k)
        distinguish_matrix[i][j] = set(different_elements)
        distinguish_matrix[j][i] = set(different_elements)

    @staticmethod
    def compute_distinguish_matrix(decision_system):
        """
        This function compute distinguish matrix of object from decision system
        :param decision_system: decision system
        :type decision_system: np.array
        :return: two dimensional array  containing number of attributes which
        are different for object with different decision
        """
        if decision_system.size == 0:
            raise ValueError("Empty decision system table!")

        decision_column_number = decision_system.shape[1] - 1
        distinguish_matrix = [[set() for _ in range(decision_system.shape[0])]
                              for _ in range(decision_system.shape[0])]

        for i, first_object in enumerate(decision_system):
            for j, second_object in enumerate(decision_system):
                if (first_object[decision_column_number] !=
                        second_object[decision_column_number]):
                    DistinguishTable._complete_distinguish_table(
                        decision_column_number, distinguish_matrix,
                        first_object, i, j, second_object)
        return distinguish_matrix

    @staticmethod
    def compute_frequency_of_attribute(distinguish_matrix):
        """
        This function compute frequency of attributes in distinguish table
        :param distinguish_matrix: distinguish table
        :return: Counter with frequency od each attribute
        """
        return sorted(
            dict(Counter(reduce(append,
                                (list(e) for e in itertools.chain(*distinguish_matrix))))).items(),
            key=operator.itemgetter(1))
