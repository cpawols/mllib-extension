"""
Creating distinguish table using spark

Divide decision system into attribute chunks:
    - transpose matrix
    - divide by split from numpy (number of chunks depends on number of computers on the cluster)
    each chunks get a list with decisions
    - for each chunk compute distinguish table which is saved as dictionary
        keys are a tuple of position in matrix and values are list attributes which distinguish the objects
        - make statistic of frequency for each object something more??? TODO
    - merge dictionaries updating values on each key

As input we take numpy array which will be converted to list of tuples because spark rdd object doesn't support np.array
"""
import numpy as np
from collections import Counter
from numpy.ma import transpose, copy

from settings import Configuration

import reduct_feature_selection


class DistinguishTable:
    def __init__(self, decision_system):
        self.decision_system = decision_system

    @staticmethod
    def _transopse_matrix(matrix):
        """Transpose a numpy matrix"""
        decision_system_transpose = transpose(matrix)
        return decision_system_transpose

    @staticmethod
    def _get_decision_list(matrix):
        """Return a decision list"""
        return matrix[:, -1]

    @staticmethod
    def _convert_to_list_of_tuples(local_decision_system):
        """Convert np.array to list of tuples"""
        list_of_tuples = [tuple(row, ) + (i,) for i, row in enumerate(local_decision_system)]
        return list_of_tuples

    @staticmethod
    def _remove_decision_column(matrix):
        """Removing decision column from np array"""
        return np.delete(matrix, -1, 1)

    def _prepare_data_make_distinguish_table(self):
        """Preparing decision system"""
        # TODO make it cleaner
        decision_system_copy = copy(self.decision_system)
        decisions = self._get_decision_list(decision_system_copy)
        decision_system_copy = self._remove_decision_column(decision_system_copy)
        transpose_decision_system = self._transopse_matrix(decision_system_copy)
        list_of_tuples = self._convert_to_list_of_tuples(transpose_decision_system)
        return list_of_tuples, decisions

    @staticmethod
    def join_dictionaries(dictionary_x, dictionary_y):
        """Join two dictionaries into one
        :param dictionary_x:
        :param dictionary_y:
        """
        dicts = [dictionary_x, dictionary_y]
        super_dict = {}
        for d in dicts:
            for k, v in d.iteritems():
                if k in super_dict:
                    super_dict[k] = super_dict[k] + [element for element in v]
                else:
                    super_dict[k] = [element for element in v]
        return super_dict

    @staticmethod
    def make_table(system, decisions=None, bor_sc=None):
        # TODO documentation
        """ Computing decision table
        :param bor_sc:
        :param decisions:
        :param system:
        """
        result_dictionary = dict()
        for i, attributes in enumerate(system):
            for j in range(len(attributes) - 1):
                for k in range(j, len(attributes) - 1):
                    if j != k and attributes[j] != attributes[k] and decisions[j] != decisions[k]:
                        if (j, k) not in result_dictionary:
                            result_dictionary[(j, k)] = [attributes[-1]]
                        else:
                            result_dictionary[(j, k)].append(attributes[-1])
        return result_dictionary

    def _check_containing(self, list_for_object, dictionary_for_implicant):
        # TODO Better documentation
        """
        This method checked if implicant for object contains element
        :param list_for_object:
        :param dictionary_for_implicant:
        :return:
        """
        for element in list_for_object:
            if element in dictionary_for_implicant:
                return True
        return False

    def _compute_implicants(self, x, bor_sc=None):
        # TODO in english
        """
        This method compute implicants and saved it into dictionary in which key is a tuple containing attributes number
        and a value is a list of values of those attributes, the last element in this list is a decision.
        :param x:TODO
        :param bor_sc: TODO
        :return: dictionary with rules.
        """
        y = copy(x)
        feequency_of_attributes = self.frequency_of_attibutes(x)
        implicants = {}
        print feequency_of_attributes
        print y
        for objekt, attributes in x.items():
            if objekt[0] in implicants:
                # Jesli obiekt jest to nalezy sprawdzic czy potrzebujemy dokladac nowy atrybut jesli tak, to to robimy
                # W sposob heurystyczny
                # print 'obiekt', objekt
                if not self._check_containing(attributes, implicants[objekt[0]]):
                    # Frequency of attributes it's a list of tuples - first element - attribute number,
                    #  second - frequency
                    self.first_heuristic_method(feequency_of_attributes, implicants, objekt[0])
            else:
                for attr in feequency_of_attributes:
                    if attr[0] in attributes:
                        implicants[objekt[0]] =[attr[0]]
                        break

            if objekt[1] in implicants:
                if not self._check_containing(attributes, implicants[objekt[1]]):
                    # Frequency of attributes it's a list of tuples - first element - attribute number,
                    #  second - frequency
                    self.first_heuristic_method(feequency_of_attributes, implicants, objekt[1])
            else:
                for attr in feequency_of_attributes:
                    if attr[0] in attributes:
                        implicants[objekt[1]] = [attr[0]]
                        break
        # Klucz - obiekt, wartosc, numery atrybutow pozwalajace go rozrozniac

        yield implicants
        # yield y it's work

    def first_heuristic_method(self, feequency_of_attributes, implicants, objekt):
        """
        TODO
        :param feequency_of_attributes:
        :param implicants:
        :param objekt:
        :return:
        """
        #print 'imp', implicants
        for attribute in feequency_of_attributes:
            if attribute[0] not in implicants[objekt]:
                implicants[objekt].append(attribute[0])
                break
                # Heuristic

    def spark_part(self, conf=None, number_of_chunks=2):
        # TODO documentation - cleaning method
        """

        :param conf:
        :param number_of_chunks:
        :return:
        """
        system, decisions = self._prepare_data_make_distinguish_table()
        system_rdd = Configuration.sc.parallelize(system, number_of_chunks)
        result = (system_rdd.mapPartitions(lambda x: self.make_table(x, decisions)).mapPartitions(
            lambda x: self._compute_implicants(x)).collect())
                  #.reduce(self.join_dictionaries))
        return result

    @staticmethod
    def frequency_of_attibutes(dictionary):
        # TODO documentation
        """

        :param dictionary:
        :return:
        """
        return Counter([values for element in dictionary.values() for values in element]).most_common()


if __name__ == "__main__":
    decision_system = np.array([[1, 0, 2, 1], [1, 1, 2, 0], [1, 1, 1, 1], [3, 3, 3, 1], [2, 1, 0, 0]])
    # conf = SparkConf().setAppName("aaaa")
    A = DistinguishTable(decision_system)
    result = A.spark_part(number_of_chunks=1)
    print result
    pass
