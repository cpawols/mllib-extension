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
import random
from collections import Counter
from numpy.ma import transpose, copy

import reduct_feature_selection
from settings import Configuration


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
        """
        Convert (numpy array) decision system into list of tuples in which the tuple represent object from original
         decision system. Converted decision system doesn't contain a decisions, decisions are returned as a second list
         Example

        :return: list of tuples, each tuple represent one object
        :return: list of decisions
        """

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

    def _compute_implicants(self, distinguish_table, heuristic_type, bor_sc=None):
        # TODO in english
        """
        This method compute implicants and saved it into dictionary in which key is a tuple containing attributes number
        and a value is a list of values of those attributes, the last element in this list is a decision.
        :param distinguish_table:TODO
        :param bor_sc: TODO
        :return: dictionary with rules.
        """
        # heuristic_type = self.first_heuristic_method
        # y = copy(x)
        attributes_frequency = self.frequency_of_attibutes(distinguish_table)
        implicants = dict()

        for object, attributes in distinguish_table.items():
            if object[0] in implicants:
                # Jesli obiekt jest to nalezy sprawdzic czy potrzebujemy dokladac nowy atrybut jesli tak, to to robimy
                # W sposob heurystyczny
                # print 'obiekt', objekt
                if not self._check_containing(attributes, implicants[object[0]]):
                    # Frequency of attributes it's a list of tuples - first element - attribute number,
                    #  second - frequency
                    # self.first_heuristic_method(attributes_frequency, implicants, objekt[0])
                    heuristic_type(attributes_frequency, implicants, object[0])
            else:
                for attr in attributes_frequency:
                    if attr[0] in attributes:
                        implicants[object[0]] = [attr[0]]
                        break

            if object[1] in implicants:
                if not self._check_containing(attributes, implicants[object[1]]):
                    # Frequency of attributes it's a list of tuples - first element - attribute number,
                    #  second - frequency
                    # self.first_heuristic_method(attributes_frequency, implicants, objekt[1])
                    heuristic_type(attributes_frequency, implicants, object[1])
            else:
                for attr in attributes_frequency:
                    if attr[0] in attributes:
                        implicants[object[1]] = [attr[0]]
                        break
        # Klucz - obiekt, wartosc, numery atrybutow pozwalajace go rozrozniac
        # print 'Impli',  implicants
        print 'Implicants', len(implicants)
        return implicants
        # yield y it's work

    def validate_rules(self, rules, validation_function, original_decision_system=None, treshold=0.05):
        # Gets a rules from each part and return this rules which
        # have a good answer in more than 'treshold' cases.
        # Maybe we have to accept the rules which have a contradict value??
        # Example of this types
        # a1 = 0 && a2 =1 -> dec = 1
        # a1 = 0 && a2 =1 -> dec = 0
        # It makes sense
        # TODO Rewrite into better way, cleaning code, implement new heuristic
        """
        :param rules:
        :param original_decision_system:
        :param treshold:
        :return:
        """
        accepted_rules = dict()
        numbers_of_all_objects = len(original_decision_system)
        for attr_numbers, attr_values in rules.items():
            rules_accepts = 0
            bad_rules = 0
            for object in original_decision_system:
                rule_istrue = True
                for attr_number, attr_value in zip(attr_numbers[:-1], attr_values[:-1]):
                    if object[attr_number] != attr_value:
                        rule_istrue = False
                        break
                if rule_istrue is True and attr_values[-1] == object[-1]:
                    rules_accepts += 1
                    # print 'Doskonala regula', attr_numbers[:-1], attr_values
                elif rule_istrue is True and attr_values[-1] != object[-1]:
                    bad_rules += 1
                    # print 'Zla regula!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'

            # print numbers_of_all_objects, bad_rules, rules_accepts
            # if numbers_of_all_objects - bad_rules >= treshold * numbers_of_all_objects:
            self.first_validation_function_to_rename(accepted_rules, attr_numbers, attr_values, bad_rules,
                                                     numbers_of_all_objects, treshold)

        print 'Accepted rules', len(accepted_rules)
        return accepted_rules

    def first_validation_function_to_rename(self, accepted_rules, attr_numbers, attr_values, bad_rules,
                                            numbers_of_all_objects, treshold):
        # TODO
        """

        :param accepted_rules:
        :param attr_numbers:
        :param attr_values:
        :param bad_rules:
        :param numbers_of_all_objects:
        :param treshold:
        :return:
        """
        if bad_rules <= treshold * numbers_of_all_objects:
            accepted_rules[attr_numbers] = attr_values

    def generate_rules_from_implicants(self, implicants, original_decision_system=None):
        # TODO
        """

        :param implicants:
        :return:
        """
        rules = dict()

        for object_number, attributes in implicants.items():
            for attribute in attributes:
                rules[tuple(attributes) + (object_number,)] = \
                    [value_of_attribute for i, value_of_attribute in enumerate(original_decision_system[object_number])
                     if i in attributes]
                rules[tuple(attributes) + (object_number,)].append(original_decision_system[object_number][-1])
        print ''
        return rules

    def print_rules(self, rules):
        """
        TODO
        :param rules:
        :return:
        """
        for attributes, values in rules.items():
            rule = ' '
            for i, (attribute, value) in enumerate(zip(attributes[:-1], values[:-1])):
                if i != 0:
                    rule += ' && ' + 'attr(' + str(attribute) + ')' + " = " + str(value)
                else:
                    rule += 'attr(' + str(attribute) + ')' + " = " + str(value)
                    # print attribute, '=', value
            rule += '  --> dec = ' + str(values[-1])
            print rule

    def get_attribute_rank_from_rules(self, rules, treshold=None):
        # TODO
        """
        First element it's a attribut number, the secon frequency of those attribute
        :param rules:
        :return:
        """
        if treshold is None:
            return Counter(attribute for attributes in rules.keys() for attribute in attributes[:-1]).most_common()
        else:
            return Counter(
                attribute for attributes in rules.keys() for attribute in attributes[:-1]).most_common(treshold)

    # @staticmethod
    def first_heuristic_method(self, attributes_frequency, implicants, object):
        """
        Heuristic method of coverage object by this attributes which are the most frequent
        :param attributes_frequency: Counter
        :param implicants:
        :param object:
        :return:
        """
        for attribute in attributes_frequency:
            if attribute[0] not in implicants[object]:
                implicants[object].append(attribute[0])
                break

    def spark_part(self, conf=None, number_of_chunks=2):
        # TODO documentation - cleaning method
        """

        :param conf:
        :param number_of_chunks:
        :return:
        """
        system, decisions = self._prepare_data_make_distinguish_table()
        system_rdd = Configuration.sc.parallelize(system, number_of_chunks)
        result = (system_rdd.mapPartitions(
            lambda x: self.make_table(x, decisions)).mapPartitions(
            lambda x: self._compute_implicants(x, self.first_heuristic_method)).mapPartitions(
            lambda x: self.generate_rules_from_implicants(x,
                                                          original_decision_system=self.decision_system)).mapPartitions(
            lambda x: self.validate_rules(x, validation_function=self.first_validation_function_to_rename,
                                          original_decision_system=self.decision_system)).collect())
        # .reduce(self.join_dictionaries)),
        print result
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
    implicants = {0: [1], 1: [1, 2], 2: [2], 3: [2], 4: [2]}
    # conf = SparkConf().setAppName("aaaa")
    rules = {(1, 2, 1): [1, 2, 0], (1, 0): [0, 1], (2, 3): [3, 1], (2, 4): [0, 0], (2, 2): [1, 1]}

    d = np.array([[random.randint(0, 4) for _ in range(10)] for __ in range(10)])
    A = DistinguishTable(d)
    A.spark_part(decision_system, number_of_chunks=4)
    # A.print_rules(rules)
    # print A.get_attribute_rank_from_rules(rules)
    # A.generate_rules_from_implicants(implicants, original_decision_system=decision_system)
    # result = A.spark_part(number_of_chunks=1)
    # print result
    pass

    # TODO Tests
    # TODO Generate Rules from implicants
    # TODO First heuristic method
    # TODO Validate Rules
    # TODO Compute implicants
    # TODO

    # TODO merge rules by keys
    # TODO Take a negation of the rules
