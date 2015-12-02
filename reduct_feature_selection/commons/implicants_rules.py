"""
This class compute implcants and rules using decision system and distinguish table
Firstly we compute implicants using some heuristic - we obtain
only one implicant for one object and then we generate rules.
This heuristic is the simplest heuristic
"""
# from collections import Counter
# import itertools


class Implicants:
    def __init__(self):
        """

        :return:
        """
        pass


    @staticmethod
    def compute_implicants(distinguish_table, frequency_of_attributes):
        """
        TODO : type of params
        Using distinguish table and and frequency of attributes in this table
        this function compute implicants.
        :param distinguish_table: distinguish table of attributes
        :param frequency_of_attributes: frequency of attributes in distinguish table
        :return:
        """

        processed_objects = [False for _ in range(len(distinguish_table))]
        processed = [False for _ in range(len(distinguish_table))]
        set_of_implicants = []

        frequency_of_attributes = [e[0] for e in frequency_of_attributes]

        frequency_of_attributes = list(reversed(frequency_of_attributes))

        for num_obj, row in enumerate(distinguish_table):
            frequency_of_attribute_tmp = frequency_of_attributes[:]
            choosen_attributes = set()
            while not all(x for x in processed_objects):

                actual_attribute = frequency_of_attribute_tmp[0]
                frequency_of_attribute_tmp.pop(0)

                for i, element in enumerate(row):
                    if not processed_objects[i]:
                        if element == set():
                            processed_objects[i] = True
                        elif actual_attribute in element:
                            processed_objects[i] = True
                            choosen_attributes.add(actual_attribute)

            processed[num_obj] = True
            set_of_implicants.append(list(choosen_attributes))
            processed_objects = [False for _ in range(len(distinguish_table))]
        return set_of_implicants

    @staticmethod
    def generate_rules(input_matrix, set_of_implicants):
        """
        TODO
        :param input_matrix:
        :param set_of_implicants:
        :return:
        """

        r = {}
        for object, attributes in enumerate(set_of_implicants):
            key = attributes[:]
            key.append(object)
            r[tuple(key)] = [a for i, a in enumerate(input_matrix[object]) if i in attributes]
            r[tuple(key)] = [r[tuple(key)], input_matrix[object][-1]]
        return r

    @staticmethod
    def get_frequency_of_attributes_in_rules(rules):
        """
        TODO
        :param rules:
        :return:
        """
        count_dict = {}
        for keys in rules.keys():
            for a in keys:
                count_dict[a] = 0
        for keys in rules.keys():
            for a in keys[:-1]:
                count_dict[a] += 1

        return count_dict
