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

from sklearn.cross_validation import train_test_split
import reduct_feature_selection

from sklearn import tree
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

        np.array[[1,0,1],[1,0,0]] Will be converted to the following list of tuples
        [(1,0), (1,0)]
        The second argument which will be returned is the following list [1,0]

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
        # TODO Temporaty and test function only - rebasing
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
    def make_table(system, decisions, bor_sc=None):
        """
        This function take a list of tuples in which each tuple represent one object from decision system (without
        decision column) and returns a dictionary which represents a distinguish table.
        As a keys in this dictionary are a tuples of two integers which represents number of two object.
        As a values this dictionary contains a list with numbers of attributes which allow to distinct those two objects

        Example If we have the following decision table:

             a_0  a_1  a_2  a_3  dec
        X_0   0    1    1    0    +
        X_1   1    1    0    1    -

        Returned dictionary will be the following:

        {(0,1):[0,2,3]}

        :param system: list of tuples
        :param decisions: list with decisions
        :param bor_sc:  # TODO
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

    def _check_containing(self, list_for_object, implicant):
        """
        This method check if it's necessary to add new attribute to implicant.
        If implicant for th

        :param list_for_object: List of attribute's numbers which allow to distinct this object
        :param implicant: List of attributes which are in implicant for object
        (We know for which object containing is checking)
        :return: Boolean - True in case when it's necessary to add new attribute, else False
        """
        for element in list_for_object:
            if element in implicant:
                return True
        return False

    def _compute_implicants(self, distinguish_table, heuristic_type, bor_sc=None):
        # TODO in english
        """
        This method compute one implicant for each object. As a key is a integer which refer to
        object number, as a value is a list of attributes number

        Example implicant:

        {0:[1,4,5]} It's means that to distinct object number 0 we have to use attributes 1,4,5

        :param distinguish_table:TODO
        :param bor_sc: TODO
        :return: dictionary with rules.
        """
        attributes_frequency = self.frequency_of_attributes(distinguish_table)
        implicants = dict()

        for object, attributes in distinguish_table.items():
            if object[0] in implicants:
                # If object is added to implicats we check if it's necessary to adding nex attribute
                if not self._check_containing(attributes, implicants[object[0]]):
                    # If it's necessary we add new attribute using one of heuristic
                    heuristic_type(attributes_frequency, implicants, object[0])
            else:
                # Add object to implicant set
                for attr in attributes_frequency:
                    if attr[0] in attributes:
                        implicants[object[0]] = [attr[0]]
                        break
            if object[1] in implicants:
                # If object is added to implicats we check if it's necessary to adding nex attribute
                if not self._check_containing(attributes, implicants[object[1]]):
                    # If it's necessary we add new attribute using one of heuristic
                    heuristic_type(attributes_frequency, implicants, object[1])
            else:
                # Add object to implicant set
                for attr in attributes_frequency:
                    if attr[0] in attributes:
                        implicants[object[1]] = [attr[0]]
                        break
        return implicants

    def validate_rules(self, rules, validation_function, original_decision_system=None, treshold=0.5):
        """
        Gets a rules from each part and return this rules which have a good answer in more than 'treshold' cases.

        :param rules: dictionary with rules
        :param original_decision_system: original decision system
        :param treshold: treshold which is pass on to validation function
        :return:
        """
        print 'Len of rules', len(rules)
        accepted_rules = dict()
        numbers_of_all_objects = len(original_decision_system)
        for attr_numbers, attr_values in rules.items():
            rules_accepts = 0
            rejected_rules = 0
            for object in original_decision_system:
                rule_is_true = True
                for attr_number, attr_value in zip(attr_numbers[:-1], attr_values[:-1]):
                    if object[attr_number] != attr_value:
                        rule_is_true = False
                        break
                if rule_is_true is True and attr_values[-1] == object[-1]:
                    rules_accepts += 1
                elif rule_is_true is True and attr_values[-1] != object[-1]:
                    rejected_rules += 1

            validation_function(accepted_rules, attr_numbers, attr_values, rejected_rules,
                                numbers_of_all_objects, treshold)
        print 'Len of accepted rules', len(accepted_rules)
        yield accepted_rules

    def validation_function_f1(self, accepted_rules, attr_numbers, attr_values, rejected_rules,
                               numbers_of_all_objects, treshold):
        # TODO
        """
        This function update a rules which passed test
        :param accepted_rules: dictionary with rules
        :param attr_numbers:
        :param attr_values:
        :param rejected_rules:
        :param numbers_of_all_objects:
        :param treshold:
        :return:
        """
        if rejected_rules <= treshold * numbers_of_all_objects:
            accepted_rules[attr_numbers] = attr_values

    def generate_rules_from_implicants(self, implicants, original_decision_system=None):
        """
        This function generate rules from Implicants and saved to the dictionary
        The last values in key and value are number of object and decision respectively
        Example rule {(0,1,1):[1,1,0]} has the following form atr(0)=1 && attr(1) = 1 -> dec=0

        :param implicants: dictionary with implicants
        :return: dictionary with rules
        """
        rules = dict()

        for object_number, attributes in implicants.items():
            for _ in attributes:
                rules[tuple(attributes) + (object_number,)] = \
                    [value_of_attribute for i, value_of_attribute in enumerate(original_decision_system[object_number])
                     if i in attributes]
                rules[tuple(attributes) + (object_number,)].append(original_decision_system[object_number][-1])
        return rules

    def print_rules(self, rules):
        """
        Pretty Print rules
        :param rules: dictionary with rules
        :return: Nothing
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

    @staticmethod
    def get_attribute_rank_from_rules(self, rules, treshold=None):
        """
        First element it's a attribute number, the second frequency of those attribute
        :param rules: dictionary with rules
        :return: List of tuples
        """
        if treshold is None:
            return Counter(attribute for attributes in rules.keys() for attribute in attributes[:-1]).most_common()
        else:
            return Counter(
                attribute for attributes in rules.keys() for attribute in attributes[:-1]).most_common(treshold)

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

    def engine(self, conf=None, number_of_chunks=2):
        """
        Engine of all perations
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
            lambda x: self.validate_rules(x, validation_function=self.validation_function_f1,
                                          original_decision_system=self.decision_system)).
                  reduce(lambda x, y: self.join_dictionaries(x, y)))
        print result
        s = (Counter(attribute for attributes in result.keys() for attribute in attributes[:-1]))
        #print 'Attr freq ', s
        #print 'Attr len', len(s)
        return s

    @staticmethod
    def frequency_of_attributes(distinguish_table):
        """
        Returns sorted list of tuples containing attributes ant frequences
        :param distinguish_table:
        :return:
        """
        return Counter([values for element in distinguish_table.values() for values in element]).most_common()


if __name__ == "__main__":

    random.seed(1)
    # decision_system = np.array([[1, 0, 2, 1], [1, 1, 2, 0], [1, 1, 1, 1], [3, 3, 3, 1], [2, 1, 0, 0]])
    decision_system = np.array([[random.randint(0, 4) for _ in range(5000)] for __ in range(700)])
    decision_system = np.append(decision_system, np.array([[random.randint(0,1)] for _ in range(700)]),1)
    X_train, X_test, y_train, y_test = train_test_split(decision_system, decision_system[:,-1],
                                                      test_size=0.33, random_state=42)
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    result = clf.predict(X_test)
    print 'Accuracy of clear decision tree', 1.0*sum(1 for x, y in zip(y_test, result) if x==y)/len(result)

    A = DistinguishTable(decision_system)
    freq = A.engine(decision_system, number_of_chunks=4)

    select_attr = [attr for attr in freq.keys() ]
    print select_attr

    X_train = X_train[:, select_attr]
    #print X_train
    clf2 = tree.DecisionTreeClassifier()
    clf2.fit(X_train, y_train)
    result2 = clf2.predict(X_test[:,select_attr])
    print 'Accuracy of selected tree', 1.0*sum(1 for x, y in zip(y_test, result2) if x==y)/len(result2)
    # A.print_rules(rules)
    # print A.get_attribute_rank_from_rules(rules)
    # A.generate_rules_from_implicants(implicants, original_decision_system=decision_system)
    # result = A.spark_part(number_of_chunks=1)
    # print result
    pass
    # TODO merge rules by keys
    # TODO Take a negation of the rules
