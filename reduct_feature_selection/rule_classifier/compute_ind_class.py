import numpy as np
from settings import sc

__author__ = 'krzysztof'


def compute_ind_class(decision_system, attr_set, subtable_num):
    """
    :param decision_system: (last column is decision column)
    :type decision_system: np.array
    :param attr_set:
    :type attr_set: list (numbers of columns)
    :return: list of tuples (rule, rule_support)
    """
    rows = decision_system.shape[0]
    cols = decision_system.shape[1]
    attrs = attr_set.append(cols - 1)  # adding decision
    subtable = decision_system[:rows, attrs]
    subtable_as_list = list([tuple(row) for row in subtable])   # tu jest coś z typem atomowym w tupli (powinien być prosty chyba a jest numpy'owy)
    num_chunks = subtable_num
    rdd_subtable = sc.parallelize(subtable_as_list, num_chunks)
    rules = rdd_subtable.mapPartitions(lambda x: (x, 1)) \
        .reduceByKey(lambda x, y: x + y).reduce(lambda x, y: x.extend(y)) #reducyBykey się wykrzacza, do reducera trzeba dopisać funkcję merge'ująca
    return rules
