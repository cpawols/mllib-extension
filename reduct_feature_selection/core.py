"""
TODO Describe algorithm for core computing
"""
import numpy as np

from reduct_feature_selection.commons.tables.distinguish_table import DistinguishTable
from settings import Configuration


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
    is_added = dict()  # Dictionary contains information which field was added to table
    for i, attributes in enumerate(system):
        for j in range(len(attributes) - 1):
            for k in range(j, len(attributes) - 1):
                if j != k and attributes[j] != attributes[k] and decisions[j] != decisions[k]:
                    if (j, k) not in result_dictionary:
                        result_dictionary[(j, k)] = [attributes[-1]]
                    else:
                        result_dictionary[(j, k)].append(attributes[-1])

    yield result_dictionary


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




def _get_core(ds_table):
    return set([attr[0] for field, attr in ds_table.items() if len(attr) == 1])


def compute_core():
    # global decisions
    ds = DistinguishTable(table)
    system, decisions = ds._prepare_data_make_distinguish_table()
    system_rdd = Configuration.sc.parallelize(system, 3)
    result = (system_rdd.mapPartitions(
        lambda x: make_table(x, decisions)).reduce(
        lambda x, y: join_dictionaries(x, y)))
    return _get_core(result)


if __name__ == "__main__":
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
    print compute_core()
