"""
TODO Describe algorithm for core computing
"""
import numpy as np

from reduct_feature_selection.commons.tables.distinguish_table import DistinguishTable
from settings import Configuration


def make_table(system, decisions, bor_sc=None):
    """
    TODO
    :param system:
    :param decisions:
    :param bor_sc:
    :return:
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

    yield result_dictionary


def join_dictionaries(dictionary_x, dictionary_y):
    """
    TODO
    :param dictionary_x:
    :param dictionary_y:
    :return:
    """

    super_dict = {}
    for d in [dictionary_x, dictionary_y]:
        for k, v in d.iteritems():
            if k in super_dict:
                super_dict[k] = super_dict[k] + v
            else:
                super_dict[k] = v
    return super_dict


def _get_core(ds_table):
    """
    TODO
    :param ds_table:
    :return:
    """
    return set([attr[0] for field, attr in ds_table.items() if len(attr) == 1])


def pr(table):
    pass

def compute_core(table):
    """
    TODO broadcast vairable
    :return:
    """
    ds = DistinguishTable(table)
    system, decisions = ds._prepare_data_make_distinguish_table()
    del table
    system_rdd = Configuration.sc.parallelize(system)
    result = (system_rdd.mapPartitions(
        lambda x: make_table(x, decisions)).reduce(
        lambda x, y: join_dictionaries(x, y)))
    return _get_core(result)


def alone_compute_Core():
    ds = DistinguishTable(table)
    system, decision = ds._prepare_data_make_distinguish_table()
    result = make_table(system, decision)
    print _get_core(next(result))


def r(x, y):
    print x, y


if __name__ == "__main__":
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
    # print compute_core()
    from random import randint

    import scipy.io as sio
    # begin = time.time()
    #
    x = sio.loadmat('/home/pawols/Develop/Mgr/mgr/BASEHOCK.mat')
    #

    #X_train,  y_train = x['X'], x['Y']
    table = np.append(x['X'], x['Y'], axis=1)
    from datetime import datetime

    a = datetime.now()
    print a
    # alone_compute_Core()
    print compute_core(table)
    b = datetime.now()
    print b - a
