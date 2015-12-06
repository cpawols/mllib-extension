from settings import sc

__author__ = 'krzysztof'


def compute_ind_class(decision_system, attr_set, subtable_num):
    """
    :param decision_system: (last column is decision column)
    :type decision_system: np.array
    :param attr_set:
    :type attr_set: list (numbers of columns)
    :return: dict of list of tuples {"partitions" : [(partition, support)], "rule" : [(rule, support)]}
    """
    rows = decision_system.shape[0]
    cols = decision_system.shape[1]
    subtable = decision_system[:rows, attr_set]
    attr_set.append(cols - 1)  # adding decision
    dec_subtable = decision_system[:rows, attr_set]

    subtable_as_list = list([tuple(row) for row in subtable])
    dec_subtable_as_list = list([tuple(row) for row in dec_subtable])

    num_chunks = subtable_num

    rdd_subtable = sc.parallelize(subtable_as_list, num_chunks)
    rdd_dec_subtable = sc.parallelize(dec_subtable_as_list, num_chunks)

    partitions = rdd_subtable.map(lambda x: (x, 1)).reduceByKey(lambda x, y: x + y).sortByKey().collect()
    rules = rdd_dec_subtable.map(lambda x: (x, 1)).reduceByKey(lambda x, y: x + y).sortByKey().collect()

    return {"partitions": partitions, "rules": rules}


def compute_rules(decision_system, attr_set, treshold, subtable_num):
    """
    :param decision_system: (last column is decision column)
    :type decision_system: np.array
    :param attr_set:
    :type attr_set: list (numbers of columns)
    :param treshold: lowerbound treshold for accuracy of rules
    :return: list of rules (last element in rule is decision)
    """
    rule_candidates = compute_ind_class(decision_system, attr_set, subtable_num)
    rule_set = []

    for rule in rule_candidates["partitions"]:
        good_rules = filter(lambda r: r[0][:-1] == rule[0]
                                      and float(r[1]) / float(rule[1]) > treshold, rule_candidates["rules"])
        rule_set.append(map(lambda x: x[0], good_rules))
    rule_set = reduce(lambda x, y: x + y, rule_set)

    return rule_set
