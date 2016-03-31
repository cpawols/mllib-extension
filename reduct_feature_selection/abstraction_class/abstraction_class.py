"""
TODO
"""
import numpy as np

from settings import Configuration
import reduct_feature_selection


class AbstractionClass:
    def __init__(self, table):
        self.table = table

    def _preprae_table(self, attr=None):
        """
        Ostatni jest numer obiektu
        :param table:
        :return:
        """
        if attr is not None:
            new_table = self.table[:, attr + [self.table.shape[1] - 1]]
        else:
            new_table = self.table

        return [tuple(attributes[:-1]) + (i,) for i, attributes in enumerate(new_table)]

    def get_abstraction_class(self, chunk_number=None, attr=None):
        """
        TODO
        :param chunk_number:
        :return:
        """
        system = self._preprae_table(attr)
        eav_rdd = Configuration.sc.parallelize(system, chunk_number)
        agregatted = eav_rdd.map(lambda x: (x[:-1], [x[-1]])).reduceByKey(lambda x, y: x + y)
        abstraction_class = []
        for tuples in agregatted.collect():
            abstraction_class.append(tuples[-1])
        return abstraction_class

    def my_get_abstraction_class(self, attr):
        """
        Check it. Not scalable?
        reduceByKey is scalable, but reduce isn't
        :param attr:
        :return:
        """
        system = self._preprae_table(attr)
        system_rdd = Configuration.sc.parallelize(system, 10)
        res = system_rdd.mapPartitions(self.my_map).reduce(self.my_reduce)
        return res.values()

    def my_map(self, list_of_tuples):

        result = {}
        for tuple in list_of_tuples:
            if tuple[:-1] in result:
                result[tuple[:-1]].append(tuple[-1])
            else:
                result[tuple[:-1]] = [tuple[-1]]
        yield result

    def my_reduce(self, x, y):
        for key, value in x.items():
            if key in y:
                y[key] += value
            else:
                y[key] = value
        return y

    def standalone_abstraction_class(self, ):
        d = {}
        for i, row in enumerate(self.table):
            if tuple(row[:-1]) in d:
                d[tuple(row[:-1])].append(i)
            else:
                d[tuple(row[:-1])] = [i]
        return d.values()

    def prepare_table(self, attr):
        if attr is not None:
            new_table = self.table[:, attr + [self.table.shape[1] - 1]]
        else:
            new_table = self.table

        return [tuple(attributes[:-1]) + (attributes[-1],) + (i,) for i, attributes in enumerate(new_table)]

    def get_pos(self, chunk_number, attr=None):
        """

        :param chunk_number:
        :param attr:
        :return:
        """

        system = self.prepare_table(attr)
        eav_rdd = Configuration.sc.parallelize(system, chunk_number)
        agr = eav_rdd.map(lambda x: (x[:-2], [x[-2], [x[-1]]])).reduceByKey(self.r2)
        q = []
        print agr.collect()
        for el in agr.collect():
            add = True
            for e in el[1][1]:
                if e == -1:
                    add = False
            if add is True:
                q += el[1][1]
        return q

    def r2(self, x, y):
        if x[0] == y[0]:
            return [x[0], x[1] + y[1]]
        return [x[0], [-1]]

    def get_positive_area(self, chunk_number, attr=None):
        """
        TODO
        :param chunk_number:
        :param attr:
        :return:
        """
        abstraction_class = self.get_abstraction_class(chunk_number, attr=attr)
        pos = []
        for ab in abstraction_class:
            a = set()
            for ob in ab:
                a.add(self.table[ob][-1])
            if len(a) == 1:
                pos = pos + ab
        return pos


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
        [2, 2, 0, 1, 1],
        [2, 2, 0, 1, 1]

    ])
    # from random import randint
    # table = np.array([[randint(1,1000) for _ in range(200)] for __ in range(10000)])
    broadcastVar = Configuration.sc.broadcast([1, 2, 3])
    is_consistent = AbstractionClass(table)
    A = is_consistent.my_get_abstraction_class(None)
    print A


    # t1 = datetime.datetime.now()
    # B = is_consistent.get_abstraction_class(4, None)
    # t2 = datetime.datetime.now()
    # print t2-t1
    #
    # t1 = datetime.datetime.now()
    # A = is_consistent.standalone_abstraction_class()
    # #print A
    # t2 = datetime.datetime.now()
    # print t2-t1
    #
    # print sorted(A)==sorted(B)

# TODO Testy
# TODO Dokumentacja
# TODO Refactoring code
# TODO cleaning code
