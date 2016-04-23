import time

import numpy as np
from sklearn import cross_validation

from reduct_feature_selection.commons.tables.eav import Eav
from pyspark import SparkContext, SparkConf


class SimpleDiscretizer(object):

    def __init__(self, dec, m):
        self.dec = dec
        self.m = m

    def generate_data(self, rows, cols):
        table = []
        for i in range(rows):
            mu = np.random.uniform(10, 100)
            sigma = np.random.uniform(10, 100)
            table.append(tuple(np.random.normal(mu, sigma, cols)))
        formats = [(str(i), float) for i in range(cols)]
        return np.array(table, dtype=formats)

    def discretize_column(self, column):
        for ind, row in enumerate(column):
            yield (row[0], row[1], ind / self.m)

    def discretize(self, table, attrs_list, sc=None):
        div_list = lambda lst, sz: [lst[i:i + sz] for i in range(0, len(lst), sz)]
        eav = Eav(Eav.convert_to_eav(table[attrs_list]))
        eav.sort(sc)
        eav_table = eav.eav
        if sc is not None:
            eav_rdd_part = sc.parallelize(eav_table, len(attrs_list))
            eav_disc = eav_rdd_part.mapPartitions(self.discretize_column).collect()
        else:
            eav_div = div_list(eav_table, len(eav_table) / len(attrs_list))
            eav_disc = reduce(lambda x, y: x + y,
                              map(lambda col: list(self.discretize_column(col)), eav_div))
        disc_table = Eav(eav_disc).convert_to_array()
        return disc_table

    def eval(self, table, clf):
        res = cross_validation.cross_val_score(clf, table, self.dec, cv=10, scoring='f1_weighted')
        return res

    def compare_eval(self, table, attrs_list, clf, sc=None):
        res_ndisc = self.eval(table, clf)
        res_disc = self.eval(self.discretize(table, attrs_list, sc), clf)
        return res_ndisc, res_disc

    def compare_time(self, table, attrs_list, sc=None):
        start_par = time.time()
        self.discretize(table, attrs_list, sc)
        time_par = time.time() - start_par
        start_npar = time.time()
        self.discretize(table, attrs_list)
        time_npar = time.time() - start_npar
        return time_par, time_npar

if __name__ == "__main__":
    conf = (SparkConf().setMaster("spark://localhost:7077").setAppName("entropy"))
    sc = SparkContext(conf=conf)
    table = np.array([(0, 1, 7), (4, 5, 8), (1, 2, 3), (3, 8, 9),
                      (0, 1, 7), (4, 5, 8), (1, 2, 3), (3, 8, 9)],
                     dtype=[('x', int), ('y', float), ('z', float)])
    dec = [0, 1, 0, 1, 0, 1, 0, 1]
    attrs_list = [str(i) for i in range(50)]
    discretizer = SimpleDiscretizer(dec, 10)
    # disc_table = discretizer.discretize(table, ['y', 'z'], par=False)
    # print table
    # print table.dtype.names
    # print disc_table
    # print disc_table.dtype.names
    #print discretizer.compare_time(table, ['y', 'z'])
    table2 = discretizer.generate_data(100, 50)
    print discretizer.compare_time(table2, attrs_list, sc)


