import time
from collections import Counter

import numpy as np
from sklearn import cross_validation

from reduct_feature_selection.commons.tables.eav import Eav
from settings import Configuration
from numpy.lib import recfunctions as rfn


class SimpleExtractor(object):

    def __init__(self, table, attrs_list, dec, cuts_limit_ratio):
        self.table =table
        self.attrs_list = attrs_list
        self.dec = dec
        self.cuts_limit_ratio = cuts_limit_ratio

    def generate_data(self, rows, cols):
        table = []
        for i in range(rows):
            mu = np.random.uniform(10, 100)
            sigma = np.random.uniform(10, 100)
            table.append(tuple(np.random.normal(mu, sigma, cols)))
        formats = [(str(i), float) for i in range(cols)]
        return np.array(table, dtype=formats)

    def extract_cuts_column(self, column):
        dec_fqs = Counter()
        column = list(column)
        for elem in column:
            dec_fqs[self.dec[elem[0]]] += 1
        dec_fqs_disc = {k: (0, v) for k, v in dec_fqs.iteritems()}
        gini_index = 0
        indexes = []
        # TODO: fix it to disc measure
        for elem in column[:-1]:
            dec = self.dec[elem[0]]
            old_gini = dec_fqs_disc[dec][0] * dec_fqs_disc[dec][1]
            k, v = dec_fqs_disc[dec]
            dec_fqs_disc[dec] = (k + 1, v - 1)
            new_gini = dec_fqs_disc[dec][0] * dec_fqs_disc[dec][1]
            gini_index += (new_gini - old_gini)
            indexes.append((gini_index, elem[1], elem[2]))
        indexes = sorted(indexes, key=lambda x: x[0])
        max_ind = indexes[-1][0]
        good_cuts = [(indexes[0][1], indexes[0][2])]
        i = 1
        while indexes[i][0] / float(max_ind) < self.cuts_limit_ratio:
            good_cuts.append((indexes[i][1], indexes[i][2]))
            i += 1
        good_cuts = list(set(good_cuts))
        column = sorted(column, key=lambda x: x[0])
        for cut in good_cuts:
            yield (cut[0], cut[1], [int(x[2] > cut[1]) for x in column])

    def add_to_table(self, new_col_set):
        new_table = np.copy(self.table)
        col_names = [col[0] + "_gt_" + str(col[1]) for col in new_col_set]
        data_set = [col[2] for col in new_col_set]
        return rfn.append_fields(new_table, names=col_names, data=data_set, usemask=False)

    def extract(self, par=False):
        div_list = lambda lst, sz: [lst[i:i + sz] for i in range(0, len(lst), sz)]
        eav = Eav(Eav.convert_to_eav(self.table[self.attrs_list]))
        eav.sort()
        eav_table = eav.eav
        if par:
            eav_rdd_part = Configuration.sc.parallelize(eav_table, len(self.attrs_list))
            new_col_set = eav_rdd_part.mapPartitions(self.extract_cuts_column).collect()
        else:
            eav_div = div_list(eav_table, len(eav_table) / len(self.attrs_list))
            new_col_set = reduce(lambda x, y: x + y,
                              map(lambda col: list(self.extract_cuts_column(col)), eav_div))
        return self.add_to_table(new_col_set)

    def eval(self, clf):
        res = cross_validation.cross_val_score(clf, self.table, self.dec, cv=10, scoring='f1_weighted')
        return res

    def compare_eval(self, clf):
        res_ndisc = self.eval(clf)
        res_disc = self.eval(self.extract(), clf)
        return res_ndisc, res_disc

    def compare_time(self):
        start_par = time.time()
        self.extract(par=True)
        time_par = time.time() - start_par
        start_npar = time.time()
        self.extract()
        time_npar = time.time() - start_npar
        return time_par, time_npar

if __name__ == "__main__":
    table = np.array([(0, 1, 7), (4, 5, 8), (1, 2, 3), (3, 8, 9),
                      (0, 1, 7), (4, 5, 8), (1, 2, 3), (3, 8, 9)],
                     dtype=[('x', int), ('y', float), ('z', float)])
    dec = [0, 1, 0, 1, 0, 1, 0, 1]
    attrs_list = ['x', 'y']
    discretizer = SimpleExtractor(table, attrs_list, dec, 0.1)
    # table = discretizer.extract(table, attrs_list, par=True)
    # print table
    # print table.dtype.names
    print discretizer.compare_time()
    # disc_table = discretizer.discretize(table, ['y', 'z'], par=False)
    # print table
    # print table.dtype.names
    # print disc_table
    # print disc_table.dtype.names
    #print discretizer.compare_time(table, ['y', 'z'])
    #table2 = discretizer.generate_data(100, 50)
    #print discretizer.compare_time(table2, attrs_list)


