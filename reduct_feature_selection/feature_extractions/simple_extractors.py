import time
from collections import Counter

import numpy as np
from sklearn import cross_validation

from reduct_feature_selection.commons.tables.eav import Eav
from reduct_feature_selection.feature_extractions.disc_measure_calculator import DiscMeasureCalculator
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

    def extract_cuts_column2(self, column):
        column = list(column)
        dec_fqs_disc = DiscMeasureCalculator.prepare_hist(self.dec)
        act_left_sum = 0
        act_right_sum = len(column)
        act_award = 0

        cuts = []
        for elem in column[:-1]:
            dec = self.dec[elem[0]]
            act_left_sum, act_right_sum, act_award = \
                DiscMeasureCalculator.update_award(dec, act_left_sum, act_right_sum, dec_fqs_disc, act_award)
            cuts.append((act_award, elem[1], elem[2]))

        good_cuts = self.select_cuts(cuts)
        column = sorted(column, key=lambda x: x[0])
        for cut in good_cuts:
            yield (cut[0], cut[1], [int(x[2] > cut[1]) for x in column])

    # TODO: add better select cuts strategies
    def select_cuts(self, cuts):
        cuts = sorted(cuts, key=lambda x: x[0], reverse=True)
        max_ind = cuts[-1][0]
        good_cuts = [(cuts[0][1], cuts[0][2])]
        i = 1
        while cuts[i][0] / float(max_ind) < self.cuts_limit_ratio:
            good_cuts.append((cuts[i][1], cuts[i][2]))
            i += 1
        return list(set(good_cuts))

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
            new_col_set = eav_rdd_part.mapPartitions(self.extract_cuts_column2).collect()
        else:
            eav_div = div_list(eav_table, len(eav_table) / len(self.attrs_list))
            new_col_set = reduce(lambda x, y: x + y,
                              map(lambda col: list(self.extract_cuts_column2(col)), eav_div))
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
    table = np.array([(0, 1, 7),
                      (4, 5, 8),
                      (1, 2, 3),
                      (3, 8, 9),
                      (0, 1, 7),
                      (4, 5, 8),
                      (1, 2, 3),
                      (3, 8, 9)],
                     dtype=[('x', int), ('y', float), ('z', float)])
    dec = [0, 1, 0, 1, 0, 1, 0, 1]
    attrs_list = ['x', 'y']
    discretizer = SimpleExtractor(table, attrs_list, dec, 0.1)
    table = discretizer.extract(par=False)
    print table
    print table.dtype.names
    #print discretizer.compare_time()
    # disc_table = discretizer.discretize(table, ['y', 'z'], par=False)
    # print table
    # print table.dtype.names
    # print disc_table
    # print disc_table.dtype.names
    #print discretizer.compare_time(table, ['y', 'z'])
    #table2 = discretizer.generate_data(100, 50)
    #print discretizer.compare_time(table2, attrs_list)


