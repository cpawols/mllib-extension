from reduct_feature_selection.commons.tables.eav import Eav
from reduct_feature_selection.discretization.simple_discretizer import SimpleDiscretizer
from collections import Counter

import math

import numpy as np

from pyspark import SparkContext, SparkConf

from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn import tree
from sklearn.naive_bayes import GaussianNB


class EntropyDiscretizer(SimpleDiscretizer):

    def __init__(self, table, attrs_list, dec):
        super(EntropyDiscretizer, self).__init__(table, attrs_list, dec)
        self.MAX_ENT = 1000000

    def entropy(self, objects):
        dec_fqs = Counter(self.dec[elem[0]] for elem in objects)
        probs = [x / float(len(objects)) for x in dec_fqs.values()]
        entropy = sum(map(lambda x: -x * math.log(x, 2), probs))
        return entropy

    @staticmethod
    def entropy_from_distribution(n, dec_fqs):
        probs = [x / float(n) for x in dec_fqs.values()]
        entropy = sum(map(lambda x: -x * math.log(x, 2), probs))
        return entropy

    def stop_criterion(self, objects, min_cut):
        n = len(objects)
        objects_first_half = objects[min_cut:]
        objects_second_half = objects[:min_cut]
        ent_objects = self.entropy(objects)
        ent_objects_first_half = self.entropy(objects_first_half)
        ent_objects_second_half = self.entropy(objects_second_half)

        info_gain = ent_objects - (min_cut / float(n)) * ent_objects_first_half + \
                    ((n - min_cut) / float(n)) * ent_objects_second_half

        dec_fqs_objects = Counter()
        dec_fqs_first_half = Counter()
        dec_fqs_seconf_half = Counter()
        for i, elem in enumerate(objects):
            if i < min_cut:
                dec_fqs_first_half[self.dec[elem[0]]] += 1
            else:
                dec_fqs_seconf_half[self.dec[elem[0]]] += 1
            dec_fqs_objects[self.dec[elem[0]]] += 1

        r = len(dec_fqs_objects.keys())
        r1 = len(dec_fqs_first_half.keys())
        r2 = len(dec_fqs_seconf_half.keys())
        ent_weighted_gain = (r * ent_objects - r1 * ent_objects_first_half - r2 * ent_objects_second_half)
        stop = math.log(n - 1, 2) / float(n) + \
               (math.log(math.pow(3, r) - 2, 2) - ent_weighted_gain) / float(n)

        return info_gain < stop

    def find_cut_lin(self, begin, objects):
        n = len(objects)
        if n > 1:
            min_cut_ent = self.MAX_ENT
            min_cut = 1
            cut = 1
            dec_fqs = Counter(self.dec[elem[0]] for elem in objects)
            dec_fqs_disc = {k: (0, v) for k, v in dec_fqs.items()}
            all_entropy = self.entropy_from_distribution(len(objects), dec_fqs)
            ent_2 = all_entropy
            ent_1 = 0
            for elem in objects[:-1]:
                dec = self.dec[elem[0]]
                f_1, f_2 = dec_fqs_disc[dec]
                p_1 = f_1 / float(n) if f_1 > 0 else 1
                p_2 = f_2 / float(n) if f_2 > 0 else 1
                ent_1_old = - p_1 * math.log(p_1, 2)
                ent_2_old = - p_2 * math.log(p_2, 2)
                dec_fqs_disc[dec] = (f_1 + 1, f_2 - 1)
                p_1 = (f_1 + 1) / float(n) if f_1 + 1 > 0 else 1
                p_2 = (f_2 - 1) / float(n) if f_2 - 1 > 0 else 1
                ent_1_new = - p_1 * math.log(p_1, 2)
                ent_2_new = - p_2 * math.log(p_2, 2)
                ent_1 += (ent_1_new - ent_1_old)
                ent_2 += (ent_2_new - ent_2_old)
                cut_ent = cut / float(n) * ent_1 + (n - cut) / float(n) * ent_2
                if cut_ent < min_cut_ent:
                    min_cut_ent = cut_ent
                    min_cut = cut
                cut += 1
            if not self.stop_criterion(objects, min_cut):
                yield begin + min_cut

                for cut in self.find_cut_lin(begin, objects[:min_cut]):
                    yield cut
                for cut in self.find_cut_lin(begin + min_cut, objects[min_cut:]):
                    yield cut

    def find_cut_quadr(self, begin, objects):
        n = len(objects)
        if n > 1:
            cuts = range(1, n)
            min_cut_ent = self.MAX_ENT
            min_cut = 1
            for cut in cuts:
                cut_ent = cut / float(n) * self.entropy(objects[:cut]) + \
                          (n - cut) / float(n) * self.entropy(objects[cut:])
                if cut_ent < min_cut_ent:
                    min_cut_ent = cut_ent
                    min_cut = cut
            if not self.stop_criterion(objects, min_cut):
                yield begin + min_cut

                for cut in self.find_cut_quadr(begin, objects[:min_cut]):
                    yield cut
                for cut in self.find_cut_quadr(begin + min_cut, objects[min_cut:]):
                    yield cut

    def discretize_column(self, column):
        column = list(column)
        dis = 0
        cuts_set = sorted(self.find_cut_lin(0, column))
        cur_cut = cuts_set[0]

        for i, elem in enumerate(column):
            if cur_cut == i:
                dis += 1
            if len(cuts_set) > dis:
                cur_cut = cuts_set[dis]

            yield (elem[0], elem[1], dis)


if __name__ == "__main__":
    # conf = (SparkConf().setMaster("spark://localhost:7077").setAppName("entropy"))
    # sc = SparkContext(conf=conf)
    # # table = np.array([(1, 7), (1, 8), (1, 3), (1, 9), (1, 1), (1, 2), (1, 5), (1, 10)],
    # #                  dtype=[('y', float), ('z', float)])
    # # U = [(0, 'x', 5), (1, 'x', 8), (2, 'x', 4), (3, 'x', 5), (4, 'x', 2), (5, 'x', 3)]
    # # dec = [1, 1, 2, 0, 0, 2]
    # # discretizer = EntropyDiscretizer(dec, 3)
    #
    # table = np.array([(1, 1),
    #                   (1, 2),
    #                   (1, 3),
    #                   (1, 4),
    #                   (1, 5),
    #                   (1, 6),
    #                   (1, 7),
    #                   (1, 8),
    #                   (1, 9),
    #                   (1, 10),
    #                   (1, 11),
    #                   (1, 12),
    #                   (1, 13),
    #                   (1, 14),
    #                   (1, 15),
    #                   (1, 16),
    #                   (1, 17),
    #                   (1, 18),
    #                   (1, 19),
    #                   (1, 20)],
    #                  dtype=[('y', float), ('z', float)])
    # dec_1 = [1,1,1,1,1,0,0,0,0,0,2,2,2,2,2,3,3,3,3,3]
    # dec_2 = [0,1,0,1,0,1,0,1,0,1,2,2,2,2,2,3,3,3,3,3]
    # attrs_list = ['C1', 'C2']
    # discretizer_1 = EntropyDiscretizer(table, attrs_list, dec_1)
    # discretizer_2 = EntropyDiscretizer(table, attrs_list, dec_2)
    # #print discretizer.discretize(table, attrs_list, par=True)
    # print discretizer_1.discretize()
    # print discretizer_2.discretize()

    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris['data'], iris['target'], test_size=0.2, random_state=42)
    iris_data = Eav.convert_to_proper_format(iris['data'])
    discretizer = EntropyDiscretizer(iris_data, ['C1', 'C2', 'C3'], iris['target'])
    table1 = discretizer.discretize()
    #table2 = discretizer.discretize()
    # clf = GaussianNB()
    # print discretizer.compare_eval(clf)
    # print "discretized"
    print table1
    #print table2
