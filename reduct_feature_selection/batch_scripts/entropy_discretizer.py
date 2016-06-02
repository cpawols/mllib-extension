import random
from collections import Counter
import time
import math
import operator
import pickle

import numpy as np
from numpy import genfromtxt

from pyspark import SparkContext, SparkConf
from sklearn import tree

from sklearn.datasets import load_iris


def timeit(method):

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print '%r (%r, %r) %2.5f sec' % \
              (method.__name__, args, kw, te - ts)
        return result

    return timed


class Eav:
    def __init__(self, eav):
        self.eav = eav
        self.dec = {}
        self._obj_index = self.update_index(0)
        self._attr_index = self.update_index(1)

    @classmethod
    def from_array(cls, array):
        return cls(cls.convert_to_eav(array))

    @property
    def dec(self):
        return self.dec

    @dec.setter
    def dec(self, value):
        self.dec = value

    # TODO: convert to format with dtype.names

    @staticmethod
    def convert_to_proper_format(array):
        ncols = len(array[0])
        formats = [('C' + str(i + 1), type(array[0][i])) for i in range(ncols)]
        new_array = map(lambda row: tuple(row), array)
        return np.array(new_array, dtype=formats)

    @staticmethod
    def convert_to_proper_array(array):
        new_array = map(lambda row: list(row), array)
        return np.array(new_array)

    @staticmethod
    def convert_to_eav(array):
        """
        this function convert numpy array to eav format (list of tuples (enitity, attribute, value))
        :param array: numpy array
        :return: list of eav tuples
        """
        if array.size:
            Eav.convert_to_proper_format(array)
            rows = range(array.shape[0])
            colnames = array.dtype.names
            list_of_eav = ([(r, c, array[c][r]) for c in colnames] for r in rows)
            return reduce(operator.add, list_of_eav)
        return []

    def convert_to_array(self):
        """
        this function convert eav format (list of tuples (entity, attribute, value)) to numpy array
        :return: numpy
        """
        if self.eav:
            rows_size = max([x[0] for x in self.eav])
            cols_size = len(self.eav) / (rows_size + 1)
            formats = sorted(list(set([(x[1], float) for x in self.eav])))
            array = np.array([tuple([0] * (cols_size))] * (rows_size + 1),
                             dtype=formats)
            for t in self.eav:
                array[t[0]][t[1]] = t[2]
            return array
        return np.array([])

    def update_index(self, attr_obj):
        objects = set([x[attr_obj] for x in self.eav])
        index = {x: [] for x in objects}
        for ind, obj in enumerate(self.eav):
            index[obj[attr_obj]].append(ind)
        return index

    def get_object(self, obj):
        return self._obj_index[obj]

    def get_attribute(self, attr):
        return self._attr_index[attr]

    def get_obj_attrs(self, obj):
        return [self.eav[ind][2] for ind in self._obj_index[obj]]

    def get_obj_count(self):
        objects = [x[0] for x in self.eav]
        return len(set(objects))

    def get_attr_count(self):
        objects = [x[1] for x in self.eav]
        return len(set(objects))

    def sort(self, sc=None):
        if sc is not None:
            eav_rdd = sc.parallelize(self.eav)
            self.eav = eav_rdd.map(lambda x: ((x[1], x[2], x[0]), 1)).sortByKey()\
                .map(lambda (k, v): (k[2], k[0], k[1])).collect()
        else:
            self.eav = sorted(self.eav, key=lambda x: (x[1], x[2], x[0]))
        self.update_index(0)
        self.update_index(1)

    @staticmethod
    def _compare(iterator):
        yield sorted(iterator, key=lambda x: (x[1], x[2], x[0]))

    def merge_sort(self, sc):
        num_chunks = 10
        eav_rdd_part = sc.parallelize(self.eav, num_chunks)
        self.eav = eav_rdd_part.mapPartitions(Eav._compare)\
            .reduce(lambda x, y: sorted(x + y, key=lambda x: (x[1], x[2], x[0])))
        self.update_index(0)
        self.update_index(1)

    def is_consistent(self):
        same_dec = [(ob1, ob2) for ob1, dec1 in self.dec.iteritems() for ob2, dec2 in self.dec.iteritems()
                    if not dec1 == dec2 and not ob1 == ob2]
        for (ob1, ob2) in same_dec:
            if self.get_obj_attrs(ob1) == self.get_obj_attrs(ob2):
                return False
        return True


class SimpleDiscretizer(object):

    def __init__(self, table, attrs_list, dec, group_count=5):
        self.dec = dec
        self.table = Eav.convert_to_proper_format(table)
        self.attrs_list = attrs_list
        self.group_count = group_count

    def set_dec(self, dec):
        self.dec = dec

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
            yield (row[0], row[1], ind / self.group_count)

    @timeit
    def discretize(self, sc=None):
        div_list = lambda lst, size: [lst[i:i + size] for i in range(0, len(lst), size)]
        eav = Eav(Eav.convert_to_eav(self.table[self.attrs_list]))
        eav.sort(sc)
        eav_table = eav.eav
        if sc is not None:
            eav_rdd_part = sc.parallelize(eav_table, len(self.attrs_list))
            eav_disc = eav_rdd_part.mapPartitions(self.discretize_column).collect()
        else:
            eav_div = div_list(eav_table, len(eav_table) / len(self.attrs_list))
            eav_disc = reduce(lambda x, y: x + y,
                              map(lambda col: list(self.discretize_column(col)), eav_div))
        disc_table = Eav(eav_disc).convert_to_array()
        return Eav.convert_to_proper_array(disc_table)

    def eval(self, table, clf):
        res = cross_validation.cross_val_score(clf, table, self.dec, cv=10, scoring='f1_weighted')
        return res

    def compare_eval(self, clf, sc=None):
        res_ndisc = self.eval(Eav.convert_to_proper_array(self.table), clf)
        res_disc = self.eval(self.discretize(sc), clf)
        return res_ndisc, res_disc

    def compare_time(self, sc=None):
        self.discretize(sc)
        self.discretize()


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


class OneRDiscretizer(SimpleDiscretizer):

    def __init__(self, table, attrs_list, dec, min_elem=6):
        super(OneRDiscretizer, self).__init__(table, attrs_list, dec)
        self.min_elem = min_elem

    def discretize_column(self, column):
        dis = 0
        dis_elems = 0
        dec_fqs = Counter()
        for elem in column:

            if dis_elems > self.min_elem and dec_fqs.most_common(1)[0][1] > dis_elems / 2:
                dis_elems = 0
                dis += 1
                dec_fqs = Counter()
            dec_fqs[self.dec[elem[0]]] += 1
            dis_elems += 1

            yield (elem[0], elem[1], dis)


def cross_val_score(X, y, k=10):
    div_list = lambda lst, size: [lst[i:i + size] for i in range(0, len(lst), size)]
    l = range(len(y))
    random.shuffle(l)
    folds = div_list(l, int(len(y) / k))
    accuracy = 0
    i = 0
    for test_ids in folds:
        train_ids = [x for x in range(0, n) if x not in test_ids]
        X_train = X[train_ids, ]
        X_test = X[test_ids, ]
        y_train = y[train_ids]
        y_test = y[test_ids]

        clf = tree.DecisionTreeClassifier()
        clf.fit(X_train, y_train)
        res = clf.predict(X_test)
        accuracy += (sum(res == y_test) / float(len(res)))
        i += 1
        print "performed " + str(i) + " cv, result: " + str((sum(res == y_test) / float(len(res))))

    return accuracy / float(k)

if __name__ == "__main__":
    conf = (SparkConf().setMaster("spark://green07:7077").setAppName("entropy"))
    sc = SparkContext(conf=conf)

    iris = load_iris()

    data = load_iris()
    X = data['data']
    y = data['target']
    n = len(y)

    data = genfromtxt("/home/students/mat/k/kr319379/Downloads/marrData.csv", delimiter=",")
    X = data[:,:-1]
    y = data[:,-1]

    X_ex = Eav.convert_to_proper_format(X)

    discretizer_ent = EntropyDiscretizer(X_ex, list(X_ex.dtype.names), y)
    #discretizer_r = OneRDiscretizer(X_ex, list(X_ex.dtype.names), y)
    X_ent = discretizer_ent.discretize()
    #X_r = discretizer_r.discretize()
    print "standard"
    print cross_val_score(X, y, 5)
    print "discretized entropy"
    print cross_val_score(X_ent, y, 5)
    with open("/home/students/mat/k/kr319379/Downloads/discr", "w") as f:
        pickle.dump(X_ent, file=f)
    # print "discretized r"
    # print cross_val_score(X_r, y, 5)
