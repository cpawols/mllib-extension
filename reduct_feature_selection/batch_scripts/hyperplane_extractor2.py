from collections import Counter
from operator import add
from random import randrange
import random
import copy
import math
import numpy as np
import time
import pickle

import scipy.io as sio
from numpy import genfromtxt

import operator
from scipy.spatial.distance import squareform, pdist
from sklearn import tree
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.datasets import load_iris, load_digits
from sklearn.metrics import accuracy_score
from numpy.lib import recfunctions as rfn

from pyspark import SparkContext, SparkConf


# from sklearn.svm import LinearSVC


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print '%r %2.5f sec' % \
              (method.__name__, te - ts)
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
            array = np.array([tuple([0] * cols_size)] * (rows_size + 1),
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
            eav_rdd = sc.parallelize(self.eav, 1000)
            self.eav = eav_rdd.map(lambda x: ((x[1], x[2], x[0]), 1)).sortByKey() \
                .map(lambda (k, v): (k[2], k[0], k[1])).collect()
        else:
            self.eav = sorted(self.eav, key=lambda x: (x[1], x[2], x[0]))
        self.update_index(0)
        self.update_index(1)

    @staticmethod
    def _compare(iterator):
        yield sorted(iterator, key=lambda x: (x[1], x[2], x[0]))

    def merge_sort(self, sc):
        num_chunks = 1000
        eav_rdd_part = sc.parallelize(self.eav, num_chunks)
        self.eav = eav_rdd_part.mapPartitions(Eav._compare) \
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


class ConsistentChecker(object):
    # TODO: add non paralell reduce by key
    @staticmethod
    def reduce_by_rows(extracted_table, dec, sc=None):
        if extracted_table:
            if sc is not None:
                table_list = [(tuple(map(lambda x: x[i], extracted_table)), (i, d)) for i, d in enumerate(dec)]

                table_list_rdd = sc.parallelize(table_list, 1000)
                return table_list_rdd.reduceByKey(add).collect()
            else:
                table_list = [tuple(map(lambda x: x[i], extracted_table)) for i in range(len(dec))]
                unique_rows = list(set(table_list))
                rows_dict = {row: [] for row in unique_rows}
                for i, d in enumerate(dec):
                    rows_dict[table_list[i]] += (i, d)
                return rows_dict.items()

        return []

    @staticmethod
    def is_consistent(extracted_table, dec, sc=None):
        '''
        :param extracted_table: list of rows
        :param dec: list of decisions
        :return: true if table is consistent else false
        '''

        row_groups = ConsistentChecker.reduce_by_rows(extracted_table, dec, sc)

        for row_group in row_groups:
            group_decisions = [row_group[1][i] for i in range(1, len(row_group[1]), 2)]
            if len(set(group_decisions)) > 1:
                return False
        return True

    @staticmethod
    def count_unconsistent_groups(extracted_table, dec, sc=None):
        '''
        :param extracted_table: list of rows
        :param dec: list of decisions
        :return: list of lists of unconistent subtables (indices of rows)
        '''

        row_groups = ConsistentChecker.reduce_by_rows(extracted_table, dec, sc)

        if row_groups:
            groups = []
            for row_group in row_groups:
                group_decisions = [row_group[1][i] for i in range(1, len(row_group[1]), 2)]
                if len(set(group_decisions)) > 1:
                    group = [row_group[1][i] for i in range(0, len(row_group[1]), 2)]
                    groups.append(group)
            return groups
        return [range(len(dec))]


class BaseDoubtfulPointsStrategy(object):
    def __init__(self, table, dec):
        self.table = table
        self.dec = dec

    def extract_points_matrix(self, objects):
        return np.array([list(self.table[obj,]) for obj in objects])

    def decision(self, objects):
        dec_set = set([self.dec[obj] for obj in objects])
        if len(dec_set) == 1:
            return self.dec[objects[0]]
        return None


class MostDecisionStrategy(BaseDoubtfulPointsStrategy):
    def __init__(self, table, dec, max_ratio):
        super(MostDecisionStrategy, self).__init__(table, dec)
        self.max_ratio = max_ratio

    def decision(self, objects):
        ob_dec = [self.dec[obj] for obj in objects]
        if len(set(ob_dec)) == 1:
            return self.dec[objects[0]]
        dec_hist = Counter(ob_dec)
        most_common_len = dec_hist.most_common(1)[0][1]
        if most_common_len / float(len(ob_dec)) > self.max_ratio or len(ob_dec) < 3:
            return dec_hist.most_common(1)[0][0]
        return None


class MinDistDoubtfulPointsStrategy(BaseDoubtfulPointsStrategy):
    def __init__(self, table, dec, min_dist):
        super(MinDistDoubtfulPointsStrategy, self).__init__(table, dec)
        self.min_dist = min_dist

    def decision(self, objects):
        ob_dec = [self.dec[obj] for obj in objects]
        if len(set(ob_dec)) == 1:
            return self.dec[objects[0]]

        point_matrix = self.extract_points_matrix(objects)
        sq_form = squareform(pdist(point_matrix))
        max_dist = np.max(sq_form)

        if max_dist <= self.min_dist:
            return Counter(ob_dec).most_common(1)[0][0]
        return None


class GeneticSearch(object):
    def __init__(self, k, dec, table, projection_axis, unconsistent_groups,
                 b=50,
                 first_generation_size=200,
                 population_size=50,
                 max_iter=20,
                 cross_chance=0.4,
                 mutation_chance=0.01,
                 stop_treshold=5):
        self.b = b
        self.k = k
        self.stop_treshold = stop_treshold
        self.first_generation_size = first_generation_size
        self.population_size = population_size if population_size % 2 == 0 else population_size + 1
        self.max_iter = max_iter
        self.cross_chance = cross_chance
        self.mutation_chance = mutation_chance
        self.dec = dec
        self.table = table
        self.table_as_matrix = np.array([[row] for row in table]) if k == 1 else np.array([list(row) for row in table])
        self.projection_axis = projection_axis
        self.unconsistent_groups = unconsistent_groups

    def init_generation(self):
        max_vec = math.pow(2, self.b)
        return [[randrange(-max_vec, max_vec) for _ in range(self.k)]
                for _ in range(self.first_generation_size)]

    @timeit
    def _get_subtable(self, table, objects):
        return np.array([table[obj] for obj in objects], dtype=table.dtype)

    @timeit
    def _get_subtable_as_matrix(self, table, objects):
        return np.array([table[obj] for obj in objects])

    # @timeit
    def count_projections(self, individual, objects):
        '''
        :param individual: hyperplane
        :param objects: objects projection to axis
        :return: list of tuples (objects, projection) projection of object to axis
        '''
        table = self.table_as_matrix[objects, :]
        proj = np.array([self.projection_axis[obj] for obj in objects])
        return zip(objects, proj - table.dot(individual))

    # TODO: add tests and docs
    def count_award(self, individual):
        if len(self.unconsistent_groups) == 1:
            return self.count_local_award(individual)
        return self.count_global_award(individual)

    def count_global_award(self, individual):
        valid_objects = reduce(add, self.unconsistent_groups)
        projections = self.count_projections(individual, valid_objects)
        objects = sorted(projections, key=lambda x: x[1])

        object_group_map = {}
        group_fqs_map = {}
        act_left_sum = {}
        act_right_sum = {}

        for ind, group in enumerate(self.unconsistent_groups):
            group_decisions = []
            for obj in group:
                group_decisions.append(self.dec[obj])
                object_group_map[obj] = ind
            act_left_sum[ind] = 0
            act_right_sum[ind] = len(group_decisions)
            group_fqs_map[ind] = DiscMeasureCalculator.prepare_hist(group_decisions)

        act_award = 0
        max_award = 0
        for obj, proj in objects:
            group_id = object_group_map[obj]
            dec = self.dec[obj]
            act_left_sum[group_id], act_right_sum[group_id], act_award = \
                DiscMeasureCalculator.update_award(dec, act_left_sum[group_id], act_right_sum[group_id],
                                                   group_fqs_map[group_id], act_award)
            if act_award > max_award:
                max_award = act_award
                good_proj = proj

        return max_award, individual, good_proj

    # @timeit
    def count_local_award(self, individual):
        valid_objects = self.unconsistent_groups[0]
        projections = self.count_projections(individual, valid_objects)
        objects = sorted(projections, key=lambda x: x[1])
        decisions = [self.dec[obj] for obj in valid_objects]

        act_left_sum = 0
        act_right_sum = len(decisions)
        dec_fqs_disc = DiscMeasureCalculator.prepare_hist(decisions)

        act_award = 0
        max_award = 0
        for obj, proj in objects:
            dec = self.dec[obj]
            act_left_sum, act_right_sum, act_award = \
                DiscMeasureCalculator.update_award(dec, act_left_sum, act_right_sum, dec_fqs_disc, act_award)
            if act_award > max_award:
                max_award = act_award
                good_proj = proj

        return max_award, individual, good_proj

    def select_best_individuals(self, population_awards):

        population = sorted(population_awards, key=lambda x: x[0], reverse=True)
        best_ind = population[:self.population_size]

        return best_ind

    def count_new_generation(self, population):
        '''
        :param population: list of individuals, eg list of numbers represeting vectors
        :return: new generation counting by crossover and mutation (list of list)
        '''

        rands = random.sample(range(len(population)), len(population))
        pairs = [(rands[i], rands[i + 1]) for i in range(0, len(population), 2)]
        new_generation = []

        for pair in pairs:

            ind1 = population[pair[0]]
            ind2 = population[pair[1]]

            if random.random() < self.cross_chance:
                el = random.choice(range(len(ind1)))
                pom = ind1[el]
                ind1[el] = ind2[el]
                ind2[el] = pom

            if random.random() < self.mutation_chance:
                el = random.choice(range(len(ind1)))
                new_vec = randrange(-math.pow(2, self.b), math.pow(2, self.b))
                ind1[el] = new_vec
            if random.random() < self.mutation_chance:
                el = random.choice(range(len(ind1)))
                new_vec = randrange(-math.pow(2, self.b), math.pow(2, self.b))
                ind2[el] = new_vec

            new_generation.append(ind1)
            new_generation.append(ind2)

        return new_generation

    def count_award_for_chunk(self, population):
        for individual in population:
            yield self.count_award(individual)

    def _eliminate_duplicates(self, population):
        sorted_population = sorted(population)
        current_ind = sorted_population[0]
        for ind in sorted_population[1:]:
            if ind == current_ind:
                sorted_population.remove(ind)
            else:
                current_ind = ind
        return sorted_population

    # TODO: add stop criterion
    @timeit
    def genetic_search(self, sc=None):

        # print "--------------------init population-------------------------------------------"
        population = self.init_generation()

        current_best_award = 0
        the_same_awards = 0
        for i in range(self.max_iter):
            print "-----------------------------performing " + str(i) + " generation---------------"
            if sc is not None:
                rdd_population = sc.parallelize(population, self.population_size * 10)
                awards = rdd_population.mapPartitions(self.count_award_for_chunk).collect()
            else:
                awards = map(self.count_award, population)

            population_awards = self.select_best_individuals(awards)

            best_individual = population_awards[0]

            if best_individual[0] > current_best_award:
                current_best_award = best_individual[0]
                the_same_awards = 0
            else:
                the_same_awards += 1

            population = map(lambda x: x[1], population_awards)

            new_generation = self.count_new_generation(copy.deepcopy(population))

            # print sorted(new_generation) == sorted(population)

            population = new_generation + population

            # set_population = self._eliminate_duplicates(population)
            # sorted_population = sorted(population)
            # print "new generation ratio"
            # print float(len(set_population)) / float(len(sorted_population))
            if the_same_awards > self.stop_treshold:
                break

        return best_individual


class DecisionTree:
    def __init__(self, value, left, right):
        self.value = value
        self.left = left
        self.right = right

    def is_leaf(self):
        if type(self.value) == list or type(self.value) == tuple:
            return False
        return True

    def _goto_right_son(self, object, svm):
        if svm:
            return np.dot(object, self.value[1]) + self.value[0] > 0
        else:
            nr_attr = int(self.value[0][1:]) - 1
            result = object[nr_attr]
            other = object[:nr_attr] + object[(nr_attr + 1):]
            for i, elem in enumerate(other):
                result -= self.value[1][1][i] * elem
            return result > self.value[1][2]

    def predict(self, object, svm=False):
        if self.is_leaf():
            return self.value
        if self._goto_right_son(object, svm):
            return self.right.predict(object, svm)
        return self.left.predict(object, svm)

    def predict_list(self, dataset, svm=False):
        return map(lambda r: self.predict(list(r), svm), dataset)

    def print_tree(self):
        print "value:"
        print self.value
        if not self.is_leaf():
            self.left.print_tree()
            self.right.print_tree()


class DiscMeasureCalculator(object):
    @staticmethod
    def prepare_hist(decisions):
        return {k: (0, v) for k, v in Counter(decisions).iteritems()}

    @staticmethod
    def update_award(dec, act_left_sum, act_right_sum, dec_fqs_disc, act_award):
        act_award += (act_right_sum - act_left_sum - 1) - \
                     (dec_fqs_disc[dec][1] - dec_fqs_disc[dec][0] - 1)
        act_left_sum += 1
        act_right_sum -= 1
        dec_fqs_disc[dec] = (dec_fqs_disc[dec][0] + 1,
                             dec_fqs_disc[dec][1] - 1)
        return act_left_sum, act_right_sum, act_award


class SimpleExtractor(object):
    def __init__(self, table, attrs_list, dec, cuts_limit_ratio=0.1):
        '''
        :param table: decision table
        :param attrs_list: list of attrbibutes to process
        :param dec: decision column
        :param cuts_limit_ratio: ratio of good cut
        '''
        self.table = table
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

    def extract(self, sc=None):
        div_list = lambda lst, size: [lst[i:i + size] for i in range(0, len(lst), size)]
        eav = Eav(Eav.convert_to_eav(self.table[self.attrs_list]))
        eav.sort(sc)
        eav_table = eav.eav
        if sc is not None:
            eav_rdd_part = sc.parallelize(eav_table, len(self.attrs_list))
            new_col_set = eav_rdd_part.mapPartitions(self.extract_cuts_column).collect()
        else:
            eav_div = div_list(eav_table, len(eav_table) / len(self.attrs_list))
            new_col_set = reduce(lambda x, y: x + y,
                                 map(lambda col: list(self.extract_cuts_column(col)), eav_div))
        return self.add_to_table(new_col_set)

    def eval(self, table, clf):
        res = cross_validation.cross_val_score(clf, table, self.dec, cv=10, scoring='f1_weighted')
        return res

    def compare_eval(self, clf, sc):
        res_ndisc = self.eval(self.table, clf)
        res_disc = self.eval(self.extract(sc), clf)
        return res_ndisc, res_disc

    def compare_time(self, sc):
        start_par = time.time()
        self.extract(sc)
        time_par = time.time() - start_par
        start_npar = time.time()
        self.extract()
        time_npar = time.time() - start_npar
        return time_par, time_npar


class HyperplaneExtractor(SimpleExtractor):
    def __init__(self, table, attrs_list, dec, dpoints_strategy, time_search_limit=1000,
                 b=50,
                 first_generation_size=200,
                 population_size=50,
                 max_iter=20,
                 cross_chance=0.4,
                 mutation_chance=0.01,
                 stop_treshold=5
                 ):
        '''
        :param table: decision table
        :param attrs_list: list of attrbibutes to process
        :param dec: decision column
        :param dpoints_strategy: strategy of classification of 'strange' points
        :param time_search_limit: time search limit in seconds for hyperplane searching
        '''
        super(HyperplaneExtractor, self).__init__(table, attrs_list, dec)
        self.dpoints_strategy = dpoints_strategy
        self.time_search_limit = time_search_limit
        self.b = b
        self.stop_treshold = stop_treshold
        self.first_generation_size = first_generation_size
        self.population_size = population_size
        self.max_iter = max_iter
        self.cross_chance = cross_chance
        self.mutation_chance = mutation_chance

    # TODO: add tests and docs
    def _count_objects_positions(self, best_hyperplane, objects=None):

        if objects is not None:
            table = np.array([self.table[i] for i in objects], dtype=self.table.dtype)
        else:
            table = self.table
        attr = best_hyperplane[0]
        axis_table = table[attr]
        other_attrs = [x for x in self.attrs_list if not x == attr]
        new_table = table[other_attrs]
        new_column = []
        for ind, row in enumerate(new_table):
            x = axis_table[ind]
            proj = x
            for i, r in enumerate(row):
                proj -= best_hyperplane[1][1][i] * r
            if proj > best_hyperplane[1][2]:
                new_column.append(1)
            else:
                new_column.append(0)

        return new_column

    def _search_best_hyperplane_for_projection(self, attr, unconsistent_groups, sc=None):

        # print attr
        # print self.table
        atr = list(attr)[0]
        # atr = attr
        projection_axis = self.table[atr]
        other_axes = [x for x in self.attrs_list if not x == atr]
        new_table = self.table[other_axes]

        gen_search = GeneticSearch(len(other_axes), self.dec, new_table,
                                   projection_axis, unconsistent_groups,
                                   max_iter=self.max_iter,
                                   stop_treshold=self.stop_treshold,
                                   first_generation_size=self.first_generation_size,
                                   population_size=self.population_size,
                                   cross_chance=self.cross_chance,
                                   mutation_chance=self.mutation_chance)
        cand_hyperplane = gen_search.genetic_search(sc)

        return atr, cand_hyperplane

    def _search_best_hyperplane(self, unconsistent_groups, sc=None):

        if False:
            hyperplanes = map(lambda x: self._search_best_hyperplane_for_projection(x, unconsistent_groups, sc),
                              self.attrs_list)
        else:
            n = len(self.attrs_list)
            atrs_list = range(n)
            rdd_attrs = sc.parallelize(self.attrs_list, n)
            # TODO: check if .map is better and less chunks
            hyperplanes = rdd_attrs.mapPartitions(
                lambda x: self._search_best_hyperplane_for_projection(x, unconsistent_groups)).collect()

            hyperp = [x for i, x in enumerate(hyperplanes) if i % 2 == 1]
            hyperplanes = zip(self.attrs_list, hyperp)

        return max(hyperplanes, key=lambda x: x[1][0])

    @timeit
    def extract(self, sc=None):
        extracted_table = []
        i = 0
        start = time.time()
        while True:
            unconsistent_groups = ConsistentChecker.count_unconsistent_groups(extracted_table, self.dec, sc)

            # print "--------------------unconsistent groups-----------------------------"
            # print unconsistent_groups

            unconsistent_groups = filter(lambda x: self.dpoints_strategy.decision(x) is None, unconsistent_groups)
            # print "--------------------unconsistent regs after clustering-----------------------------"
            # print unconsistent_groups
            i += 1
            time_spent = time.time() - start
            print "time spent: " + str(time_spent)
            if unconsistent_groups and time_spent < self.time_search_limit:
                print "-------------------performing " + str(i) + " iteration-----------------------"
                best_hyperplane = self._search_best_hyperplane(unconsistent_groups, sc)
                #print best_hyperplane
                extracted_table.append(self._count_objects_positions(best_hyperplane))
            else:
                break

        return np.transpose(np.array(extracted_table))

    @timeit
    def count_decision_tree(self, objects, sc=None, svm=False):
        decision = self.dpoints_strategy.decision(objects)
        if decision is not None:
            return DecisionTree(decision, 0, 0)

        if svm:
            X = [self.table[i] for i in objects]
            y = [self.dec[i] for i in objects]
            svm = LinearSVC()
            svm.fit(Eav.convert_to_proper_array(X), y)
            coefs = svm.coef_[0]
            inters = svm.intercept_[0]
            best_hyperplane = (inters, coefs)
            hyperplane_indicator = map(lambda r: (np.dot(list(r), coefs) + inters > 0), X)
            left_son_objects = map(lambda x: x[1],
                                   filter(lambda (i, x): not hyperplane_indicator[i], enumerate(objects)))
            right_son_objects = map(lambda x: x[1], filter(lambda (i, x): hyperplane_indicator[i], enumerate(objects)))
        else:
            best_hyperplane = self._search_best_hyperplane([objects], sc)
            hyperplane_indicator = self._count_objects_positions(best_hyperplane, objects)
            left_son_objects = map(lambda x: x[1],
                                   filter(lambda (i, x): hyperplane_indicator[i] == 0, enumerate(objects)))
            right_son_objects = map(lambda x: x[1],
                                    filter(lambda (i, x): hyperplane_indicator[i] == 1, enumerate(objects)))

        # print "podzial zbioru przez node"
        # print left_son_objects
        # print right_son_objects
        if left_son_objects == [] or right_son_objects == []:
            decision = Counter([self.dec[i] for i in objects]).most_common()[0][0]
            return DecisionTree(decision, 0, 0)

        return DecisionTree(best_hyperplane, self.count_decision_tree(left_son_objects, sc=sc, svm=svm),
                            self.count_decision_tree(right_son_objects, sc=sc, svm=svm))


# def cross_val_score(X, y, k=10):
#     div_list = lambda lst, size: [lst[i:i + size] for i in range(0, len(lst), size)]
#     l = range(len(y))
#     random.shuffle(l)
#     folds = div_list(l, int(len(y) / k))
#     accuracy = 0
#     i = 0
#     for test_ids in folds:
#         train_ids = [x for x in range(0, n) if x not in test_ids]
#         X_train = X[train_ids, ]
#         X_test = X[test_ids, ]
#         y_train = y[train_ids]
#         y_test = y[test_ids]
#
#         clf = tree.DecisionTreeClassifier()
#         clf.fit(X_train, y_train)
#         res = clf.predict(X_test)
#         accuracy += (sum(res == y_test) / float(len(res)))
#         i += 1
#         print "performed " + str(i) + " cv, result: " + str((sum(res == y_test) / float(len(res))))
#
#     return accuracy / float(k)


def cross_val_score_gen_dec_tree(X, y, k=10, sc=None, svm=False, mi=40, ps=50, bs=40, mdr=0.8):
    div_list = lambda lst, size: [lst[i:i + size] for i in range(0, len(lst), size)]
    l = range(len(y))
    random.shuffle(l)
    folds = div_list(l, int(len(y) / k))
    accuracy = 0
    i = 0
    for test_ids in folds:
        train_ids = [x for x in range(0, n) if x not in test_ids]
        X_train = X[train_ids,]
        X_test = X[test_ids,]
        y_train = y[train_ids]
        y_test = y[test_ids]

        X_ex = Eav.convert_to_proper_format(X_train)
        md = MostDecisionStrategy(X_ex, y_train, mdr)

        extractor = HyperplaneExtractor(X_ex, list(X_ex.dtype.names), y, md, 3000,
                                            max_iter=mi, population_size=ps, b=bs)

        dec_tree = extractor.count_decision_tree(range(len(y_train)), sc=sc, svm=svm)
        res = dec_tree.predict_list(X_test, svm=svm)
        accuracy += (sum(res == y_test) / float(len(res)))
        i += 1
        print "result len: " + str(len(res))
        print "all set len: " + str(len(y))
        print "performed " + str(i) + " cv, result: " + str((sum(res == y_test) / float(len(res))))

    return accuracy / float(k)


if __name__ == "__main__":
    conf = (SparkConf().setMaster("spark://green07:7077").setAppName("extractor"))
    sc = SparkContext(conf=conf)
    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("org").setLevel(logger.Level.ERROR)
    logger.LogManager.getLogger("akka").setLevel(logger.Level.ERROR)

    ################ prepare data #############################
    data = load_iris()
    X = data['data']
    y = data['target']
    n = len(y)

    # data = sio.loadmat("/home/students/mat/k/kr319379/Downloads/BASEHOCK.mat")
    #
    # X = data['X']
    # y = y = np.array(map(lambda x: x[0], data['Y']))
    # data = genfromtxt("/home/students/mat/k/kr319379/Downloads/marrData.csv", delimiter=",")
    # X = data[:,:-1]
    # y = data[:,-1]

    # data = genfromtxt("/home/students/mat/k/kr319379/Downloads/waveform.data", delimiter=",")
    # X = data[:, :-1]
    # y = data[:, -1]

    train_ids = random.sample(range(0, n), int(0.66 * n))
    test_ids = [x for x in range(0, n) if x not in train_ids]
    X_train = X[train_ids,]
    X_test = X[test_ids,]
    y_train = y[train_ids]
    y_test = y[test_ids]

    X_ex = Eav.convert_to_proper_format(X)
    md = MostDecisionStrategy(X_ex, y, 0.8)

    extractor = HyperplaneExtractor(X_ex, list(X_ex.dtype.names), y, md, 3000)

    ####################################################################################

    ###################### extract table ###############################################
    # new_table = np.column_stack((X, extractor.extract(sc)))
    #
    # new_X_train = new_table[train_ids, ]
    # new_X_test = new_table[test_ids, ]
    #
    # standard_tree = tree.DecisionTreeClassifier()
    # standard_tree_newt = tree.DecisionTreeClassifier()
    #
    # standard_tree.fit(X_train, y_train)
    # results_standard = standard_tree.predict(X_test)
    #
    # standard_tree_newt.fit(new_X_train, y_train)
    # results_standard_newt = standard_tree_newt.predict(new_X_test)
    # print "standard"
    # print accuracy_score(results_standard, y_test)
    # print "standard new table"
    # print accuracy_score(results_standard_newt, y_test)

    # clf = tree.DecisionTreeClassifier()
    # print "standard"
    # print np.mean(cross_val_score(clf, X, y, scoring="accuracy", cv=5))
    # print "standard new table"
    # scores = []
    # times = []
    # params = {'max_iter': [10, 17, 25], 'population_size': [20, 35, 50], 'b': [35]}
    # count = 0
    # for mi in params['max_iter']:
    #     for ps in params['population_size']:
    #         for bs in params['b']:
    #             count += 1
    #             #print "score for " + str(pi) + "max_iter"
    #             start = time.time()
    #             scores = np.mean(cross_val_score(clf, np.column_stack(
    #                 (X, HyperplaneExtractor(X_ex, list(X_ex.dtype.names), y, md, 3000,
    #                 max_iter=mi, population_size=ps, b=bs).extract(sc))), y, scoring="accuracy", cv=5))
    #             times = time.time() - start
    #             res = {'pop': ps, 'mi': mi, 'score': scores, 'time': times, 'b': bs}
    #             fn = "/home/students/mat/k/kr319379/mgr/results/scores" + str(count) + ".pickle"
    #             with open(fn, "wb") as f:
    #                 pickle.dump(res, file=f)
    #             print "results " + str(count)
    #             with open(fn, "rb") as f:
    #                 print pickle.load(file=f)

    # for x in [18, 23, 28]:
    #     print "score for " + str(x) + "max_iter"
    #     start = time.time()
    #     scores.append(np.mean(cross_val_score(clf, np.column_stack(
    #         (X, HyperplaneExtractor(X_ex, list(X_ex.dtype.names), y, md, 3000, max_iter=x).extract(sc))), y,
    #                                           scoring="accuracy", cv=5)))
    #     times.append(time.time() - start)

    # with open("scores.pickle", "wb") as f:
    #     pickle.dump(scores, file=f)
    # with open("times.pickle", "wb") as f:
    #     pickle.dump(times, file=f)
    #
    # print "results:"
    # with open("scores.pickle", "rb") as f:
    #     print pickle.load(file=f)
    # with open("times.pickle", "rb") as f:
    #     print pickle.load(file=f)
    ####################### decision tree #######################################
    print "standard"
    print np.mean(cross_val_score(tree.DecisionTreeClassifier, X, y, scoring="accuracy", cv=5))
    print "standard new table"
    scores = []
    times = []
    params = {'max_iter': [10, 17, 25], 'population_size': [20, 35, 50], 'b': [35], 'mdr': [0.8, 0.9]}
    count = 0
    for mi in params['max_iter']:
        for ps in params['population_size']:
            for bs in params['b']:
                for mdr in params['mdr']:
                    count += 1
                    # print "score for " + str(pi) + "max_iter"
                    start = time.time()
                    tree = cross_val_score_gen_dec_tree(X, y, k=5, sc=sc, mi=mi, ps=ps, bs=bs, mdr=mdr)
                    times = time.time() - start
                    res = {'pop': ps, 'mi': mi, 'score': scores, 'time': times, 'b': bs}
                    fn = "/home/students/mat/k/kr319379/mgr/results/tree_scores" + str(count) + ".pickle"
                    with open(fn, "wb") as f:
                        pickle.dump(res, file=f)
                    print "results " + str(count)
                    with open(fn, "rb") as f:
                        print pickle.load(file=f)
    # print "svm"
    # print cross_val_score_gen_dec_tree(X, y, 5, sc, svm=True)
