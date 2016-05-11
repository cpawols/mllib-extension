from random import randrange
import random
import copy
import math
from operator import add
from collections import Counter

import numpy as np
import time
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_digits

from reduct_feature_selection.commons.tables.eav import Eav
from reduct_feature_selection.feature_extractions.disc_measure_calculator import DiscMeasureCalculator

#from pyspark import SparkContext, SparkConf


def timeit(method):

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print '%r %2.5f sec' % \
              (method.__name__, te - ts)
        return result

    return timed


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
        self.population_size = population_size
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

    # TODO: add test
    @timeit
    def count_projections2(self, individual, objects):
        '''
        :param individual: hyperplane
        :param objects: objects projection to axis
        :return: list of tuples (objects, projection) projection of object to axis
        '''
        projections = []
        table = self._get_subtable(self.table, objects)
        for obj, row in enumerate(table):
            proj = self.projection_axis[objects[obj]]
            if self.k > 1:
                for ind, r in enumerate(row):
                    proj -= individual[ind] * r
            else:
                proj -= individual[0] * row
            projections.append((objects[obj], proj))
        return projections

    @timeit
    def count_projections(self, individual, objects):
        '''
        :param individual: hyperplane
        :param objects: objects projection to axis
        :return: list of tuples (objects, projection) projection of object to axis
        '''
        table = self.table_as_matrix[objects, :]
        proj = [self.projection_axis[obj] for obj in objects]
        return zip(objects, proj - table.dot(individual))

    @timeit
    def count_projections3(self, individual, objects):
        '''
        :param individual: hyperplane
        :param objects: objects projection to axis
        :return: list of tuples (objects, projection) projection of object to axis
        '''
        table = self._get_subtable_as_matrix(self.table_as_matrix, objects)
        proj = [self.projection_axis[obj] for obj in objects]
        return zip(objects, proj - table.dot(individual))

    # TODO: add tests and docs
    def count_award(self, individual):
        if len(self.unconsistent_groups) == 1:
            return self.count_local_award(individual)
        return self.count_global_award(individual)

    def count_global_award2(self, individual):
        valid_objects = reduce(add, self.unconsistent_groups)
        projections = self.count_projections(individual, valid_objects)
        objects = [x for x in sorted(projections, key=lambda x: x[1])]

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

    @timeit
    def count_local_award2(self, individual):
        valid_objects = self.unconsistent_groups[0]
        projections = self.count_projections(individual, valid_objects)
        objects = [x for x in sorted(projections, key=lambda x: x[1])]
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

    @timeit
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
    def genetic_search(self, sc=None):

        print "--------------------init population-------------------------------------------"
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

if __name__ == "__main__":
    # conf = (SparkConf().setMaster("spark://localhost:7077").setAppName("extractor"))
    # sc = SparkContext(conf=conf)
    # logger = sc._jvm.org.apache.log4j
    # logger.LogManager.getLogger("org").setLevel(logger.Level.ERROR)
    # logger.LogManager.getLogger("akka").setLevel(logger.Level.ERROR)

    iris = load_digits(3)
    X_train, X_test, y_train, y_test = train_test_split(
        iris['data'], iris['target'], test_size=0.33, random_state=42)
    iris_train = Eav.convert_to_proper_format(X_train)
    iris_test = Eav.convert_to_proper_format(X_test)
    tab_names = ['C' + str(i + 2) for i in range(63)]
    gen = GeneticSearch(63, y_train, iris_train[tab_names], iris_train['C1'], [range(len(iris_train))])
    gen.count_local_award2(range(63))
    gen.count_local_award(range(63))
    # gen.count_projections(range(63), range(len(iris_train)))
    # gen.count_projections2(range(63), range(len(iris_train)))
    # gen.count_projections3(range(63), range(len(iris_train)))
