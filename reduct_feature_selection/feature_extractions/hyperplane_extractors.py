import math
from collections import Counter
from operator import add
from random import randrange
import random
import copy
import collections

import numpy as np
from scipy.spatial.distance import squareform, pdist

from settings import Configuration
from reduct_feature_selection.feature_extractions.simple_extractors import SimpleExtractor


class GeneticSearch(object):

    def __init__(self, k, dec, table, axis_table, unconsistent_reg,
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
        self.axis_table = axis_table
        self.unconsistent_reg = unconsistent_reg

    def init_generation(self):
        max_vec = math.pow(2, self.b)
        return [[randrange(-max_vec, max_vec) for _ in range(self.k)]
                for _ in range(self.first_generation_size)]

    def count_global_award(self, individual):
        projections = []
        valid_objects = reduce(add, self.unconsistent_reg)
        for ind, row in enumerate(self.table):
            if ind in valid_objects:
                x = self.axis_table[ind]
                proj = x
                if self.k > 1:
                    for i, r in enumerate(row):
                        proj -= individual[i] * r
                else:
                    proj -= individual[0] * row
                projections.append((ind, proj))
        objects = [x for x in sorted(projections, key=lambda x: x[1])]
        reg_ob_map = {}
        reg_fqs_map = {}
        act_left_sum = {}
        act_sum_sq = {}
        act_right_sum = {}

        for ind, reg in enumerate(self.unconsistent_reg):
            reg_fqs = Counter()
            for obj in reg:
                reg_ob_map[obj] = ind
                reg_fqs[self.dec[obj]] += 1
            sum = 0
            for fq in reg_fqs.values():
                sum += fq

            act_left_sum[ind] = 0
            act_right_sum[ind] = sum
            act_sum_sq[ind] = 0
            reg_fqs_map[ind] = {k: (0, v) for k, v in reg_fqs.iteritems()}

        act_award = 0
        max_award = 0
        for obj, proj in objects:
            ind = reg_ob_map[obj]
            act_award += (act_right_sum[ind] - act_left_sum[ind] - 1) - \
                         (reg_fqs_map[ind][self.dec[obj]][1] - reg_fqs_map[ind][self.dec[obj]][0] - 1)
            act_left_sum[ind] += 1
            act_right_sum[ind] -= 1
            reg_fqs_map[ind][self.dec[obj]] = (reg_fqs_map[ind][self.dec[obj]][0] + 1,
                                              reg_fqs_map[ind][self.dec[obj]][1] - 1)
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
            yield self.count_global_award(individual)

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
    def genetic_search(self, par=False):

        print "--------------------init population-------------------------------------------"
        population = self.init_generation()

        current_best_award = 0
        the_same_awards = 0
        for i in range(self.max_iter):
            print "-----------------------------performing " + str(i) + " generation---------------"
            if par:
                rdd_population = Configuration.sc.parallelize(population, self.population_size * 10)
                awards = rdd_population.mapPartitions(self.count_award_for_chunk).collect()
            else:
                awards = map(self.count_global_award, population)

            population_awards = self.select_best_individuals(awards)

            best_individual = population_awards[0]

            if best_individual[0] > current_best_award:
                current_best_award = best_individual[0]
                the_same_awards = 0
            else:
                the_same_awards += 1

            population = map(lambda x: x[1], population_awards)

            new_generation = self.count_new_generation(copy.deepcopy(population))

            print sorted(new_generation) == sorted(population)

            population = new_generation + population

            set_population = self._eliminate_duplicates(population)
            sorted_population = sorted(population)
            print "new generation ratio"
            print float(len(set_population)) / float(len(sorted_population))
            if the_same_awards > self.stop_treshold:
                break

        return best_individual


class HyperplaneExtractor(SimpleExtractor):

    def _get_unconsistent_reg(self, extracted_table):
            if extracted_table:
                table_list = []
                for i, decision in enumerate(self.dec):
                    row = tuple(map(lambda x: x[i], extracted_table))
                    table_list.append((row, (i, decision)))

                table_list_rdd = Configuration.sc.parallelize(table_list)
                unconsistent_reg_candidates = table_list_rdd.reduceByKey(add).collect()

                regs = []
                for reg in unconsistent_reg_candidates:
                    reg_decisions = [reg[1][i] for i in range(1, len(reg[1]), 2)]
                    if len(set(reg_decisions)) > 1:
                        reg_objects = [reg[1][i] for i in range(0, len(reg[1]), 2)]
                        regs.append(reg_objects)
                return regs
            return [range(len(self.dec))]

    def _is_small_distance(self, region, min_dist):
        points = [list(self.table[obj, ]) for obj in region]
        distances = np.array(points)
        D = squareform(pdist(distances))
        N = np.max(D)
        if N <= min_dist:
            return True
        return False

    def _get_new_column(self, best_hyperplane):
            attr = best_hyperplane[0]
            axis_table = self.table[attr]
            other_attrs = [x for x in self.attrs_list if not x == attr]
            new_table = self.table[other_attrs]
            new_column = []
            for ind, row in enumerate(new_table):
                x = axis_table[ind]
                proj = x
                for i, r in enumerate(row):
                    proj -= best_hyperplane[1][1][i] * r
                if proj >= best_hyperplane[1][2]:
                   new_column.append(1)
                else:
                    new_column.append(0)

            return new_column

    def _search_best_hyperplane(self, unconsistent_reg, par):
        best_hyperplane = ('x', (-1, [0], 0))
        # TODO: make it paralell
        for attr in self.attrs_list:
            print "-------------------counting " + attr + " attrbiute-----------------------"
            axis_table = self.table[attr]
            other_attrs = [x for x in self.attrs_list if not x == attr]
            new_table = self.table[other_attrs]

            gen_search = GeneticSearch(len(other_attrs), self.dec, new_table,
                                               axis_table, unconsistent_reg)
            cand_hyperplane = gen_search.genetic_search(par)
            if cand_hyperplane[0] > best_hyperplane[1][0]:
                best_hyperplane = attr, cand_hyperplane

        return best_hyperplane

    # TODO: maybe add table and attrs_list to object arguments
    def extract(self, par=False):

        extracted_table = []
        i = 0
        while True:
            unconsistent_reg = self._get_unconsistent_reg(extracted_table)

            print "--------------------unconsistent regs-----------------------------"
            print unconsistent_reg

            # TODO: other strategy of closer points
            unconsistent_reg = filter(lambda x: not self._is_small_distance(x, 1), unconsistent_reg)
            print "--------------------unconsistent regs after clustering-----------------------------"
            print unconsistent_reg
            i += 1
            # TODO: add stop criterion if doesn't stop
            if unconsistent_reg:
                print "-------------------performing " + str(i) + " iteration-----------------------"
                best_hyperplane = self._search_best_hyperplane(unconsistent_reg, par)
                extracted_table.append(self._get_new_column(best_hyperplane))
            else:
                break

        # TODO: make classifier from table
        return np.array(extracted_table)

    # TODO: make decision tree
    def count_decision_tree(self, objects):
        dec_set = set([self.dec[obj] for obj in objects])

        if len(dec_set) == 1:
            return DecisionTree(self.dec[objects[0]], 0, 0)
        if self._is_small_distance(objects, 3):
            return DecisionTree(Counter(dec_set).most_common(1)[0][0], 0, 0)

        best_hyperplane = self._search_best_hyperplane([objects], False)
        hyperplane_indicator = self._get_new_column(best_hyperplane)
        left_son_objects = filter(lambda x: hyperplane_indicator[x] == 0, objects)
        right_son_objects = filter(lambda x: hyperplane_indicator[x] == 1, objects)

        if left_son_objects == [] or right_son_objects == []:
            print left_son_objects
            print right_son_objects
            print "nie znaleziono plaszczyzny"
            return self.count_decision_tree(objects)
        return DecisionTree(best_hyperplane, self.count_decision_tree(left_son_objects),
                            self.count_decision_tree(right_son_objects))


class DecisionTree:

    def __init__(self, value, left, right):
        self.value = value
        self.left = left
        self.right = right

    def is_leaf(self):
        if type(self.value) == list or type(self.value) == tuple:
            return False
        return True

    def print_tree(self):
        print "value:"
        print self.value
        if not self.is_leaf():
            self.left.print_tree()
            self.left.print_tree()


if __name__ == "__main__":
    logger = Configuration.sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("org"). setLevel(logger.Level.ERROR)
    logger.LogManager.getLogger("akka").setLevel(logger.Level.ERROR)
    table = np.array([(0, 1, 7, 1, 1, 1),
                      (4, 5, 8, 2, 3, 4),
                       (1, 2, 3, 5, 6, 2),
                       (3, 8, 9, 2, 3, 5),
                       (0, 1, 7, 6, 7, 1),
                       (4, 5, 8, 5, 4, 1),
                       (1, 2, 3, 8, 9, 1),
                       (10, 2, 3, 8, 9, 1),
                       (400, 2, 3, 8, 9, 1),
                       (7, 2, 3, 8, 9, 1),
                       (900, 2, 3, 8, 9, 1),
                       (3, 8, 9, 5, 1, 2)],
                      dtype=[('x', int), ('y', float), ('z', float), ('a', int), ('b', float), ('c', float)])
    dec = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    attrs_list = ['x', 'y', 'z', 'a', 'b', 'c']
    discretizer = HyperplaneExtractor(table, attrs_list, dec, 0.1)
    # TODO: [1, 3, 5, 6, 9, 11] nie znajduje rozdzielenia
    #table = discretizer.extract()
    dec_tree = discretizer.count_decision_tree(range(12))
    dec_tree.print_tree()
    # un_reg = [[0,1,2,3,4]]
    # table = np.array([(-1,-1),
    #                   (-1,1),
    #                   (1,1),
    #                   (1,-1),
    #                   (2,-1)],
    #                  dtype=[('x', float), ('y', float)])
    # dec = [0,0,0,1,1]
    # axis_table = table['x']
    # new_table = table['y']
    # gen = GeneticSearch(1, dec, new_table, axis_table, un_reg)
    # ind1 = [0.5]
    # print gen.count_award(ind1)
    # table2 = np.array([(-1,-1),
    #                   (-1,1),
    #                   (1,1),
    #                   (1,-1),
    #                   (2,-1),
    #                   (2,1)],
    #                  dtype=[('x', float), ('y', float)])
    # dec = [0,0,1,0,1,1]
    # axis_table = table2['x']
    # new_table = table2['y']
    # un_reg1 = [[0,1,2,3,4,5]]
    # un_reg2 = [[0,1,2], [3,4,5]]
    # gen1 = GeneticSearch(1, dec, new_table, axis_table, un_reg1)
    # gen2 = GeneticSearch(1, dec, new_table, axis_table, un_reg2)
    # ind1 = [-0.25]
    # print gen1.count_award(ind1)
    # print gen2.count_award(ind1)

