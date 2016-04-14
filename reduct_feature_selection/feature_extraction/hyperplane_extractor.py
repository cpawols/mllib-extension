import math
from collections import Counter
from operator import add
from random import randrange, random

import numpy as np

from settings import Configuration
from simple_extractor import SimpleExtractor


class GeneticSearch(object):

    def __init__(self, k, dec, table, axis_table, unconsistent_reg,
                 b=50,
                 first_generation_size=20000,
                 population_size=200,
                 max_iter=100,
                 cross_chance=0.4,
                 mutation_chance=0.01):
        self.b = b
        self.k = k
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
        return [[randrange(-max_vec, max_vec) for j in range(self.k)]
                for i in range(self.first_generation_size)]

    def count_award(self, individual):
        projections = []
        for ind, row in enumerate(self.table):
            x = self.axis_table[ind]
            proj = x
            for i, r in enumerate(row):
                proj -= individual[i] * r
            projections.append((ind, proj))
        objects = [x for x in sorted(projections, lambda x: x[1])]
        reg_ob_map = {}
        reg_fqs_map = {}
        conflicts_map = {}
        act_left_sum = {}
        act_left_sum_sq = {}
        act_right_sum = {}
        act_right_sum_sq = {}

        for ind, reg in enumerate(self.unconsistent_reg):
            reg_fqs = Counter()
            for ob in reg:
                reg_ob_map[ob] = ind
                reg_fqs[self.dec[ob]] += 1
            sum = 0
            sum_sq = 0
            for fq in reg_fqs.values():
                sum += fq
                sum_sq += fq * fq
            conflict = sum * sum - sum_sq
            act_left_sum[ind] = 0
            act_right_sum[ind] = sum
            act_left_sum_sq[ind] = 0
            act_right_sum_sq[ind] = sum_sq
            reg_fqs_map[ind] = {k: (0, v) for k, v in reg_fqs.iteritems()}
            conflicts_map[ind] = conflict

        act_award = 0
        max_award = 0
        for obj, proj in objects:
            ind = reg_ob_map[obj]
            old_award = conflicts_map[ind] \
                        - (act_left_sum[ind] * act_left_sum[ind] - act_left_sum_sq[ind]) \
                        - (act_right_sum[ind] * act_right_sum[ind] - act_right_sum_sq[ind])
            act_left_sum[ind] += 1
            act_right_sum[ind] -= 1
            act_left_sum_sq[ind] += (2 * reg_fqs_map[ind][self.dec[ob]][0] + 1)
            act_right_sum_sq[ind] -= (2 * reg_fqs_map[ind][self.dec[ob]][1] - 1)
            reg_fqs_map[ind][self.dec[ob]] = (reg_fqs_map[ind][self.dec[ob]][0] + 1,
                                         reg_fqs_map[ind][self.dec[ob]][1] - 1)
            new_award = conflicts_map[ind] \
                        - (act_left_sum[ind] * act_left_sum[ind] - act_left_sum_sq[ind]) \
                        - (act_right_sum[ind] * act_right_sum[ind] - act_right_sum_sq[ind])
            gain = new_award - old_award
            act_award += gain
            if act_award > max_award:
                max_award = act_award
                good_proj = proj

        return max_award, individual, good_proj

    def select_best_individuals(self, population_awards):

        population = sorted(population_awards, key=lambda x: x[0], reverse=True)
        best_ind = population[:self.population_size]

        return best_ind

    def get_new_generation(self, population):

        rands = random.sample(range(len(population)), len(population))
        pairs = [(rands[i], rands[i + 1]) for i in range(0, population, 2)]
        new_generation = []

        for pair in pairs:

            ind1 = population[pair[0]]
            ind2 = population[pair[1]]

            if random.random < self.cross_chance:
                el = random.choice(range(len(ind1)))
                pom = ind1[el]
                ind1[el] = ind2[el]
                ind2[el] = pom

            if random.random < self.mutation_chance:
                el = random.choice(range(len(ind1)))
                new_vec = randrange(-math.pow(2, self.b), math.pow(2, self.b))
                ind1[el] = new_vec
            if random.random < self.mutation_chance:
                el = random.choice(range(len(ind1)))
                new_vec = randrange(-math.pow(2, self.b), math.pow(2, self.b))
                ind2[el] = new_vec

            new_generation.append(ind1)
            new_generation.append(ind2)

        return new_generation

    def count_award_for_chunk(self, population):
        for individual in population:
            yield self.count_award(individual)

    # TODO: add stop criterion
    def genetic_search(self, par=False):

        print "--------------------init population-------------------------------------------"
        population = self.init_generation()

        for i in range(self.max_iter):
            print "-----------------------------performing " + str(i) + " generation---------------"
            if par:
                rdd_population = Configuration.sc.parallelize(population, self.population_size * 10)
                awards = rdd_population.mapPartitions(self.count_award_for_chunk).collect()
            else:
                awards = map(self.count_award, population)

            population = self.select_best_individuals(awards)

            best_individual = population[0]

            new_generation = self.get_new_generation(population)

            population = new_generation + population

        return best_individual


class HyperplaneExtractor(SimpleExtractor):

    def extract(self, table, attrs_list, par=False):

        def get_unconsistent_reg(extracted_table, dec):
            table_list = []
            for i, row in enumerate(extracted_table):
                row_k_v = (list(row), (i, dec[i]))
                table_list.append(row_k_v)

            table_list_rdd = Configuration.sc.parallelize(table_list)
            unconsistent_reg_candidates = table_list_rdd.reduceByKey(add).collect()

            regs = []
            for reg in unconsistent_reg_candidates:
                if len(set([x[1] for x in reg])) > 1:
                   regs.append([x[0] for x in reg])
            return regs

        def get_new_column(table, best_hyperplane):
            attr = best_hyperplane[0]
            axis_table = table[attr]
            other_attrs = [x for x in attrs_list if not x == attr]
            new_table = table[other_attrs]
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

        extracted_table = []
        i = 0
        while True:
            best_hyperplane = (0, [0, 0, 0])
            unconsistent_reg = get_unconsistent_reg(extracted_table, self.dec)

            print "--------------------unconsistent regs-----------------------------"
            print unconsistent_reg
            print "-------------------performing " + str(i) + " iteration-----------------------"
            i += 1
            for attr in attrs_list:
                axis_table = table[attr]
                other_attrs = [x for x in attrs_list if not x == attr]
                new_table = table[other_attrs]

                if unconsistent_reg:
                    gen_search = GeneticSearch(len(other_attrs), self.dec, new_table,
                                           axis_table, unconsistent_reg)
                    cand_hyperplane = gen_search.genetic_search(par)
                    if cand_hyperplane[0] > best_hyperplane[1][0]:
                        best_hyperplane = attr, cand_hyperplane
                else:
                    break

            extracted_table.append(get_new_column(table, best_hyperplane))

        return np.array(extracted_table)

if __name__ == "__main__":
    table = np.array([(0, 1, 7), (4, 5, 8), (1, 2, 3), (3, 8, 9),
                      (0, 1, 7), (4, 5, 8), (1, 2, 3), (3, 8, 9)],
                     dtype=[('x', int), ('y', float), ('z', float)])
    dec = [0, 1, 0, 1, 0, 1, 0, 1]
    attrs_list = ['x', 'y', 'z']
    discretizer = HyperplaneExtractor(dec, 0.1)
    table = discretizer.extract(table, attrs_list)
    print table

