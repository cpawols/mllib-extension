from random import randrange
import random
import copy
import math
from operator import add
from collections import Counter

from settings import Configuration


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