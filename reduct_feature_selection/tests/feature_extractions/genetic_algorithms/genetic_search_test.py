import copy
from unittest import TestCase

import numpy as np

from reduct_feature_selection.feature_extractions.genetic_algortihms.genetic_search import GeneticSearch


class TestGeneticSearch(TestCase):

    def test_count_award1(self):
        un_reg = [[0,1,2,3,4]]
        table = np.array([(-1,-1),
                        (-1,1),
                        (1,1),
                        (1,-1),
                        (2,-1)],
                        dtype=[('x', float), ('y', float)])
        dec = [0,0,0,1,1]
        axis_table = table['x']
        new_table = table['y']
        gen = GeneticSearch(1, dec, new_table, axis_table, un_reg)
        individual = [0.5]
        award = gen.count_global_award(individual)
        award_local = gen.count_local_award(individual)
        self.assertEqual(award[0], 6)
        self.assertEqual(award[2], 0.5)
        self.assertEqual(award, award_local)

    def test_count_award2(self):
        table2 = np.array([(-1,-1),
                        (-1,1),
                        (1,1),
                        (1,-1),
                        (2,-1),
                        (2,1)],
                        dtype=[('x', float), ('y', float)])
        dec = [0,0,1,0,1,1]
        axis_table = table2['x']
        new_table = table2['y']
        un_reg = [[0,1,2], [3,4,5]]
        gen = GeneticSearch(1, dec, new_table, axis_table, un_reg)
        ind = [-0.25]
        award = gen.count_global_award(ind)
        self.assertEqual(award[0], 4)
        self.assertEqual(award[2], 0.75)

    def test_new_generation(self):
        un_reg = [[0,1,2,3,4]]
        table = np.array([(-1,-1),
                        (-1,1),
                        (1,1),
                        (1,-1),
                        (2,-1)],
                        dtype=[('x', float), ('y', float)])
        dec = [0,0,0,1,1]
        axis_table = table['x']
        new_table = table['y']
        gen = GeneticSearch(1, dec, new_table, axis_table, un_reg)
        population = gen.init_generation()
        self.assertFalse(sorted(gen.count_new_generation(copy.deepcopy(population))) == sorted(population))
