from collections import Counter
from operator import add

import numpy as np
from scipy.spatial.distance import squareform, pdist

from reduct_feature_selection.commons.tables.consistent_checker import ConsistentChecker
from reduct_feature_selection.feature_extractions.doubtful_points_strategies.min_dist_doubtful_points_strategy import \
    MinDistDoubtfulPointsStrategy
from reduct_feature_selection.feature_extractions.genetic_algortihms.genetic_search import GeneticSearch
from reduct_feature_selection.feature_extractions.hyperplane_decison_tree.decision_tree import DecisionTree
from reduct_feature_selection.feature_extractions.simple_extractors import SimpleExtractor
from settings import Configuration


class HyperplaneExtractor(SimpleExtractor):
    def __init__(self, table, attrs_list, dec, cuts_limit_ratio, dpoints_strategy):
        super(HyperplaneExtractor, self).__init__(table, attrs_list, dec, cuts_limit_ratio)
        self.dpoints_strategy = dpoints_strategy

    # TODO: add tests and docs
    def _count_objects_positions(self, best_hyperplane):
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
            if proj > best_hyperplane[1][2]:
                new_column.append(1)
            else:
                new_column.append(0)

        return new_column

    def _search_best_hyperplane_for_projection(self, attr, unconsistent_groups):

        projection_axis = self.table[attr]
        other_axes = [x for x in self.attrs_list if not x == attr]
        new_table = self.table[other_axes]

        gen_search = GeneticSearch(len(other_axes), self.dec, new_table,
                                   projection_axis, unconsistent_groups)
        cand_hyperplane = gen_search.genetic_search(False)

        return attr, cand_hyperplane

    # def _search_best_hyperplane2(self, unconsistent_groups, par):
    #     hyperplanes = []
    #     for attr in self.attrs_list:
    #
    #         projection_axis = self.table[attr]
    #         other_axes = [x for x in self.attrs_list if not x == attr]
    #         new_table = self.table[other_axes]
    #
    #         gen_search = GeneticSearch(len(other_axes), self.dec, new_table,
    #                                    projection_axis, unconsistent_groups)
    #         cand_hyperplane = gen_search.genetic_search(False)
    #
    #         hyperplanes.append((attr, cand_hyperplane))
    #
    #     return max(hyperplanes, key=lambda x: x[1][0])
    #
    # def _search_best_hyperplane(self, unconsistent_reg, par):
    #     best_hyperplane = ('x', (-1, [0], 0))
    #     for attr in self.attrs_list:
    #         print "-------------------counting " + attr + " attrbiute-----------------------"
    #         axis_table = self.table[attr]
    #         other_attrs = [x for x in self.attrs_list if not x == attr]
    #         new_table = self.table[other_attrs]
    #
    #         gen_search = GeneticSearch(len(other_attrs), self.dec, new_table,
    #                                    axis_table, unconsistent_reg)
    #         cand_hyperplane = gen_search.genetic_search(par)
    #         if cand_hyperplane[0] > best_hyperplane[1][0]:
    #             best_hyperplane = attr, cand_hyperplane
    #
    #     return best_hyperplane

    def _search_best_hyperplane(self, unconsistent_groups, par):

        if not par:
            hyperplanes = map(lambda x: self._search_best_hyperplane_for_projection(x, unconsistent_groups),
                              self.attrs_list)
        else:
            rdd_attrs = Configuration.sc.parallelize(self.attrs_list)
            hyperplanes = rdd_attrs.map(
                lambda x: self._search_best_hyperplane_for_projection(x, unconsistent_groups)).collect()

        return max(hyperplanes, key=lambda x: x[1][0])

    def extract(self, par=False):
        extracted_table = []
        i = 0
        while True:
            unconsistent_groups = ConsistentChecker.count_unconsistent_groups(extracted_table, self.dec)

            print "--------------------unconsistent groups-----------------------------"
            print unconsistent_groups

            unconsistent_groups = filter(lambda x: self.dpoints_strategy.decision(x) is None, unconsistent_groups)
            print "--------------------unconsistent regs after clustering-----------------------------"
            print unconsistent_groups
            i += 1
            # TODO: add stop criterion if doesn't stop
            if unconsistent_groups:
                print "-------------------performing " + str(i) + " iteration-----------------------"
                best_hyperplane = self._search_best_hyperplane(unconsistent_groups, par)
                extracted_table.append(self._count_objects_positions(best_hyperplane))
            else:
                break

        return np.array(extracted_table)

    def count_decision_tree(self, objects):
        decision = self.dpoints_strategy.decision(objects)
        if decision is not None:
            return DecisionTree(decision, 0, 0)

        best_hyperplane = self._search_best_hyperplane([objects], False)
        hyperplane_indicator = self._count_objects_positions(best_hyperplane)
        left_son_objects = filter(lambda x: hyperplane_indicator[x] == 0, objects)
        right_son_objects = filter(lambda x: hyperplane_indicator[x] == 1, objects)

        # if left_son_objects == [] or right_son_objects == []:
        #     print left_son_objects
        #     print right_son_objects
        #     print "nie znaleziono plaszczyzny"
        #     return self.count_decision_tree(objects)
        # print left_son_objects
        # print right_son_objects
        # print best_hyperplane

        return DecisionTree(best_hyperplane, self.count_decision_tree(left_son_objects),
                            self.count_decision_tree(right_son_objects))


if __name__ == "__main__":
    logger = Configuration.sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("org").setLevel(logger.Level.ERROR)
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
    new_objects = [1, 3, 5, 6, 9, 11]
    new_dec = [dec[i] for i in new_objects]
    new_table = table[new_objects,]
    md = MinDistDoubtfulPointsStrategy(table, dec, 3)
    discretizer = HyperplaneExtractor(table, attrs_list, dec, 0.1, md)
    # TODO: [1, 3, 5, 6, 9, 11] nie znajduje rozdzielenia
    table = discretizer.extract(par=False)
    #dec_tree = discretizer.count_decision_tree(range(12))
    #dec_tree.print_tree()
    print table
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
