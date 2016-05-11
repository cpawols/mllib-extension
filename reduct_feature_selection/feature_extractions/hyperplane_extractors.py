from collections import Counter
from operator import add

import numpy as np
import time
from scipy.spatial.distance import squareform, pdist
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_iris, load_digits
from sklearn.metrics import accuracy_score

from reduct_feature_selection.commons.tables.consistent_checker import ConsistentChecker
from reduct_feature_selection.commons.tables.eav import Eav
from reduct_feature_selection.feature_extractions.doubtful_points_strategies.min_dist_doubtful_points_strategy import \
    MinDistDoubtfulPointsStrategy
from reduct_feature_selection.feature_extractions.genetic_algortihms.genetic_search import GeneticSearch
from reduct_feature_selection.feature_extractions.hyperplane_decison_tree.decision_tree import DecisionTree
from reduct_feature_selection.feature_extractions.simple_extractors import SimpleExtractor
from pyspark import SparkContext, SparkConf

from sklearn.svm import LinearSVC


class HyperplaneExtractor(SimpleExtractor):
    def __init__(self, table, attrs_list, dec, dpoints_strategy, time_search_limit=1000):
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

        projection_axis = self.table[attr]
        other_axes = [x for x in self.attrs_list if not x == attr]
        new_table = self.table[other_axes]

        gen_search = GeneticSearch(len(other_axes), self.dec, new_table,
                                   projection_axis, unconsistent_groups)
        cand_hyperplane = gen_search.genetic_search(sc)

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

    def _search_best_hyperplane(self, unconsistent_groups, sc=None):

        if sc is None:
            hyperplanes = map(lambda x: self._search_best_hyperplane_for_projection(x, unconsistent_groups, sc),
                              self.attrs_list)
        else:
            rdd_attrs = sc.parallelize(self.attrs_list)
            hyperplanes = rdd_attrs.map(
                lambda x: self._search_best_hyperplane_for_projection(x, unconsistent_groups)).collect()

        return max(hyperplanes, key=lambda x: x[1][0])

    def extract(self, sc=None):
        extracted_table = []
        i = 0
        start = time.time()
        while True:
            unconsistent_groups = ConsistentChecker.count_unconsistent_groups(extracted_table, self.dec, sc)

            print "--------------------unconsistent groups-----------------------------"
            print unconsistent_groups

            unconsistent_groups = filter(lambda x: self.dpoints_strategy.decision(x) is None, unconsistent_groups)
            print "--------------------unconsistent regs after clustering-----------------------------"
            print unconsistent_groups
            i += 1
            time_spent = time.time() - start
            if unconsistent_groups and time_spent < self.time_search_limit:
                print "-------------------performing " + str(i) + " iteration-----------------------"
                best_hyperplane = self._search_best_hyperplane(unconsistent_groups, sc)
                extracted_table.append(self._count_objects_positions(best_hyperplane))
            else:
                break

        return np.transpose(np.array(extracted_table))

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
            left_son_objects = map(lambda x: x[1], filter(lambda (i, x): not hyperplane_indicator[i], enumerate(objects)))
            right_son_objects = map(lambda x: x[1], filter(lambda (i, x): hyperplane_indicator[i], enumerate(objects)))
        else:
            best_hyperplane = self._search_best_hyperplane([objects], sc)
            hyperplane_indicator = self._count_objects_positions(best_hyperplane, objects)
            left_son_objects = map(lambda x: x[1], filter(lambda (i, x): hyperplane_indicator[i] == 0, enumerate(objects)))
            right_son_objects = map(lambda x: x[1], filter(lambda (i, x): hyperplane_indicator[i] == 1, enumerate(objects)))

        print "podzial zbioru przez node"
        print left_son_objects
        print right_son_objects
        if left_son_objects == [] or right_son_objects == []:
            decision = Counter([self.dec[i] for i in objects]).most_common()[0][0]
            return DecisionTree(decision, 0, 0)

        return DecisionTree(best_hyperplane, self.count_decision_tree(left_son_objects, svm=svm),
                            self.count_decision_tree(right_son_objects, svm=svm))


if __name__ == "__main__":
    conf = (SparkConf().setMaster("spark://localhost:7077").setAppName("extractor"))
    sc = SparkContext(conf=conf)
    logger = sc._jvm.org.apache.log4j
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
    attrs_list = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']
    new_objects = [1, 3, 5, 6, 9, 11]
    new_dec = [dec[i] for i in new_objects]
    new_table = table[new_objects,]
    md = MinDistDoubtfulPointsStrategy(table, dec, 3)
    discretizer = HyperplaneExtractor(table, attrs_list, dec, md, 60)
    table = discretizer.extract(sc)
    print table
    # dec_tree = discretizer.(range(12), svm=True)
    # dec_tree.print_tree()
    # for row in table:
    #     print dec_tree.predict(list(row), svm=True)

    # iris = load_iris()
    # X_train, X_test, y_train, y_test = train_test_split(
    #     iris['data'], iris['target'], test_size=0.33, random_state=42)
    # iris_train = Eav.convert_to_proper_format(X_train)
    # iris_test = Eav.convert_to_proper_format(X_test)
    # md = MinDistDoubtfulPointsStrategy(iris_train, y_train, 3)
    # extractor = HyperplaneExtractor(iris_train, list(iris_train.dtype.names), y_train, md, 300)
    # dec_tree = extractor.count_decision_tree(range(len(iris_train)))
    # svm_dec_tree = extractor.count_decision_tree(range(len(iris_train)), svm=True)
    # standard_tree = tree.DecisionTreeClassifier()
    # standard_tree.fit(Eav.convert_to_proper_array(iris_train), y_train)
    # results_standard = standard_tree.predict(Eav.convert_to_proper_array(iris_test))
    # results_gen = dec_tree.predict_list(iris_test)
    # results_svm = svm_dec_tree.predict_list(iris_test, svm=True)
    # print "genetic"
    # print accuracy_score(results_gen, y_test)
    # print "svm"
    # print accuracy_score(results_svm, y_test)
    # print "standard"
    # print accuracy_score(results_standard, y_test)

    iris = load_iris()
    X = Eav.convert_to_proper_format(iris['data'])
    y = iris['target']
    md = MinDistDoubtfulPointsStrategy(X, y, 3)
    extractor = HyperplaneExtractor(X, list(X.dtype.names), y, md, 300)
    extracted_table = extractor.extract()
    X_train, X_test, y_train, y_test = train_test_split(
        extracted_table, y, test_size=0.33, random_state=42)
    standard_tree = tree.DecisionTreeClassifier()
    standard_tree.fit(Eav.convert_to_proper_array(X_train), y_train)
    results = standard_tree.predict(Eav.convert_to_proper_array(X_test))
    print "extracted table"
    print extracted_table
    print "score"
    print accuracy_score(y_test, results)

    # svm = LinearSVC()
    # prop_tab = Eav.convert_to_proper_array(table)
    # svm.fit(prop_tab, dec)
    # print svm.coef_
    # print svm.intercept_
    # #print svm.predict(table)
    # coefs = svm.coef_[0]
    # inters = svm.intercept_[0]
    # l = map(lambda row: np.dot(row, coefs) + inters > 0, prop_tab)
    # print "------------------res-----------------------------------"
    # print l
    # print svm.predict(prop_tab)


    #print table
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
