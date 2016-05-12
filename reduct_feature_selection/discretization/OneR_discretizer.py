from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB

from reduct_feature_selection.commons.tables.eav import Eav
from reduct_feature_selection.discretization.simple_discretizer import SimpleDiscretizer
from collections import Counter

import numpy as np
from pyspark import SparkContext, SparkConf


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

if __name__ == "__main__":
    conf = (SparkConf().setMaster("spark://localhost:7077").setAppName("entropy"))
    sc = SparkContext(conf=conf)
    table = np.array([(1, 7),
                      (1, 8),
                      (1, 3),
                      (1, 9),
                      (1, 1),
                      (1, 2),
                      (1, 5),
                      (1, 10)])
    # dec = [0,1,1,1,0,0,1,1]
    # attrs_list = ['C1', 'C2']
    # discretizer = OneRDiscretizer(table, attrs_list, dec, 2)

    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris['data'], iris['target'], test_size=0.2, random_state=42)
    iris_data = Eav.convert_to_proper_format(iris['data'])
    discretizer = OneRDiscretizer(iris_data, ['C1', 'C2', 'C3'], iris['target'])
    table1 = discretizer.discretize()
    # table2 = discretizer.discretize()
    # clf = GaussianNB()
    # print discretizer.compare_eval(clf)
    # print "discretized"
    print table1



