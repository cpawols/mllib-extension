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
    dec = [0,1,1,1,0,0,1,1]
    attrs_list = ['C1', 'C2']
    discretizer = OneRDiscretizer(table, attrs_list, dec, 2)
    discretizer.compare_time()



