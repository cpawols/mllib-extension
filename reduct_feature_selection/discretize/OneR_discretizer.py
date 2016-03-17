from simple_discretizer import SimpleDiscretizer
from collections import Counter

import numpy as np


class OneRDiscretizer(SimpleDiscretizer):

    def __init__(self, dec, m):
        self.dec = dec
        self.m = m

    def discretize_column(self, column):
        dis = 0
        dis_elems = 0
        dec_fqs = Counter()
        for elem in column:

            if dis_elems > self.m and dec_fqs.most_common(1)[0][1] > dis_elems / 2:
                dis_elems = 0
                dis += 1
                dec_fqs = Counter()
            dec_fqs[self.dec[elem[0]]] += 1
            dis_elems += 1

            yield (elem[0], elem[1], dis)

if __name__ == "__main__":
    table = np.array([(1, 7), (1, 8), (1, 3), (1, 9), (1, 1), (1, 2), (1, 5), (1, 10)],
                     dtype=[('y', float), ('z', float)])
    dec = [0,1,1,1,0,0,1,1]
    attrs_list = ['z']
    discretizer = OneRDiscretizer(dec, 2)
    print discretizer.discretize(table, attrs_list, par=True)
    print discretizer.discretize(table, attrs_list, par=False)





