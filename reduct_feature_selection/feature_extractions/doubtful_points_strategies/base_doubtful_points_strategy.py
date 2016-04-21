import numpy as np


class BaseDoubtfulPointsStrategy(object):

    def __init__(self, table, dec):
        self.table = table
        self.dec = dec

    def extract_points_matrix(self, objects):
        return np.array([list(self.table[obj, ]) for obj in objects])

    def decision(self, objects):
        dec_set = set([self.dec[obj] for obj in objects])
        if len(dec_set) == 1:
            return self.dec[objects[0]]
        return None
