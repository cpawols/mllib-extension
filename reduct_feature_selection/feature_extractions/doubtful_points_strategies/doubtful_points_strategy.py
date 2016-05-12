import numpy as np


class DoubtfulPointsStrategy(object):

    def __init__(self, table, region):
        self.table = table
        self.region = region

    def extract_points_matrix(self):
        return np.array([list(self.table[obj, ]) for obj in self.region])

    def filter(self):
        return True
