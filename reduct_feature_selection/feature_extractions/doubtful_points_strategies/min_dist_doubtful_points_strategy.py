from collections import Counter

from scipy.spatial.distance import pdist, squareform
import numpy as np

from reduct_feature_selection.feature_extractions.doubtful_points_strategies.base_doubtful_points_strategy import \
    BaseDoubtfulPointsStrategy


class MinDistDoubtfulPointsStrategy(BaseDoubtfulPointsStrategy):
    def __init__(self, table, dec, min_dist):
        super(MinDistDoubtfulPointsStrategy, self).__init__(table, dec)
        self.min_dist = min_dist

    def decision(self, objects):
        ob_dec = [self.dec[obj] for obj in objects]
        if len(set(ob_dec)) == 1:
            return self.dec[objects[0]]

        point_matrix = self.extract_points_matrix(objects)
        sq_form = squareform(pdist(point_matrix))
        max_dist = np.max(sq_form)

        if max_dist <= self.min_dist:
            return Counter(ob_dec).most_common(1)[0][0]
        return None
