from collections import Counter

from scipy.spatial.distance import pdist, squareform
import numpy as np

from reduct_feature_selection.feature_extractions.doubtful_points_strategies.base_doubtful_points_strategy import \
    BaseDoubtfulPointsStrategy


class MostDecisionStrategy(BaseDoubtfulPointsStrategy):
    def __init__(self, table, dec, max_ratio):
        super(MostDecisionStrategy, self).__init__(table, dec)
        self.max_ratio = max_ratio

    def decision(self, objects):
        ob_dec = [self.dec[obj] for obj in objects]
        if len(set(ob_dec)) == 1:
            return self.dec[objects[0]]
        dec_hist = Counter(ob_dec)
        most_common_len = dec_hist.most_common(1)[0][1]
        if most_common_len / float(len(ob_dec)) > self.max_ratio or ob_dec < 4:
            return dec_hist.most_common(1)[0][0]
        return None
