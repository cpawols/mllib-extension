from collections import Counter
from unittest import TestCase

from reduct_feature_selection.abstraction_class.aproximation_class_set import SetAbstractionClass


class CatOffTest(TestCase):
    def test_cat_off_thre_attributes(self):
        attributes_rank = Counter({1:1.123, 2:2.333, 3:3.321})
        print attributes_rank.most_common()
        print SetAbstractionClass.cut_attributes(attributes_rank.most_common())
