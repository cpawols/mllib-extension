import numpy as np


class DecisionTree:

    def __init__(self, value, left, right):
        self.value = value
        self.left = left
        self.right = right

    def is_leaf(self):
        if type(self.value) == list or type(self.value) == tuple:
            return False
        return True

    def _goto_right_son(self, object, svm):
        if svm:
            return np.dot(object, self.value[1]) + self.value[0] > 0
        else:
            nr_attr = int(self.value[0][1:]) - 1
            result = object[nr_attr]
            other = object[:nr_attr] + object[(nr_attr + 1):]
            for i, elem in enumerate(other):
                result -= self.value[1][1][i] * elem
            return result > self.value[1][2]

    def predict(self, object, svm=False):
        if self.is_leaf():
            return self.value
        if self._goto_right_son(object, svm):
            return self.right.predict(object, svm)
        return self.left.predict(object, svm)

    def predict_list(self, dataset, svm=False):
        return map(lambda r: self.predict(list(r), svm), dataset)

    def print_tree(self):
        print "value:"
        print self.value
        if not self.is_leaf():
            self.left.print_tree()
            self.right.print_tree()