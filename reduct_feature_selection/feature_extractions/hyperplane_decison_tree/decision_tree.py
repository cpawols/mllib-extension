
class DecisionTree:

    def __init__(self, value, left, right):
        self.value = value
        self.left = left
        self.right = right

    def is_leaf(self):
        if type(self.value) == list or type(self.value) == tuple:
            return False
        return True

    def print_tree(self):
        print "value:"
        print self.value
        if not self.is_leaf():
            self.left.print_tree()
            self.right.print_tree()