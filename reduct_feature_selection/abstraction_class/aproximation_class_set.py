"""
TODO
"""

import itertools
import numpy as np
import operator
from collections import Counter
from random import randint

from reduct_feature_selection.abstraction_class.abstraction_class import AbstractionClass
from settings import Configuration


class SetAbstractionClass:
    def __init__(self, table):
        """
        TODO
        :param table: numpy array
        :return:
        """
        self.table = table

    def _prepare_table(self):
        """
        This function convert numpy array to list of tuple.
        One row is represented by one tuple.
        :return: List of tuples.
        """
        return [tuple(row) + (i,) for i, row in enumerate(self.table)]

    def _choose_subsets(self):
        """
        This method gets this subsets for which is computed approximation.
        :return:
        """
        pass

    def _compute_belief(self, lower_approximation, decision_distribution, decision_subset):
        """
        This function compute believe function for given subset.
        Believe function it's a ratio LowerApproximation(X_i \cup ... \cup X_n)
        to the |(X_1 \cup ... \cup X_n)|.
        Where X_1 and X_2 are a decisions.
        :return: Value of ratio for given subset
        """
        card_lower = sum(len(e) for e in lower_approximation)
        card_decision = self.compute_card_decision(decision_distribution, decision_subset)
        return 1.0 * card_lower / card_decision

    @staticmethod
    def compute_card_decision(decision_distribution, decision_subset):
        """
        TODO
        :param decision_distribution:
        :param decision_subset:
        :return:
        """
        card_decision = sum(value for key, value in decision_distribution.items() if key in decision_subset)
        return card_decision

    def _compute_pl(self, upper_approximation, decision_distribution, decision_subset):
        """
        Compute ratio UpperApproximation(X_i \cup ... \cup X_n)
        to the |(X_1 \cup ... \cup X_n)|.
        :return:
        """
        card_upper = sum(len(e) for e in upper_approximation)
        card_decision = self.compute_card_decision(decision_distribution, decision_subset)
        return 1.0 * card_upper / card_decision

    def compute_approximation(self, decision_subset, broadcast_abstraction_class, broadcast_decision_system,
                              approximation_list=False):
        """
        Compute
        :param broadcast_decision_system:
        :param broadcast_abstraction_class:
        :param decision_subset: subset of decisions.
        :return:
        """

        lower_approximation = self._compute_lower_approximation(
            decision_subset, broadcast_abstraction_class, broadcast_decision_system)

        upper_approximation = self._compute_upper_approximation(
            decision_subset, broadcast_abstraction_class, broadcast_decision_system)

        decision_distribution = self._get_decision_distribution()

        belief = self._compute_belief(lower_approximation, decision_distribution, decision_subset)
        pl = self._compute_pl(upper_approximation, decision_distribution, decision_subset)

        if approximation_list:
            return [belief, pl, decision_subset, lower_approximation, upper_approximation]
        else:
            return [abs(belief - pl), decision_subset]

    @staticmethod
    def _compute_upper_approximation(decision_subset, broadcast_abstraction_class, broadcast_decision_system):
        """
        Only one have to be.
        :param decision_subset:
        :param abstraction_class:
        :param decision_system:
        :return:
        """
        upper_approximation = []

        for abstraction_class in broadcast_abstraction_class.value:
            for object in abstraction_class:
                if (broadcast_decision_system.value[object][-1] in decision_subset and
                            abstraction_class not in upper_approximation):
                    upper_approximation.append(abstraction_class)
                    continue
        return upper_approximation

    @staticmethod
    def _compute_lower_approximation(decision_subset, broadcast_abstraction_class, broadcast_decision_system):
        """
        TODO
        :param decision_subset:
        :param broadcast_abstraction_class:
        :param broadcast_decision_system:
        :return:
        """
        lower_approximation = []
        for abstraction_class in broadcast_abstraction_class.value:
            add_class = True
            for object in abstraction_class:
                if broadcast_decision_system.value[object][-1] not in decision_subset:
                    add_class = False
            if add_class:
                lower_approximation.append(abstraction_class)
        return lower_approximation

    def engine(self, subsets_cardinality=2, approximation_list=False):
        """
        TODO
        :return:
        """
        abstraction_class = self.get_abstraction_class()

        broadcast_table = Configuration.sc.broadcast(self.table)
        broadcast_abstraction_class = Configuration.sc.broadcast(abstraction_class)
        decision_subset = self._create_decision_subsets(subsets_cardinality)
        rdd_decision_subset = Configuration.sc.parallelize(decision_subset)

        res = rdd_decision_subset.map(
            lambda x: self.compute_approximation(x, broadcast_abstraction_class, broadcast_table, approximation_list))
        return res

    def _create_decision_subsets(self, range_of_subsets):
        """
        TODO at the moment choose all subsets to given range.
        :param range_of_subsets:
        :return:
        """
        decision_distribution = self._get_decision_distribution()
        subsets = []
        for k in range(1, range_of_subsets):
            subsets.append(
                list(itertools.combinations(range(min(decision_distribution), max(decision_distribution) + 1), k + 1)))
        return [list(e) for e in reduce(operator.add, subsets)]

    def _get_decision_distribution(self):
        """
        TODO
        :return:
        """
        return Counter(decision[-1] for decision in self.table)

    def get_abstraction_class(self):
        """
        TODO
        :return:
        """
        abstraction_class = AbstractionClass(self.table)
        abstraction_class = abstraction_class.get_abstraction_class()
        return abstraction_class


if __name__ == "__main__":
    table = np.array(
        [
            [1, 1, 0, 0, 1],
            [1, 1, 0, 0, 2],
            [0, 0, 1, 1, 2],
            [0, 1, 1, 0, 1],
            [1, 0, 0, 1, 2],
            [1, 0, 0, 1, 2],
            [1, 1, 1, 0, 3],
            [1, 1, 1, 0, 2],
            [1, 1, 1, 0, 1],
            [1, 1, 1, 0, 1]

        ]
    )

    # table = np.array([[randint(0, 6) for _ in range(5)] for _ in range(5000)])

    # table = np.append(table, np.array([[randint(1, 50)] for _ in range(5000)]), axis=1)
    a = SetAbstractionClass(table)
    res = a.engine(2, approximation_list=True).collect()
    res = sorted(res)
    print res
