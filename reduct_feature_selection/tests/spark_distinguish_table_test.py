import numpy as np
from unittest import TestCase

from commons.spark.tables.distinguish_table import DistinguishTable


class TestDistinguishTable(TestCase):

    def test_convert_to_tuple_of_list(self):
        decision_system = np.array([[1, 1, 0, 1], [0, 1, 0, 0], [1, 1, 1, 1]])
        a = DistinguishTable(decision_system)
        a._convet_to_list_of_tuples()
        self.assertEqual(a._convet_to_list_of_tuples(), [(1, 1, 0, 1), (0, 1, 0, 0), (1, 1, 1, 1)])
