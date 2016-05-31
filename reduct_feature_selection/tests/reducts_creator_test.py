import numpy as np
from unittest import TestCase

from reduct_feature_selection.reducts.compute_reducts import ReductsCreator
from reduct_feature_selection.reducts.distinguish_matrix import DistniguishMatrixCreator


class ReductCreatorTest(TestCase):
    def test_reduct_creator(self):
        table = np.array([
            [1, 1, 0, 0, 0],
            [1, 1, 0, 1, 0],
            [2, 1, 0, 0, 1],
            [0, 2, 0, 0, 1],
            [0, 0, 1, 0, 1],
            [0, 0, 1, 1, 0],
            [2, 0, 1, 1, 1],
            [1, 2, 0, 0, 0],
            [1, 0, 1, 0, 1],
            [0, 2, 1, 0, 1],
            [1, 2, 1, 1, 1],
            [2, 2, 0, 1, 1]
        ])

        ds_creator = DistniguishMatrixCreator(table)
        ds_matrix = ds_creator.create_distinguish_table()
        reduct_generator = ReductsCreator(table, sorted(ds_matrix.keys()))
        list_of_reducts = reduct_generator.compute_up_reduct(ds_matrix)

        reducts =reduct_generator.check_if_reduct(table, list_of_reducts)
        self.assertEqual(sorted(reducts), sorted([[0, 1, 3], [0, 2, 3]]))

    def test_reduct_creator_2(self):
        table = np.array([
            [1, 1, 0, 0, 0],
            [1, 1, 0, 1, 0],
            [2, 1, 0, 0, 1],
        ])

        ds_creator = DistniguishMatrixCreator(table)
        ds_matrix = ds_creator.create_distinguish_table()
        reduct_generator = ReductsCreator(table, sorted(ds_matrix.keys()))
        list_of_reducts = reduct_generator.compute_up_reduct(ds_matrix)

        reducts =reduct_generator.check_if_reduct(table, list_of_reducts)

        self.assertEqual(sorted(reducts), sorted([[0]]))

    def test_reduct_creator_3(self):
        table = np.array([
            [0, 2, 1, 0, 1],
            [1, 2, 1, 1, 1],
            [2, 2, 0, 1, 1]
        ])
        ds_creator = DistniguishMatrixCreator(table)
        ds_matrix = ds_creator.create_distinguish_table()
        reduct_generator = ReductsCreator(table, sorted(ds_matrix.keys()))
        list_of_reducts = reduct_generator.compute_up_reduct(ds_matrix)

        reducts =reduct_generator.check_if_reduct(table, list_of_reducts)
        self.assertEqual(sorted(reducts), sorted([[]]))

    def test_reduct_creator_4(self):
        table = np.array([
                [1, 1, 0, 0, 0],
                [1, 2, 0, 1, 0],
                [2, 1, 0, 0, 1],
                [1, 2, 0, 0, 1],
                [2, 2, 0, 1, 1]
            ])

        ds_creator = DistniguishMatrixCreator(table)
        ds_matrix = ds_creator.create_distinguish_table()
        reduct_generator = ReductsCreator(table, sorted(ds_matrix.keys()))
        list_of_reducts = reduct_generator.compute_up_reduct(ds_matrix)

        reducts =reduct_generator.check_if_reduct(table, list_of_reducts)

        self.assertEqual(sorted(reducts), sorted([[0, 1, 3]]))

    def test_reduct_creator_5(self):
        table = np.array([
                [12, 13, 0, 8, 0],
                [1, 2, 0, 13, 0],
                [2, 1, 15, 0, 1],
                [4, 5, 0, 2, 1],
                [2, 1, 9, 1, 1]
            ])

        ds_creator = DistniguishMatrixCreator(table)
        ds_matrix = ds_creator.create_distinguish_table()
        print ds_matrix
        reduct_generator = ReductsCreator(table, sorted(ds_matrix.keys()))
        list_of_reducts = reduct_generator.compute_up_reduct(ds_matrix)
        print list_of_reducts

        reducts =reduct_generator.check_if_reduct(table, list_of_reducts)
        print reducts
        #self.assertEqual(sorted(reducts), sorted([[0, 1, 3]]))