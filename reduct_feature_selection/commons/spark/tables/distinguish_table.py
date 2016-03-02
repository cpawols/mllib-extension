"""
Creating distinguish table using spark

Divide decision system into attribute chunks:
    - transpose matrix
    - divide by split from numpy (number of chunks depends on number of computers on the cluster)
    each chunks get a list with decisions
    - for each chunk compute distinguish table which is saved as dictionary
        keys are a tuple of position in matrix and values are list attributes which distinguish the objects
        - make statistic of frequency for each object something more??? TODO
    - merge dictionaries updating values on each key

As input we take numpy array which will be converted to list of tuples because spark rdd object doesn't support np.array
"""
import numpy as np
from numpy.ma import transpose, copy



class DistinguishTable:
    # @staticmethod
    # def compute_distinguish_table(decision_system, subtable_num):
    #     row_number = decision_system.shape[0]
    #     col_number = decision_system.shape[1]
    #     decision_number = col_number - 1
    #     decision_system = np.transpose(decision_system)
    #     dec_list = list(tuple(row) for row in decision_system)
    #     dec_par = sc.parallelize(dec_list, subtable_num)
    #     par = dec_par.map(lambda x: (x, 1)).reduceByKey(lambda x, y: x + y).sortByKey().collect()
    #     print(par)
    #     print(dec_list)

    def __init__(self, decision_system):
        self.decision_system = decision_system

    def _transopse_matrix(self, matrix):
        """Transpose a numpy matrix"""
        decision_system_transpose = transpose(matrix)
        return decision_system_transpose

    def _get_decision_list(self, matrix):
        """Return a decision list"""
        return matrix[:, -1]

    def _convert_to_list_of_tuples(self, local_decision_system):
        """Convert np.array to list of tuples"""
        list_of_tuples = [tuple(row, ) + (i,) for i, row in enumerate(local_decision_system)]
        return list_of_tuples

    def _remove_decision_column(self, matrix):
        """Removing decision column from np array"""
        return np.delete(matrix, -1, 1)

    def _prepare_data_make_distinguish_table(self):
        """Preparing decision system"""
        # TODO make it cleaner
        copyd = copy(self.decision_system)
        decisions = self._get_decision_list(copyd)
        copyd = self._remove_decision_column(copyd)
        transopse_decision_system = self._transopse_matrix(copyd)
        list_of_tuples = self._convert_to_list_of_tuples(transopse_decision_system)
        return list_of_tuples, decisions

    def make_table(self, system, decisions):
        """ Computing decision table"""

        # system, decisions = self._prepare_data_make_distinguish_table()

        if len(system[0]) - 1 != len(decisions):
            raise ValueError("Different length of decisions and objects")

        res = dict()
        for i, attributes in enumerate(system):
            for j in range(len(attributes) - 1):
                for k in range(j, len(attributes) - 1):
                    if j != k and system[i][j] != system[i][k] and decisions[j] != decisions[k]:
                        if (j, k) not in res:
                            res[(j, k)] = [attributes[-1]]
                        else:
                            res[(j, k)].append(attributes[-1])

        return res

    def spark_part(self, decision_system, subtable_num=5):
        pass
        # convert_ds = self._convert_to_list_of_tuples(decision_system)
        # dec_par = sc.parallelize(convert_ds, subtable_num)
        # par = dec_par.map(self.make_table)
        # print(par)




if __name__ == "__main__":

    x = {'a': [1, 2], 'b': [3, 4]}
    y = {'a': [3], 'b': [1, 2]}
    res = {}
    for (k1, v1), (k2, v2) in zip(x.items(), y.items()):
        res[k1] = v1 + v2
        # print res
