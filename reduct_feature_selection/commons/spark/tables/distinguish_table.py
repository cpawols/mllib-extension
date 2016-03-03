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

from pyspark import SparkConf, SparkContext


class DistinguishTable:
    def __init__(self, decision_system):
        self.decision_system = decision_system

    @staticmethod
    def _transopse_matrix(matrix):
        """Transpose a numpy matrix"""
        decision_system_transpose = transpose(matrix)
        return decision_system_transpose

    @staticmethod
    def _get_decision_list(matrix):
        """Return a decision list"""
        return matrix[:, -1]

    @staticmethod
    def _convert_to_list_of_tuples(local_decision_system):
        """Convert np.array to list of tuples"""
        list_of_tuples = [tuple(row, ) + (i,) for i, row in enumerate(local_decision_system)]
        return list_of_tuples

    @staticmethod
    def _remove_decision_column(matrix):
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

    @staticmethod
    def join_dictionaries(x, y):
        """Join two dictionaries into one"""
        dicts = [x, y]
        super_dict = {}
        for d in dicts:
            for k, v in d.iteritems():
                if k in super_dict:
                    super_dict[k] = super_dict[k] + [element for element in v]
                else:
                    super_dict[k] = [element for element in v]
        return super_dict

    @staticmethod
    def make_table(system, decisions=None):
        """ Computing decision table"""
        res = dict()
        res.update()
        for i, attributes in enumerate(system):
            for j in range(len(attributes) - 1):
                for k in range(j, len(attributes) - 1):
                    if j != k and attributes[j] != attributes[k] and decisions[j] != decisions[k]:
                        if (j, k) not in res:
                            res[(j, k)] = [attributes[-1]]
                        else:
                            res[(j, k)].append(attributes[-1])
        yield res

    def spark_part(self, conf, number_of_chunks=2):
        system, decisions = self._prepare_data_make_distinguish_table()
        sc = SparkContext(conf=conf)
        system_rdd = sc.parallelize(system, number_of_chunks)
        result = system_rdd.mapPartitions(lambda x: self.make_table(x, decisions)).reduce(self.join_dictionaries)
        return result



if __name__ == "__main__":
    decision_system = np.array([[1, 0, 2,1], [0, 1,2,0],[1,1,1,1],[3,3,3,1],[2,1,0,0]])
    conf = SparkConf().setAppName("aaaa")
    A = DistinguishTable(decision_system)
    result = A.spark_part(conf, number_of_chunks=1)
    print result

