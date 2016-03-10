import numpy as np
from settings import Configuration


class DistinguishTable:
    def __init__(self, decision_system):
        self.decision_system = decision_system

    @staticmethod
    def compute_distinguish_table(decision_system, subtable_num):
        row_number = decision_system.shape[0]
        col_number = decision_system.shape[1]
        decision_number = col_number - 1
        decision_system = np.transpose(decision_system)
        dec_list = list(tuple(row) for row in decision_system)
        dec_par = Configuration.sc.parallelize(dec_list, subtable_num)
        par = dec_par.map(lambda x : (x,1)).reduceByKey(lambda x,y : x+y).sortByKey().collect()
        print(par)
        print(dec_list)


if __name__ == "__main__":
    dec = np.array([[1, 2, 3], [4, 5, 6]])
    DistinguishTable.compute_distinguish_table(dec, 4)
