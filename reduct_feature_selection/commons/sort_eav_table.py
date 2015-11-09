"""TODO"""

from pyspark import SparkContext, SparkConf

class SortEavTableSpark:
    """
    This class  contains function which sort eav table
     firstly by attribute then by value of attribute
    """

    @staticmethod
    def sort_eav_table(eav_table, sc):
        eav_table = sc.parallelize(eav_table)
        return eav_table.map(lambda x: ((x[1], x[2], x[0]), 1)).sortByKey()
