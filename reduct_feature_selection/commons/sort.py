# from __future__ import print_function
import sys
try:
    from pyspark import SparkContext
    from pyspark import SparkConf
except ImportError as e:
    print ("Error importing Spark Modules", e)
    sys.exit(1)

if __name__ == "__main__":

    conf = SparkConf().setAppName("PythonSort")
    sc = SparkContext(conf=conf)

    list_to_sort = [(0, "b", 1), (0, "c", 1), (0, "a", 0), (1, "a", 1), (1, "b", 3), (1, "c", 0),
                    (2, "a", 1), (2, "b", 2), (2, "c", 0), (3, "a", 0), (3, "b", 0), (3, "c", 1)]
    lines = sc.parallelize(list_to_sort)

    sortedCount = lines.map(lambda x: ((x[1], x[2], x[0]), 1)).sortByKey()

    output = sortedCount.collect()

    print(type(output))
    #print(output.unitcount())
    table = []

    for (num, unitcount) in output:
        # table.append((num[2], num[0], num[1]))
        print(num, unitcount)

    sc.stop()

    for row in table:
        print(row)
