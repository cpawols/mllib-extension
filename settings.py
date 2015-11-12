# Settings for test
import os
import sys
try:
    from pyspark import SparkContext
    from pyspark import SparkConf
except ImportError as e:
    print ("Error importing Spark Modules", e)
    sys.exit(1)

csv_file1 = os.getcwd() + "/data/csv_test_file.csv"
csv_file2 = os.getcwd() + "/data/csv_test_float_numbers.csv"
csv_file3 = os.getcwd() + "/data/csv_test_float_numbersDOESNOTEXIST.csv"
conf = SparkConf().setAppName("PythonSort")
sc = SparkContext(conf=conf)
# configure the environmental variables
spark_home = "/home/krzysztof/mgr/spark-1.5.1"
pythonpath = "/home/krzysztof/mgr/spark-1.5.1/python"
