# Settings for test
import os
import sys
import yaml
try:
    from pyspark import SparkContext
    from pyspark import SparkConf
except ImportError as e:
    print ("Error importing Spark Modules", e)
    sys.exit(1)


class Configuration(object):

    @classmethod
    def assign(cls):
        cls.csv_file1 = "../../data/csv_test_file.csv"
        cls.csv_file2 = "../../data/csv_test_float_numbers.csv"
        cls.csv_file3 = os.getcwd() + "../../data/csv_test_float_numbers.csv"
        cls.conf = SparkConf().setAppName("MllibExtension")
        cls.sc = SparkContext(conf=cls.conf)
        # configure the environmental variables
        try:
            paths = yaml.load(open("config.yaml", "r"))
            cls.spark_home = paths['spark_home']
            cls.pythonpath = paths["pythonpath"]
        except Exception as e:
            print ("No config file or wrong paths", e)
            sys.exit(1)
