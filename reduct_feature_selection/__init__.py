import os
import sys
#from settings import spark_home, pythonpath
__author__ = 'krzysztof'

os.environ['SPARK_HOME'] = "/home/pawols/Spark/spark-1.5.1"
# Append to PYTHONPATH so that pyspark could be found
sys.path.append("/home/pawols/Spark/spark-1.5.1/python")

# Now we are ready to import Spark Modules

