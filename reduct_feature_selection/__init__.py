import os
import sys
import yaml
from settings import Configuration
#from settings import spark_home, pythonpath
__author__ = 'krzysztof'

Configuration.assign()

try:
    paths = yaml.load(open("config.yaml", "r"))
    os.environ['SPARK_HOME'] = paths["spark_home"]
    # Append to PYTHONPATH so that pyspark could be found
    sys.path.append(paths["pythonpath"])
except OSError:
    print "No config file"
    sys.exit(1)

# Now we are ready to import Spark Modules

