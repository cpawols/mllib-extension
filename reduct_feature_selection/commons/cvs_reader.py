"""
 This class will be using to reading csv files
"""
import os

from numpy import genfromtxt

class CSVReader:

    def __init__(self, path):
        """
        Constructor
        :param path: path to file
        :return:
        """
        self.path = path

    @staticmethod
    def read_csv(path):

        """
        This function read to np.array csv dile
        :param path: path to csv file
        :return: np.array
        """
        if not os.path.isfile(path):
            raise ValueError("File does not exist!")
        return genfromtxt(path, delimiter=', ')
