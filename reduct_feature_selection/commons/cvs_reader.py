"""
 This class will be using to reading csv files
"""

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
    def read_csv(path, number_of_header_lines=0):
        """
        This function read to np.array csv file
        :param path: path to csv file
        :return: np.array
        """
        # if not os.path.isfile(path):
        try:
            return genfromtxt(path, delimiter=', ', skip_header=number_of_header_lines)
        except:
            raise ValueError("File does not exist!", path)
