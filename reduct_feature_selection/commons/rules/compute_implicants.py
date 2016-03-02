"""
TODO
"""
import itertools


class Implicants:
    def __init__(self):
        """

        :return:
        """
        pass

    @staticmethod
    def compute_implicants(self, distinguish_table, frequency_of_attributes):
        """

        :param distinguish_table:
        :param frequency_of_attributes:
        :return:
        """

        processed_objects = [False for _ in range(len(distinguish_table))]

        processed = [False for _ in range(len(distinguish_table))]
        set_of_implicants = []

        frequency_of_attributes = [e[0] for e in frequency_of_attributes]

        aa = list(reversed(frequency_of_attributes))

        for num_obj, row in enumerate(distinguish_table):
            frequency_of_attribute_tmp = aa[:]
            choosen_attributes = set()
            while not all(x for x in processed_objects):

                actual_attribute = frequency_of_attribute_tmp[0]
                frequency_of_attribute_tmp.pop(0)

                for i, element in enumerate(row):
                    if not processed_objects[i]:
                        if element == set():
                            processed_objects[i] = True
                        elif actual_attribute in element:
                            processed_objects[i] = True
                            choosen_attributes.add(actual_attribute)

            processed[num_obj] = True
            set_of_implicants.append(list(choosen_attributes))

            processed_objects = [False for _ in range(len(distinguish_table))]

        print((set_of_implicants))
        set_of_implicants.sort()

        return ((k for k,_ in itertools.groupby(set_of_implicants)))
