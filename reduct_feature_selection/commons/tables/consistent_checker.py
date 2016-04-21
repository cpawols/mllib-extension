from settings import Configuration
from operator import add


class ConsistentChecker(object):
    @staticmethod
    def reduce_by_rows(extracted_table, dec):
        if extracted_table:
            table_list = []
            for i, decision in enumerate(dec):
                row = tuple(map(lambda x: x[i], extracted_table))
                table_list.append((row, (i, decision)))

            table_list_rdd = Configuration.sc.parallelize(table_list)
            return table_list_rdd.reduceByKey(add).collect()

        return []

    @staticmethod
    def is_consistent(extracted_table, dec):
        '''
        :param extracted_table: list of rows
        :param dec: list of decisions
        :return: true if table is consistent else false
        '''

        row_groups = ConsistentChecker.reduce_by_rows(extracted_table, dec)

        for row_group in row_groups:
            group_decisions = [row_group[1][i] for i in range(1, len(row_group[1]), 2)]
            if len(set(group_decisions)) > 1:
                return False
        return True

    @staticmethod
    def count_unconsistent_groups(extracted_table, dec):
        '''
        :param extracted_table: list of rows
        :param dec: list of decisions
        :return: list of lists of unconistent subtables (indices of rows)
        '''

        row_groups = ConsistentChecker.reduce_by_rows(extracted_table, dec)

        if row_groups:
            groups = []
            for row_group in row_groups:
                group_decisions = [row_group[1][i] for i in range(1, len(row_group[1]), 2)]
                if len(set(group_decisions)) > 1:
                    group = [row_group[1][i] for i in range(0, len(row_group[1]), 2)]
                    groups.append(group)
            return groups
        return [range(len(dec))]
