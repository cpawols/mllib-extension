# # coding=utf-8
# import matplotlib.pyplot as plt
# import os
# from os.path import isfile, join, splitext
#
# path = '/home/pawols/Develop/Mgr/Wyniki/Reduktowe/Marr/WynikiEwaluacji'
# path_to_save = '/home/pawols/Develop/Mgr/Wyniki/Reduktowe/Marr/Wykresy'
#
# onlyfiles = [f for f in os.listdir(path) if isfile(join(path, f))]
#
# for wyn in onlyfiles:
#     x = []
#     y = []
#     with open(join(path, wyn), 'rb') as file:
#         for line in file:
#             line = line.strip().split("\t")
#             x.append(int(line[0]))
#             y.append(float(line[1]))
#
#         file.close()
#
#     plt.plot(y)
#     plt.savefig(join(path_to_save, splitext(wyn)[0] )+'.png')
#     plt.close()
#
from collections import Counter
from random import randint
import numpy as np

x = np.zeros((3000, 501))


for row in range(3000):
    for col in range(500):
        x[row][col] = randint(0,5)
        if sum(x[row][:20]) > 50:
            x[row][-1] = 1
        else:
            x[row][-1] = 0

print Counter(x[:,-1])

np.savetxt('/home/pawols/Develop/Mgr/mllib-extension/sztuczna_tabela.csv', x, delimiter=',')


