# coding=utf-8
import os
from collections import Counter

from os.path import isfile, join, splitext

import pickle

from sklearn.cross_validation import train_test_split
from sklearn.tree import tree
import numpy as np
from sympy.stats import variance

path='/home/pawols/Develop/Mgr/experiment-results/rules_selection/marr/'
result_path='/home/pawols/Develop/Mgr/Wyniki/Reduktowe/Marr2/ewaluacja/'
x = open('/home/pawols/Pulpit/discr', 'rw')
x = pickle.load(x)
x.astype(int)
data = np.genfromtxt("/home/pawols/Develop/Mgr/mgr/marrData.csv", delimiter=",")

# import scipy.io as sio
# table = sio.loadmat('/home/pawols/Develop/Mgr/mgr/BASEHOCK.mat')
y = data[:, -1]
onlyfiles = [f for f in os.listdir(path) if isfile(join(path, f))]
onlyfiles = filter(lambda x: x.startswith('BAS_dis'), onlyfiles)

print onlyfiles
# onlyfiles = ['BAS_reducts_150_10_9000.p', 'BAS_reducts_150_10_8000.p', 'BAS_reducts_150_10_7000.p',
#              'BAS_reducts_150_10_6000.p', 'BAS_reducts_150_10_10000.p']

onlyfiles = ['rules_attr_sel_5_15_5500.p']
k_folds=10

for file in onlyfiles:
    if True is True:
        f = open(join(path, file), 'rb')
        res = pickle.load(f)
        #::q
        # print res
        #res = Counter(a for e in res for a in e)
        avg_score2 = 0
        var_all = []
        coun = 0
        av_var = 0
        with open(join(result_path, splitext(file)[0]) + '.txt', 'w') as rs:

            for z in range(1, 1000, 3):
                avg_score = 0
                sel = [e[0] for j, e in enumerate(res.most_common()) if j < z]
                #print "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
                wq = []
                for i in range(k_folds):
                    X_train, X_test, y_train, y_test = train_test_split(
                        x, y, test_size=0.33, random_state=42)
                    clf = tree.DecisionTreeClassifier()
                    clf.fit(X_train[:, sorted(sel)], y_train)
                    wq.append(clf.score(X_test[:, sorted(sel)], y_test))
                    var_all.append(wq[-1])
                    avg_score += clf.score(X_test[:, sorted(sel)], y_test)
                avg_score2 += 1.0 * avg_score / k_folds
                av_var += np.var(wq)
                tmp_mean = sum(wq)/k_folds
                tmp_var = 0
                for g in wq:
                    tmp_var += (g-tmp_mean)**2

                print np.var(wq), (len(wq)), 1.0*tmp_var/(k_folds)
                coun += 1
                # rs.write(str(z) + '\t' + str(1.0 * avg_score / k_folds) + '\n')
            #print file, 1.0*avg_score2 / 1500
            print "Ostateczna warincja", 1.0*av_var/coun
        rs.close()
        print 1.0*sum(var_all)/len(var_all), np.var(wq)




