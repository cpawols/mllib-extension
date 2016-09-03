import os
import pickle
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.tree import tree

from os.path import basename

path='/home/pawols/Develop/Mgr/experiment-results/reducts/marr/'

x = open('/home/pawols/Pulpit/discr', 'rw')
x = pickle.load(x)
x.astype(int)
data = np.genfromtxt("/home/pawols/Develop/Mgr/mgr/marrData.csv", delimiter=",")
X = data[:, :-1]
y = data[:, -1]

table = np.column_stack((x, y))
table = table.astype(int)


for file in os.listdir(path):
    tmp = path + 'up_reducts_1500_4500.p'
    f = open(tmp, 'rb')
    # f = open(path + file, 'rb')
    res = pickle.load(f)
    k_folds = 15
    print file
    for z in range(1000,1500, 1):
        sel = [e[0] for j, e in enumerate(res.most_common()) if j < z]
        avg_score = 0
        for i in range(k_folds):
            X_train, X_test, y_train, y_test = train_test_split(
                x, y, test_size=0.3, random_state=42)
            clf = tree.DecisionTreeClassifier()
            clf.fit(X_train[:, sorted(sel)], y_train)
            avg_score += clf.score(X_test[:, sorted(sel)], y_test)
        with open(path + 'evaluation_results/' + basename(file) + '_new.txt','a' ) as output_file:
            output_file.write(str(len(sel)) + ' ' + str(1.0*avg_score/k_folds) + '\n')
        print len(sel), 1.0*avg_score/k_folds
    print file

