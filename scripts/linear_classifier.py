import sys
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import KFold

train = pd.read_csv(
    '../data/train/x_train_gr_smpl_normalized.csv').values / 255.0
train_labels = pd.read_csv('../data/train/y_train_smpl.csv').values.ravel()

if sys.argv[1] == '-kfold':  # 10 fold validation
    print('USING 10 FOLD CROSSVALIDATION')
    for train_index, test_index in KFold(10).split(train):
        x_train, x_test = train[train_index], train[test_index]
        y_train, y_test = train_labels[train_index], train_labels[test_index]
        clf = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)
        clf.fit(x_train, y_train)
        print(clf.score(x_test, y_test))
elif sys.argv[1] == '-testset':
    test = pd.read_csv('../data/test/x_test_gr_smpl_normalized.csv').values / 255.0
    test_labels = pd.read_csv('../data/test/y_test_smpl.csv').values.ravel()

    clf = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)
    clf.fit(train, train_labels)
    print(clf.score(test, test_labels))

else:
    print('Invalid arguments.')
    print('Launch script with "-kfold" or "-testset"')
