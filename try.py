from uci_loader import *
X, y = getdataset('diabetes')
import pandas
from rr_forest import RRForestClassifier
from rr_extra_forest import RRExtraTreesClassifier
from sklearn.ensemble.forest import RandomForestClassifier
import numpy as np
import random
def splitdata(X,Y,ratio,seed):
    '''This function is to split the data into train and test data randomly and preserve the pos/neg ratio'''
    n_samples = X.shape[0]
    y = Y.astype(int)
    y_bin = np.bincount(y)
    classes = np.nonzero(y_bin)[0]
    #fint the indices for each class
    indices = []
    print()
    for i in classes:
        indice = []
        for j in range(n_samples):
            if y[j] == i:
                indice.append(j)
        #print(len(indice))
        indices.append(indice)
    train_indices = []
    for i in indices:
        k = int(len(i)*ratio)
        train_indices += (random.Random(seed).sample(i,k=k))
    #find the unused indices
    s = np.bincount(train_indices,minlength=n_samples)
    mask = s==0
    test_indices = np.arange(n_samples)[mask]
    return train_indices,test_indices

def RR_rf_dis(n_trees, X,Y,train_indices,test_indices,seed):
    clf = RRForestClassifier(n_estimators=n_trees,
                                 random_state=seed,  n_jobs=-1)
    clf.fit(X[train_indices], Y[train_indices])
    pred = clf.predict(X[test_indices])
    weight = clf.score(X[test_indices], Y[test_indices])
    #print(1 - clf.oob_score_)
    n_samples = X.shape[0]
    dis = np.zeros((n_samples,n_samples))
    for i in range(n_samples):
        dis[i][i] = 1
    res = clf.apply(X)
    for i in range(n_samples):
        for j in range(i+1,n_samples):
            a = np.ravel(res[i])
            b = np.ravel(res[j])
            score = a == b
            d = float(score.sum())/n_trees
            dis[i][j]  =dis[j][i] = d
    X_features1 = np.transpose(dis)
    X_features2 = X_features1[train_indices]
    X_features3 = np.transpose(X_features2)
    return X_features3[train_indices],X_features3[test_indices],weight,pred

from uci_loader import *
X, y = getdataset('diabetes')
print(X.shape)
clf = RRForestClassifier(n_estimators=20, random_state=1000,  n_jobs=-1)
indd = int(len(y) / 2)
clf.fit(X[:indd], y[:indd])
print("Random Rotation Forest Accuracy:", np.mean(clf.predict(X[indd:]) == y[indd:]))

classifier = RRExtraTreesClassifier(n_estimators=20)
classifier.fit(X[:indd], y[:indd])
print("Random Rotation Extra Trees Accuracy:", np.mean(classifier.predict(X[indd:]) == y[indd:]))

classifier = RandomForestClassifier(n_estimators=20)
classifier.fit(X[:indd], y[:indd])
print("Random Forest Accuracy:", np.mean(classifier.predict(X[indd:]) == y[indd:]))

train_indices, test_indices = splitdata(X=X, Y=y, ratio=0.5, seed=1000)
print("Start rest")
# view1
seed =1000
X_features_train1, X_features_test1, w1, pred1 = RR_rf_dis(n_trees=10, X=X, Y=y, train_indices=train_indices,
                                                           test_indices=test_indices, seed=seed)
m12 = RandomForestClassifier(n_estimators=500, random_state=seed, oob_score=True, n_jobs=1).fit(
    X_features_train1, y[train_indices])
pre1 = m12.predict(X_features_test1)

print("finished view1")