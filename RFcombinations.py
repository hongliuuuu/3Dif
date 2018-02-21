
import numpy as np
import pandas
import random

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV
from collections import Counter
from math import floor
from joblib import Parallel, delayed
from sklearn.ensemble import ExtraTreesClassifier
from rr_forest import RRForestClassifier
from rr_extra_forest import RRExtraTreesClassifier
def floored_percentage(val, digits):
    val *= 10 ** (digits + 2)
    return '{1:.{0}f}\%\pm '.format(digits, floor(val) / 10 ** digits)
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

def rf_dis(n_trees, X,Y,train_indices,test_indices,seed):
    clf = RandomForestClassifier(n_estimators=500,
                                 random_state=seed, oob_score=True, n_jobs=-1)
    clf = clf.fit(X[train_indices], Y[train_indices])
    pred = clf.predict(X[test_indices])
    weight = clf.oob_score_
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

def Extreme_rf_dis(n_trees, X,Y,train_indices,test_indices,seed):
    clf = ExtraTreesClassifier(n_estimators=500,
                                 random_state=seed, oob_score=True, n_jobs=-1)
    clf = clf.fit(X[train_indices], Y[train_indices])
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

def RR_rf_dis(n_trees, X,Y,train_indices,test_indices,seed):
    clf = RRForestClassifier(n_estimators=500,
                                 random_state=seed, oob_score=True, n_jobs=-1)
    clf = clf.fit(X[train_indices], Y[train_indices])
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

def RRExtra_rf_dis(n_trees, X,Y,train_indices,test_indices,seed):
    clf = RRExtraTreesClassifier(n_estimators=500,
                                 random_state=seed, oob_score=True, n_jobs=-1)
    clf = clf.fit(X[train_indices], Y[train_indices])
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


def nLsvm_patatune(train_x,train_y,test_x, test_y):
    tuned_parameters = [
        {'kernel': ['precomputed'], 'C': [0.01, 0.1, 1, 10, 100, 1000]}]
    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5, n_jobs=1
                       )  # SVC(probability=True)#SVC(kernel="linear", probability=True)
    clf.fit(train_x, train_y)
    #print(clf.score(test_x,test_y))
    return clf.best_params_['C']

"""
url = 'nus_1.csv'
dataframe = pandas.read_csv(url)  # , header=None)
array = dataframe.values
X = array[:, 1:]

for i in range(4):
    url = 'nus_' + str(i + 2) + '.csv'
    dataframe = pandas.read_csv(url)  # , header=None)
    array = dataframe.values
    X1 = array[:, 1:]
    X = np.concatenate((X, X1), axis=1)
Y = pandas.read_csv('nus_label.csv')
Y = Y.values

Y = Y[:, 1:]
# Y = Y.transpose()
Y = np.ravel(Y)
train_indices, test_indices = splitdata(X=X, Y=Y, ratio=0.1, seed=1000 )

n_trees = 500
n_feat = selected_f(639)  # features selecleted
n_features = X.shape[1]
for i in range(n_features):
    s = X[:,i]
    mn = np.max(s)-np.min(s)
    if mn == 0:
        print(i)
"""


def mcode(ite):
    R = 0.5
    e8 = []
    erfsvm = []
    e8oob = []
    erfsvmoob = []
    e8oobm = []
    erfsvmoobm = []
    e8oobm4 = []
    erfsvmoobm4 = []
    e8ran = []
    erfsvmran = []

    #data reading
    if ite ==0:
        ss = "lowGrade"
        url = '../lowGrade/text_lg_1.csv'
        dataframe = pandas.read_csv(url, header=None)
        array = dataframe.values
        X = array
        Y = pandas.read_csv('../lowGrade/label_lowGrade.csv', header=None)
        Y = Y.values
        Y = np.ravel(Y)
        print(Y.shape)

        for i in range(4):
            url = '../lowGrade/text_lg_' + str(i + 2) + '.csv'
            dataframe = pandas.read_csv(url, header=None)
            array = dataframe.values
            X1 = array
            print(X1.shape)
            X = np.concatenate((X, X1), axis=1)

        Xnew1 = X[:, 0:1680]
        Xnew2 = X[:, 1680:3360]
        Xnew3 = X[:, 3360:5040]
        Xnew4 = X[:, 5040:6720]
        Xnew5 = X[:, 6720:6745]
    elif ite ==1:
        ss = "IDHCodel"
        url = '../IDHCodel/text_pr_1.csv'
        dataframe = pandas.read_csv(url, header=None)
        array = dataframe.values
        X = array
        Y = pandas.read_csv('../IDHCodel/label_IDHCodel.csv', header=None)
        Y = Y.values
        Y = np.ravel(Y)
        print(Y.shape)

        for i in range(4):
            url = '../IDHCodel/text_pr_' + str(i + 2) + '.csv'
            dataframe = pandas.read_csv(url, header=None)
            array = dataframe.values
            X1 = array
            print(X1.shape)
            X = np.concatenate((X, X1), axis=1)

        Xnew1 = X[:, 0:1680]
        Xnew2 = X[:, 1680:3360]
        Xnew3 = X[:, 3360:5040]
        Xnew4 = X[:, 5040:6720]
        Xnew5 = X[:, 6720:6745]
    elif ite == 2:
        ss = "nonIDH1"
        url = '../nonIDH1/text_nonIDH1_1.csv'
        dataframe = pandas.read_csv(url, header=None)
        array = dataframe.values
        X = array
        Y = pandas.read_csv('../nonIDH1/label_nonIDH1.csv', header=None)
        Y = Y.values
        Y = np.ravel(Y)
        print(Y.shape)

        for i in range(4):
            url = '../nonIDH1/text_nonIDH1_' + str(i + 2) + '.csv'
            dataframe = pandas.read_csv(url, header=None)
            array = dataframe.values
            X1 = array
            print(X1.shape)
            X = np.concatenate((X, X1), axis=1)

        Xnew1 = X[:, 0:1680]
        Xnew2 = X[:, 1680:3360]
        Xnew3 = X[:, 3360:5040]
        Xnew4 = X[:, 5040:6720]
        Xnew5 = X[:, 6720:6745]
    else:
        ss = "progression"
        url = '../progression/text_pr_1.csv'
        dataframe = pandas.read_csv(url, header=None)
        array = dataframe.values
        X = array
        Y = pandas.read_csv('../progression/label_progression.csv', header=None)
        Y = Y.values
        Y = np.ravel(Y)
        print(Y.shape)

        for i in range(4):
            url = '../progression/text_pr_' + str(i + 2) + '.csv'
            dataframe = pandas.read_csv(url, header=None)
            array = dataframe.values
            X1 = array
            print(X1.shape)
            X = np.concatenate((X, X1), axis=1)

        Xnew1 = X[:, 0:1680]
        Xnew2 = X[:, 1680:3360]
        Xnew3 = X[:, 3360:5040]
        Xnew4 = X[:, 5040:6720]
        Xnew5 = X[:, 6720:6745]
    testfile = open(("RF"+ss+"%f_%f.txt" % (R, ite)), 'w')
    y = Y.astype(int)
    y_bin = np.bincount(y)
    classes = np.nonzero(y_bin)[0]
    n_classes = len(classes)
    for ii in range(10):



        seed = 1000+ii
        train_indices, test_indices = splitdata(X=X, Y=Y, ratio=R, seed=seed)
        print("Start rest")
        # view1

        X_features_train1, X_features_test1, w1, pred1 = rf_dis(n_trees=500, X=Xnew1, Y=Y, train_indices=train_indices,
                                                                test_indices=test_indices, seed=seed)
        m12 = RandomForestClassifier(n_estimators=500, random_state=seed, oob_score=True, n_jobs=1).fit(
            X_features_train1, Y[train_indices])
        pre1 = m12.predict(X_features_test1)
        #e12.append(m12.score(X_features_test1, Y[test_indices]))
        #e11.append(w1)
        # view 2

        X_features_train2, X_features_test2, w2, pred2 = rf_dis(n_trees=500, X=Xnew2, Y=Y, train_indices=train_indices,
                                                                test_indices=test_indices, seed=seed)
        m22 = RandomForestClassifier(n_estimators=500, random_state=seed, oob_score=True, n_jobs=1).fit(
            X_features_train2, Y[train_indices])
        pre2 = m22.predict(X_features_test2)
        #e22.append(m22.score(X_features_test2, Y[test_indices]))
        #e21.append(w2)

        # view 3

        X_features_train3, X_features_test3, w3, pred3 = rf_dis(n_trees=500, X=Xnew3, Y=Y, train_indices=train_indices,
                                                                test_indices=test_indices, seed=seed)
        m32 = RandomForestClassifier(n_estimators=500, random_state=seed, oob_score=True, n_jobs=1).fit(
            X_features_train3, Y[train_indices])
        pre3 = m32.predict(X_features_test3)
        #e32.append(m32.score(X_features_test3, Y[test_indices]))
        #e31.append(w3)

        # view 4

        X_features_train4, X_features_test4, w4, pred4 = rf_dis(n_trees=500, X=Xnew4, Y=Y, train_indices=train_indices,
                                                                test_indices=test_indices, seed=seed)
        m42 = RandomForestClassifier(n_estimators=500, random_state=seed, oob_score=True, n_jobs=1).fit(
            X_features_train4, Y[train_indices])
        pre4 = m42.predict(X_features_test4)
        #e42.append(m42.score(X_features_test4, Y[test_indices]))
        #e41.append(w4)

        # view 5

        X_features_train5, X_features_test5, w5, pred5 = rf_dis(n_trees=500, X=Xnew5, Y=Y, train_indices=train_indices,
                                                                test_indices=test_indices, seed=seed)
        m52 = RandomForestClassifier(n_estimators=500, random_state=seed, oob_score=True, n_jobs=1).fit(
            X_features_train5, Y[train_indices])
        pre5 = m52.predict(X_features_test5)
        #e52.append(m52.score(X_features_test5, Y[test_indices]))
        #e51.append(w5)


        # multi view
        X_features_trainm = (
                                    X_features_train1 + X_features_train2 + X_features_train3 + X_features_train4 + X_features_train5) / 5
        X_features_testm = (
                                   X_features_test1 + X_features_test2 + X_features_test3 + X_features_test4 + X_features_test5) / 5
        mv = RandomForestClassifier(n_estimators=500, random_state=seed, oob_score=True, n_jobs=1).fit(
            X_features_trainm, Y[train_indices])
        e8.append(mv.score(X_features_testm, Y[test_indices]))

        # RFSVM
        c = nLsvm_patatune(train_x=X_features_trainm, train_y=Y[train_indices], test_x=X_features_testm,
                           test_y=Y[test_indices])

        clf = SVC(C=c, kernel='precomputed')
        clf.fit(X_features_trainm, Y[train_indices])
        erfsvm.append(clf.score(X_features_testm, Y[test_indices]))

        # multi view oob weight
        oobw = [w1,w2,w3,w4,w5]
        X_features_trainm = (
                                    X_features_train1*oobw[0] + X_features_train2*oobw[1] + X_features_train3*oobw[2] + X_features_train4*oobw[3] + X_features_train5*oobw[4]) / (sum(oobw))
        X_features_testm = (
                                   X_features_test1*oobw[0] + X_features_test2*oobw[1] + X_features_test3*oobw[2] + X_features_test4*oobw[3] + X_features_test5*oobw[4]) / (sum(oobw))
        mv = RandomForestClassifier(n_estimators=500, random_state=seed, oob_score=True, n_jobs=1).fit(
            X_features_trainm, Y[train_indices])
        e8oob.append(mv.score(X_features_testm, Y[test_indices]))

        # RFSVM
        c = nLsvm_patatune(train_x=X_features_trainm, train_y=Y[train_indices], test_x=X_features_testm,
                           test_y=Y[test_indices])

        clf = SVC(C=c, kernel='precomputed')
        clf.fit(X_features_trainm, Y[train_indices])
        erfsvmoob.append(clf.score(X_features_testm, Y[test_indices]))

        # multi view oob weight modified
        oobwm = [max(w1-1/n_classes,0), max(w2-1/n_classes,0), max(w3-1/n_classes,0), max(w4-1/n_classes,0), max(w5-1/n_classes,0)]
        X_features_trainm = (
                                    X_features_train1 * oobwm[0] + X_features_train2 * oobwm[1] + X_features_train3 *
                                    oobwm[2] + X_features_train4 * oobwm[3] + X_features_train5 * oobwm[4]) / (sum(oobwm))
        X_features_testm = (
                                   X_features_test1 * oobwm[0] + X_features_test2 * oobwm[1] + X_features_test3 * oobwm[
                               2] + X_features_test4 * oobwm[3] + X_features_test5 * oobwm[4]) / (sum(oobwm))
        mv = RandomForestClassifier(n_estimators=500, random_state=seed, oob_score=True, n_jobs=1).fit(
            X_features_trainm, Y[train_indices])
        e8oobm.append(mv.score(X_features_testm, Y[test_indices]))

        # RFSVM
        c = nLsvm_patatune(train_x=X_features_trainm, train_y=Y[train_indices], test_x=X_features_testm,
                           test_y=Y[test_indices])

        clf = SVC(C=c, kernel='precomputed')
        clf.fit(X_features_trainm, Y[train_indices])
        erfsvmoobm.append(clf.score(X_features_testm, Y[test_indices]))

        # multi view oob weight modified4
        oobwm4 = [pow(wei,4) for wei in oobwm]
        X_features_trainm = (
                                    X_features_train1 * oobwm4[0] + X_features_train2 * oobwm4[1] + X_features_train3 *
                                    oobwm4[2] + X_features_train4 * oobwm4[3] + X_features_train5 * oobwm4[4]) / (
                                sum(oobwm4))
        X_features_testm = (
                                   X_features_test1 * oobwm4[0] + X_features_test2 * oobwm4[1] + X_features_test3 * oobwm4[
                               2] + X_features_test4 * oobwm4[3] + X_features_test5 * oobwm4[4]) / (sum(oobwm4))
        mv = RandomForestClassifier(n_estimators=500, random_state=seed, oob_score=True, n_jobs=1).fit(
            X_features_trainm, Y[train_indices])
        e8oobm4.append(mv.score(X_features_testm, Y[test_indices]))

        # RFSVM
        c = nLsvm_patatune(train_x=X_features_trainm, train_y=Y[train_indices], test_x=X_features_testm,
                           test_y=Y[test_indices])

        clf = SVC(C=c, kernel='precomputed')
        clf.fit(X_features_trainm, Y[train_indices])
        erfsvmoobm4.append(clf.score(X_features_testm, Y[test_indices]))

        # multi view random weight
        np.random.seed(seed)
        ranw = list(np.random.rand(5))
        X_features_trainm = (
                                    X_features_train1 * ranw[0] + X_features_train2 * ranw[1] + X_features_train3 *
                                    ranw[2] + X_features_train4 * ranw[3] + X_features_train5 * ranw[4]) / (
                                sum(ranw))
        X_features_testm = (
                                   X_features_test1 * ranw[0] + X_features_test2 * ranw[1] + X_features_test3 *
                                   ranw[
                                       2] + X_features_test4 * ranw[3] + X_features_test5 * ranw[4]) / (sum(ranw))
        mv = RandomForestClassifier(n_estimators=500, random_state=seed, oob_score=True, n_jobs=1).fit(
            X_features_trainm, Y[train_indices])
        e8ran.append(mv.score(X_features_testm, Y[test_indices]))

        # RFSVM
        c = nLsvm_patatune(train_x=X_features_trainm, train_y=Y[train_indices], test_x=X_features_testm,
                           test_y=Y[test_indices])

        clf = SVC(C=c, kernel='precomputed')
        clf.fit(X_features_trainm, Y[train_indices])
        erfsvmran.append(clf.score(X_features_testm, Y[test_indices]))


    testfile.write(
        "RFSVM&%s pm%s & " % (floored_percentage(np.mean(erfsvm), 2), floored_percentage(np.std(erfsvm), 2)) + '\n')
    testfile.write("RFDIS &%s pm%s & " % (floored_percentage(np.mean(e8), 2), floored_percentage(np.std(e8), 2)) + '\n')
    testfile.write(
        "RFSVMoob&%s pm%s & " % (floored_percentage(np.mean(erfsvmoob), 2), floored_percentage(np.std(erfsvmoob), 2)) + '\n')
    testfile.write("RFDISoob &%s pm%s & " % (floored_percentage(np.mean(e8oob), 2), floored_percentage(np.std(e8oob), 2)) + '\n')

    testfile.write(
        "RFSVMoobm&%s pm%s & " % (floored_percentage(np.mean(erfsvmoobm), 2), floored_percentage(np.std(erfsvmoobm), 2)) + '\n')
    testfile.write("RFDISoobm &%s pm%s & " % (floored_percentage(np.mean(e8oobm), 2), floored_percentage(np.std(e8oobm), 2)) + '\n')

    testfile.write("RFSVMoobm4&%s pm%s & " % (floored_percentage(np.mean(erfsvmoobm4), 2), floored_percentage(np.std(erfsvmoobm4), 2)) + '\n')
    testfile.write("RFDISoobm4 &%s pm%s & " % (floored_percentage(np.mean(e8oobm4), 2), floored_percentage(np.std(e8oobm4), 2)) + '\n')

    testfile.write("RFSVMran&%s pm%s & " % (
    floored_percentage(np.mean(erfsvmran), 2), floored_percentage(np.std(erfsvmran), 2)) + '\n')
    testfile.write("RFDISran &%s pm%s & " % (
    floored_percentage(np.mean(e8ran), 2), floored_percentage(np.std(e8ran), 2)) + '\n')

    print(ss)
    print("RFSVM&%s pm%s & " % (floored_percentage(np.mean(erfsvm), 2), floored_percentage(np.std(erfsvm), 2)) + '\n')
    print("RFDIS &%s pm%s & " % (floored_percentage(np.mean(e8), 2), floored_percentage(np.std(e8), 2)) + '\n')

    print(
        "RFSVMoob&%s pm%s & " % (
        floored_percentage(np.mean(erfsvmoob), 2), floored_percentage(np.std(erfsvmoob), 2)) + '\n')
    print(
        "RFDISoob &%s pm%s & " % (floored_percentage(np.mean(e8oob), 2), floored_percentage(np.std(e8oob), 2)) + '\n')

    print(
        "RFSVMoobm&%s pm%s & " % (
        floored_percentage(np.mean(erfsvmoobm), 2), floored_percentage(np.std(erfsvmoobm), 2)) + '\n')
    print("RFDISoobm &%s pm%s & " % (
    floored_percentage(np.mean(e8oobm), 2), floored_percentage(np.std(e8oobm), 2)) + '\n')

    print("RFSVMoobm4&%s pm%s & " % (
    floored_percentage(np.mean(erfsvmoobm4), 2), floored_percentage(np.std(erfsvmoobm4), 2)) + '\n')
    print("RFDISoobm4 &%s pm%s & " % (
    floored_percentage(np.mean(e8oobm4), 2), floored_percentage(np.std(e8oobm4), 2)) + '\n')

    print("RFSVMran&%s pm%s & " % (
        floored_percentage(np.mean(erfsvmran), 2), floored_percentage(np.std(erfsvmran), 2)) + '\n')
    print("RFDISoobm4 &%s pm%s & " % (
        floored_percentage(np.mean(e8ran), 2), floored_percentage(np.std(e8ran), 2)) + '\n')

if __name__ == '__main__':
    Parallel(n_jobs=4)(delayed(mcode)(ite=i) for i in range(4))