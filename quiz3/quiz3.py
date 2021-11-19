from sklearn import datasets
import numpy as np
from joblib import dump, load


def decisioncommon():
    digits = datasets.load_digits()
    data = digits.images
    target = digits.target
    data = data.reshape((data.shape[0], -1))
    return target, data


def test_digit_correct_0():
    data, target = decisioncommon()
    l = np.where(target == 0)
    k = l[0]
    clf = load("model1svm.joblib")
    pred = clf.predict(data[k])
    # print(pred)
    assert pred[0] == 0


def test_digit_correct_1():
    data, target = decisioncommon()
    l = np.where(target == 1)
    k = l[0]
    clf = load("model1svm.joblib")
    pred = clf.predict(data[k])
    # print(pred)
    assert pred[0] == 1


def test_digit_correct_2():
    data, target = decisioncommon()
    l = np.where(target == 2)
    k = l[0]
    clf = load("model1svm.joblib")
    pred = clf.predict(data[k])
    # print(pred)
    assert pred[0] == 2


def test_digit_correct_3():
    data, target = decisioncommon()
    l = np.where(target == 3)
    k = l[0]
    clf = load("model1svm.joblib")
    pred = clf.predict(data[k])
    # print(pred)
    assert pred[0] == 3


def test_digit_correct_4():
    data, target = decisioncommon()
    l = np.where(target == 4)
    k = l[0]
    clf = load("model1svm.joblib")
    pred = clf.predict(data[k])
    # print(pred)
    assert pred[0] == 4


def test_digit_correct_5():
    data, target = decisioncommon()
    l = np.where(target == 5)
    k = l[0]
    clf = load("model1svm.joblib")
    pred = clf.predict(data[k])
    assert pred[1] == 5


def test_digit_correct_6():
    data, target = decisioncommon()
    l = np.where(target == 6)
    k = l[0]
    clf = load("model1svm.joblib")
    pred = clf.predict(data[k])
    assert pred[0] == 6


def test_digit_correct_7():
    data, target = decisioncommon()
    l = np.where(target == 7)
    k = l[0]
    clf = load("model1svm.joblib")
    pred = clf.predict(data[k])

    assert pred[0] == 7


def test_digit_correct_8():
    data, target = decisioncommon()
    l = np.where(target == 8)
    k = l[0]
    clf = load("model1svm.joblib")
    pred = clf.predict(data[k])
    assert pred[0] == 8


def test_digit_correct_9():
    data, target = decisioncommon()
    l = np.where(target == 9)
    k = l[0]
    clf = load("model1svm.joblib")
    pred = clf.predict(data[k])
    assert pred[0] == 9


def test_digit_correct_0dt():
    data, target = decisioncommon()
    l = np.where(target == 0)
    k = l[0]
    clf = load("modeldecisiontree.joblib")
    pred = clf.predict(data[k])
    assert pred[0] == 0


def test_digit_correct_1dt():
    data, target = decisioncommon()
    l = np.where(target == 1)
    k = l[0]
    clf = load("modeldecisiontree.joblib")
    pred = clf.predict(data[k])
    assert pred[0] == 1


def test_digit_correct_2dt():
    data, target = decisioncommon()
    l = np.where(target == 2)
    k = l[0]
    clf = load("modeldecisiontree.joblib")
    pred = clf.predict(data[k])
    assert pred[0] == 2


def test_digit_correct_3dt():
    data, target = decisioncommon()
    l = np.where(target == 3)
    k = l[0]
    clf = load("modeldecisiontree.joblib")
    pred = clf.predict(data[k])
    assert pred[0] == 3


def test_digit_correct_4dt():
    data, target = decisioncommon()
    l = np.where(target == 4)
    k = l[0]
    clf = load("modeldecisiontree.joblib")
    pred = clf.predict(data[k])
    assert pred[0] == 4


def test_digit_correct_5dt():
    data, target = decisioncommon()
    l = np.where(target == 5)
    k = l[0]
    clf = load("modeldecisiontree.joblib")
    pred = clf.predict(data[k])
    assert pred[1] == 5


def test_digit_correct_6dt():
    data, target = decisioncommon()
    l = np.where(target == 6)
    k = l[0]
    clf = load("modeldecisiontree.joblib")
    pred = clf.predict(data[k])
    assert pred[0] == 6


def test_digit_correct_7dt():
    data, target = decisioncommon()
    l = np.where(target == 7)
    k = l[0]
    clf = load("modeldecisiontree.joblib")
    pred = clf.predict(data[k])
    assert pred[0] == 7


def test_digit_correct_8dt():
    data, target = decisioncommon()
    l = np.where(target == 8)
    k = l[0]
    clf = load("modeldecisiontree.joblib")
    pred = clf.predict(data[k])
    assert pred[0] == 8


def test_digit_correct_9dt():
    data, target = decisioncommon()
    l = np.where(target == 9)
    k = l[0]
    clf = load("modeldecisiontree.joblib")
    pred = clf.predict(data[k])
    assert pred[0] == 9
