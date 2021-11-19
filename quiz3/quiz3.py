from sklearn import datasets
import numpy as np
from joblib import dump, load


def decisioncommon():
    digits = datasets.load_digits()
    data = digits.images
    target = digits.target
    data = data.reshape((data.shape[0], -1))
    return data, target


def test_digit_correct_0():
    data, target = decisioncommon()
    target = np.where(target == 0)
    k = target[0]
    clf = load("model1svm.joblib")
    pred = clf.predict(data[k])
    assert pred[0] == 0


def test_digit_correct_1():
    data, target = decisioncommon()
    target = np.where(target == 1)
    k = target[0]
    clf = load("model1svm.joblib")
    pred = clf.predict(data[k])
    assert pred[0] == 1


def test_digit_correct_2():
    data, target = decisioncommon()
    target = np.where(target == 2)
    k = target[0]
    clf = load("model1svm.joblib")
    pred = clf.predict(data[k])
    assert pred[0] == 2


def test_digit_correct_3():
    data, target = decisioncommon()
    target = np.where(target == 3)
    k = target[0]
    clf = load("model1svm.joblib")
    pred = clf.predict(data[k])
    assert pred[0] == 3


def test_digit_correct_4():
    data, target = decisioncommon()
    target = np.where(target == 4)
    k = target[0]
    clf = load("model1svm.joblib")
    pred = clf.predict(data[k])
    assert pred[0] == 4


def test_digit_correct_5():
    data, target = decisioncommon()
    target = np.where(target == 5)
    k = target[0]
    clf = load("model1svm.joblib")
    pred = clf.predict(data[k])
    assert pred[1] == 5


def test_digit_correct_6():
    data, target = decisioncommon()
    target = np.where(target == 6)
    k = target[0]
    clf = load("model1svm.joblib")
    pred = clf.predict(data[k])
    assert pred[0] == 6


def test_digit_correct_7():
    data, target = decisioncommon()
    target = np.where(target == 7)
    k = target[0]
    clf = load("model1svm.joblib")
    pred = clf.predict(data[k])

    assert pred[0] == 7


def test_digit_correct_8():
    data, target = decisioncommon()
    target = np.where(target == 8)
    k = target[0]
    clf = load("model1svm.joblib")
    pred = clf.predict(data[k])
    assert pred[0] == 8


def test_digit_correct_9():
    data, target = decisioncommon()
    target = np.where(target == 9)
    k = target[0]
    clf = load("model1svm.joblib")
    pred = clf.predict(data[k])
    assert pred[0] == 9


def test_digit_correct_0_dt():
    data, target = decisioncommon()
    target = np.where(target == 0)
    k = target[0]
    clf = load("modeldecisiontree.joblib")
    pred = clf.predict(data[k])
    assert pred[0] == 0


def test_digit_correct_1_dt():
    data, target = decisioncommon()
    target = np.where(target == 1)
    k = target[0]
    clf = load("modeldecisiontree.joblib")
    pred = clf.predict(data[k])
    assert pred[0] == 1


def test_digit_correct_2_dt():
    data, target = decisioncommon()
    target = np.where(target == 2)
    k = target[0]
    clf = load("modeldecisiontree.joblib")
    pred = clf.predict(data[k])
    assert pred[0] == 2


def test_digit_correct_3_dt():
    data, target = decisioncommon()
    target = np.where(target == 3)
    k = target[0]
    clf = load("modeldecisiontree.joblib")
    pred = clf.predict(data[k])
    assert pred[0] == 3


def test_digit_correct_4_dt():
    data, target = decisioncommon()
    target = np.where(target == 4)
    k = target[0]
    clf = load("modeldecisiontree.joblib")
    pred = clf.predict(data[k])
    assert pred[0] == 4


def test_digit_correct_5_dt():
    data, target = decisioncommon()
    target = np.where(target == 5)
    k = target[0]
    clf = load("modeldecisiontree.joblib")
    pred = clf.predict(data[k])
    assert pred[1] == 5


def test_digit_correct_6_dt():
    data, target = decisioncommon()
    target = np.where(target == 6)
    k = target[0]
    clf = load("modeldecisiontree.joblib")
    pred = clf.predict(data[k])
    assert pred[0] == 6


def test_digit_correct_7_dt():
    data, target = decisioncommon()
    target = np.where(target == 7)
    k = target[0]
    clf = load("modeldecisiontree.joblib")
    pred = clf.predict(data[k])
    assert pred[0] == 7


def test_digit_correct_8_dt():
    data, target = decisioncommon()
    target = np.where(target == 8)
    k = target[0]
    clf = load("modeldecisiontree.joblib")
    pred = clf.predict(data[k])
    assert pred[0] == 8


def test_digit_correct_9_dt():
    data, target = decisioncommon()
    target = np.where(target == 9)
    k = target[0]
    clf = load("modeldecisiontree.joblib")
    pred = clf.predict(data[k])
    assert pred[0] == 9
