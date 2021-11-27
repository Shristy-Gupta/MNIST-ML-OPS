import matplotlib.pyplot as plt
import numpy as np
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics, tree
from sklearn.model_selection import train_test_split
#import warnings
#from sklearn.exceptions import DataConversionWarning
#from skimage.transform import rescale, resize, downscale_local_mean
import pandas as pd

#############################################################################
# Digits dataset
# --------------
#
# The digits dataset consists of 8x8
# pixel images of digits. The ``images`` attribute of the dataset stores
# 8x8 arrays of grayscale values for each image. We will use these arrays to
# visualize the first 4 images. The ``target`` attribute of the dataset stores
# the digit each image represents and this is included in the title of the 4
# plots below.
#
# Note: if we were working from image files (e.g., 'png' files), we would load
# them using :func:`matplotlib.pyplot.imread`.
digits = datasets.load_digits()
# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
# Create a classifier: a support vector classifier
gamma = [1e-7, 1e-5, 1e-3, 0.01, 0.02, 0.1]
split = [0.15]
optimal_SVM = []
optimal_gamma = []
optimal_f1 = []
underperforming_gamma = []


def calculate_accuracy(gamma_vals):
   # 70:15:15 train:test:dev
    clf = svm.SVC(gamma=gamma_vals)
    X_train, X_test_valid, y_train, y_test_valid = train_test_split(
        data, digits.target, test_size=0.30, shuffle=False
    )

    X_test, X_valid, y_test, y_valid = train_test_split(
        X_test_valid, y_test_valid, test_size=0.5, shuffle=False,
    )
    clf.fit(X_train, y_train)
    predicted_valid = clf.predict(X_valid)
    valid_accuracy = metrics.accuracy_score(
        y_pred=predicted_valid, y_true=y_valid)
    valid_f1 = metrics.f1_score(
        y_pred=predicted_valid, y_true=y_valid, average="macro"
    )

    return valid_accuracy, valid_f1


for i in range(3):
    accuracy_array = []
    f1score_array = []
    for j in gamma:
        acc, f1 = calculate_accuracy(j)
        accuracy_array.append(acc)
        f1score_array.append(f1)
    max_acc_index = accuracy_array.index(max(accuracy_array))
    max_acc = max(accuracy_array)
    best_gamma = gamma[max_acc_index]
    min_acc_index = accuracy_array.index(min(accuracy_array))
    print(gamma[min_acc_index])
    underperforming_gamma.append(gamma[min_acc_index])
    optimal_gamma.append(best_gamma)
    optimal_SVM.append(max_acc)
    optimal_f1.append(max(f1score_array))


data = {"Run": [1, 2, 3],
        "gamma": optimal_gamma, "beta": optimal_f1, "Underperforming_gamma": underperforming_gamma, "Accuracy_score": optimal_SVM}


df = pd.DataFrame(data)

mean_svm = str(df["Accuracy_score"].mean())


std_svm = str(round(df["Accuracy_score"].std(), 4))
plusminus_symbol = "\u00B1"


ms = mean_svm+plusminus_symbol+std_svm
pd.options.display.float_format = '{:.2f}'.format
s = pd.Series([' ', ' ', ' ', ' '.format, ms], index=[
              'Run', 'gamma', 'beta', 'Underperforming_gamma', 'Accuracy_score'])
df = df.append(s, ignore_index=True)
print(df)
