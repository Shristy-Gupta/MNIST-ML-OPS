# -*- coding: utf-8 -*-
print(__doc__)

import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from skimage import transform
import numpy as np

"""for image size 16"""

digits = datasets.load_digits()
new_features = np.array(list
                        (map
                         (lambda img: transform.resize(
                                        img.reshape(8,8),#old shape
                                          (16, 16), #new shape
                                          mode='constant',
                                         #flatten the resized image
                                          preserve_range=True).ravel(),
             digits.images)))

digits.images = new_features
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
clf = svm.SVC(gamma=0.001)

# different train-test set
testSize = [0.5,0.8,0.9]
for i in testSize:
  print("*********************************************")
  print(f'Displaying test size set of {i}')
  X_train, X_test, y_train, y_test = train_test_split(
      data, digits.target, test_size=i, shuffle=False)
  clf.fit(X_train, y_train)
  predicted = clf.predict(X_test)

  print(f"Classification report for classifier {clf}:\n"
      f"{metrics.classification_report(y_test, predicted)}\n")
digits = datasets.load_digits()

"""for image size 32"""

digits = datasets.load_digits()
new_features = np.array(list
                        (map
                         (lambda img: transform.resize(
                                        img.reshape(8,8),#old shape
                                          (32, 32), #new shape
                                          mode='constant',
                                         #flatten the resized image
                                          preserve_range=True).ravel(),
             digits.images)))

digits.images = new_features
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
clf = svm.SVC(gamma=0.001)

# different train-test set
testSize = [0.5,0.8,0.9]
for i in testSize:
  print(f'Displaying test size set of {i}')
  X_train, X_test, y_train, y_test = train_test_split(
      data, digits.target, test_size=i, shuffle=False)
  clf.fit(X_train, y_train)
  predicted = clf.predict(X_test)

  print(f"classifier results {clf}:\n"
      f"{metrics.classification_report(y_test, predicted)}\n")
digits = datasets.load_digits()

"""for image size 64"""

digits = datasets.load_digits()
new_features = np.array(list
                        (map
                         (lambda img: transform.resize(
                                        img.reshape(8,8),#old shape
                                          (32, 32), #new shape
                                          mode='constant',
                                         #flatten the resized image
                                          preserve_range=True).ravel(),
             digits.images)))

digits.images = new_features
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
clf = svm.SVC(gamma=0.001)

# different train-test set
testSize = [0.5,0.8,0.9]
for i in testSize:

  print(f'Displaying test size set of {i}')
  X_train, X_test, y_train, y_test = train_test_split(
      data, digits.target, test_size=i, shuffle=False)
  clf.fit(X_train, y_train)
  predicted = clf.predict(X_test)

  print(f"Larrifier results {clf}:\n"
      f"{metrics.classification_report(y_test, predicted)}\n")
digits = datasets.load_digits()