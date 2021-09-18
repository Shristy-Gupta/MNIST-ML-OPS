
# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import  accuracy_score
from joblib import dump,load
import os
import argparse

###############################################################################
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

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Training: %i' % label)

###############################################################################
# Classification
# --------------
#
# To apply a classifier on this data, we need to flatten the images, turning
# each 2-D array of grayscale values from shape ``(8, 8)`` into shape
# ``(64,)``. Subsequently, the entire dataset will be of shape
# ``(n_samples, n_features)``, where ``n_samples`` is the number of images and
# ``n_features`` is the total number of pixels in each image.
#
# We can then split the data into train and test subsets and fit a support
# vector classifier on the train samples. The fitted classifier can
# subsequently be used to predict the value of the digit for the samples
# in the test subset.

# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Create a classifier: a support vector classifier
gamma=[0.001,0.002,0.003,0.01]
# Split data into 50% train and 50% test subsets
X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.3, shuffle=False)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, shuffle=False)
def train_for_gamma_vals(gamma_vals,x_train,y_train):
    clf=svm.SVC(gamma=gamma_vals)
    clf.fit(x_train, y_train)
    return clf
def calculate_f1score_and_accuracy(clf,x1_val,y1_val):
    predicted = clf.predict(x1_val)
    f1score=f1_score(y1_val,predicted,average='micro')
    accuracyscore=accuracy_score(y1_val,predicted)
    return f1score,accuracyscore
finalans=0
finalgamma = 0
finalModel=None
accuracy_array=[]
f1_score_array=[]
curr = os.getcwd()
for gamma_vals in gamma:
    clf = train_for_gamma_vals(gamma_vals,X_train,y_train)
    # Predict the value of the digit on the test subset
    f1score,accuracyscore=calculate_f1score_and_accuracy(clf,X_val,y_val)
    f1_score_array.append(f1score)
    accuracy_array.append(accuracyscore)
    path = curr+'/models/model_{}.joblib'.format(gamma_vals)
    dump(finalModel,path)
    if accuracyscore>finalans:
        finalgamma = gamma_vals
        finalans=accuracyscore
        finalModel=clf
    print(gamma_vals,accuracyscore,f1score)

print('final accuracy score of best model',finalans)
testpred = finalModel.predict(X_test)
accuracyscore=accuracy_score(y_test, testpred)
print("Best performing model has gamma = {} and a test accuracy of {}".format(finalgamma, accuracyscore))
    ###############################################################################
