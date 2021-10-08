# Standard scientific Python imports
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets, svm, metrics
from sklearn.metrics import  accuracy_score
from sklearn.metrics import f1_score

import os
import numpy as np
from utility import run_classification_experiement

digits = datasets.load_digits()
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Training: %i' % label)
# def test_rand_acc_balanced():
#     y=np.array([1,1,2,2,3,3])
#     assert get_random_acc(y)==1.0/3.0

# def test_rand_acc_imbalanced():
#     y=np.array([1,3,3,3,3])
#     assert get_random_acc(y)==0.8

# def test_model_writing():
#     # flatten the images
#     n_samples = len(digits.images)
#     data = digits.images.reshape((n_samples, -1))
#     gamma_vals=0.001
#     # Split data into 50% train and 50% test subsets
#     X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.3, shuffle=False)
#     x_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, shuffle=False)
#     clf=svm.SVC(gamma=gamma_vals)
#     curr = "models/"
#     output_model_file = curr+ "_"+str(gamma_vals)+".joblib"
#     run_classification_experiement(
#         clf,X_train,y_train,x_valid,y_valid,gamma_vals,output_model_file)
#     assert os.path.isfile(output_model_file)

# def test_small_data_overfit_checking():
#     # flatten the images
#     n_samples = len(digits.images)
#     data = digits.images.reshape((n_samples, -1))
#     label=digits.target[:int(0.5*len(data))]
#     data=data[:int(0.5*len(data))]
#     gamma_vals=0.001
#     # Split data into 50% train and 50% test subsets
#     X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.3, shuffle=False)
#     x_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, shuffle=False)
#     clf=svm.SVC(gamma=gamma_vals)
#     curr = "models/"
#     output_model_file = curr+ "_"+str(gamma_vals)+".joblib"
#     output_matrix=run_classification_experiement(
#         clf,X_train,y_train,x_valid,y_valid,gamma_vals,output_model_file)
#     assert output_matrix["accuracy"]>0.8
#     assert output_matrix["f1"]>0.8 

def test_split():
    # flatten the images
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    gamma_vals=0.001
    # Split data into 70% train and 30% test subsets
    X_train, X_test, y_train, y_test = train_test_split(data[:100], digits.target[:100], test_size=0.3, shuffle=False)
    x_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.66, shuffle=False)
    assert X_train.shape[0] == 70
    assert X_test.shape[0] == 20
    assert x_valid.shape[0] == 10
    assert X_train.shape[0] + X_test.shape[0] + x_valid.shape[0] == 100
