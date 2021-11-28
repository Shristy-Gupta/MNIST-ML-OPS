import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

clf = svm.SVC(gamma=0.001)

X_train, X_test_valid, y_train, y_test_valid = train_test_split(
    data, digits.target, test_size=0.2, shuffle=False
)

X_test, X_valid, y_test, y_valid = train_test_split(X_test_valid,
                                                    y_test_valid, test_size=0.5, shuffle=False,
                                                    )

trainsize = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
testsize = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
accuracylist = []
for value in testsize:
    X_train1, X_test_valid1, y_train1, y_test_valid1 = train_test_split(
        X_train, y_train, test_size=value, shuffle=False
    )
    clf.fit(X_train1, y_train1)
    predicted_valid = clf.predict(X_test)
    f1_valid = metrics.f1_score(
        y_pred=predicted_valid, y_true=y_test, average="macro"
    )
    accuracylist.append(f1_valid)


plt.plot(trainsize, accuracylist)
plt.title('Model Accuracy')
plt.xlabel('Train Data Size')
plt.ylabel('Accuracy')
plt.show()


print("Comparison with 20 and 30 percent training data")
trainsize = [0.8, 0.7]
for value in trainsize:
    X_train1, X_test_valid1, y_train1, y_test_valid1 = train_test_split(
        X_train, y_train, test_size=value, shuffle=False
    )
    clf.fit(X_train1, y_train1)
    predicted_valid = clf.predict(X_test)
    print(f'Confusion matrix of {1-value} training data')
    disp = metrics.plot_confusion_matrix(clf, X_test, y_test)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")
