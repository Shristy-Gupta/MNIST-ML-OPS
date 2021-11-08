from sklearn import datasets, svm, metrics

def train_for_gamma_vals(gamma_vals,x_train,y_train):
    clf=svm.SVC(gamma=gamma_vals)
    clf.fit(x_train, y_train)
    return clf