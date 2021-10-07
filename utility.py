import os
import numpy as np
from numpy.lib.function_base import average
from joblib import dump, load
from sklearn import metrics

def test_model(best_model_path,x,y):
    clf=load(best_model_path)
    metrics=test(clf,x,y)
    return metrics

def test(clf,x,y):
    predicted=clf.predict(x)
    acc=metrics.accuracy_score(y_pred=predicted,y_true=y)
    f1=metrics.f1_score(y_pred=predicted,y_true=y,average="macro")
    return {"accuracy":acc, "f1": f1}

def get_random_acc(y):
    return max(np.bincount(y))/len(y)

def run_classification_experiement(clf,x_train,y_train,x_valid,y_valid,gamma,output_model_file,skip_dummy=True):
    random_val_acc=get_random_acc(y_valid)
    #create a classifer : support vector machine
    #clf=classifier(gamma=gamma)
    #learn the digits on the train subset
    clf.fit(x_train,y_train)
    # predict the value of the digits on the validaiton subset
    metrics_valid=test(clf,x_valid,y_valid)
    # we will ensure to throw away some of the models that yield random like performance
    if skip_dummy and metrics_valid["accuracy"]<random_val_acc:
        print("skipping for {}".format(gamma))
        return None
    output_folder=os.path.dirname(output_model_file)
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    dump(clf,output_model_file)
    return metrics_valid