from sklearn.metrics import  accuracy_score
from sklearn.metrics import f1_score
def calculate_f1score_and_accuracy(clf,x1_val,y1_val):
    predicted = clf.predict(x1_val)
    f1score=f1_score(y1_val,predicted,average='micro')
    accuracyscore=accuracy_score(y1_val,predicted)
    return f1score,accuracyscore