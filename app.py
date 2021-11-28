import pickle
from flask import Flask
from flask import request
from joblib import dump, load
import numpy as np


app = Flask(__name__)

bestmodeltree = "modeldecisiontree.joblib"
bestmodelsvm = "model1svm.joblib"


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/svm_predict", methods=['POST'])
def svm_predict():
    inputjason = request.json
    image = inputjason['image']
    image = np.array(image).reshape(1, -1)
    file = open(bestmodelsvm, "rb")
    clf = load(file)
    predicted = clf.predict(image)
    return str(predicted[0])


@app.route("/decision_tree_predict", methods=['POST'])
def decision_tree_predict():
    inputjason = request.json
    image = inputjason['image']
    image = np.array(image).reshape(1, -1)
    file = open(bestmodeltree, "rb")
    clf = load(file)
    predicted = clf.predict(image)
    return str(predicted[0])
