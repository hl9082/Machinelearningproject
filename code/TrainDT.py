"""
TrainDT.py

Author: Noah Shin

Given a numpy array of labels loaded from Y.npy and a numpy
array of encoded vectors loaded from X.npy, trains a decision
tree classifier and saves it as a .joblib file.
"""

import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from pathlib import Path
import joblib

dir = Path(__file__).resolve().parent

vectors = np.load(dir/"X.npy")
labels = np.load(dir/"Y.npy")

v_train, v_test, l_train, l_test = train_test_split(vectors, labels, test_size=0.2)

dt = DecisionTreeClassifier(max_depth=10)
dt.fit(v_train, l_train)
l_predicted = dt.predict(v_test)
print("Accuracy: " + str(accuracy_score(l_test, l_predicted)))

joblib.dump(dt, dir/"DT.joblib")


