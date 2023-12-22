import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

configurations = [
    {"C": 0.5, "kernel": "rbf"},
    {"C": 1, "kernel": "rbf"},
    {"C": 2, "kernel": "rbf"},
    {"C": 0.5, "kernel": "linear"},
    {"C": 1, "kernel": "linear"},
    {"C": 2, "kernel": "linear"},
]

dataset, labels = pickle.load(open("../data/part2_dataset2.data", "rb"))

svm = SVC()
clf = GridSearchCV(svm, configurations, scoring="accuracy")
X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2)

scaler = StandardScaler()
