import numpy as np
from DataLoader import DataLoader

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score


# FOUR ALGORITHMS TO COMPARE
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


data_path = "data/credit.data"

dataset, labels = DataLoader.load_credit_with_onehot(data_path)

scaler = MinMaxScaler()

outer_cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=5)
inner_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5)

# SVM
svm = SVC()
cfg_svm = {
    "svm__kernel": (
        "linear",
        "rbf",
    ),
    "svm__C": [0.1, 1, 10],
}
pipe_svm = Pipeline([("scaler", scaler), ("svm", svm)])
clf_svm = GridSearchCV(pipe_svm, param_grid=cfg_svm, cv=inner_cv, scoring="accuracy")
# TODO do cross validation outer

# KNN
knn = KNeighborsClassifier()
cfg_knn = {
    "knn__n_neighbors": [3, 7],
    "knn__weights": ["uniform", "distance"],
}
pipe_knn = Pipeline([("scaler", scaler), ("knn", knn)])
clf_knn = GridSearchCV(pipe_knn, param_grid=cfg_knn, cv=inner_cv, scoring="accuracy")
# TODO do cross validation outer

# Random Forest
rf = RandomForestClassifier()
cfg_rf = {
    "rf__n_estimators": [20, 100],
    "rf__criterion": ["gini", "entropy"],
}
pipe_rf = Pipeline([("scaler", scaler), ("rf", rf)])
clf_rf = GridSearchCV(pipe_rf, param_grid=cfg_rf, cv=inner_cv, scoring="accuracy")
# TODO do cross validation outer

# Decision Tree
dt = DecisionTreeClassifier()
cfg_dt = {
    "dt__max_depth": [10, 20],
    "dt__min_samples_split": [5, 10],
}
pipe_dt = Pipeline([("scaler", scaler), ("dt", dt)])
clf_dt = GridSearchCV(pipe_dt, param_grid=cfg_dt, cv=inner_cv, scoring="accuracy")
# TODO do cross validation outer
