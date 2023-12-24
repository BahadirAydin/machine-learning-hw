import pickle
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

configurations = {
    "svm__kernel": (
        "linear",
        "rbf",
    ),
    "svm__C": [0.1, 1, 10],
}

dataset, labels = pickle.load(open("../data/part2_dataset2.data", "rb"))

scaler = StandardScaler()
svm = SVC()
pipeline = Pipeline([("scaler", scaler), ("svm", svm)])
f = StratifiedKFold(n_splits=10, shuffle=True)
clf = GridSearchCV(
    pipeline,
    param_grid=configurations,
    scoring="accuracy",
    cv=f,
    refit=True,
)

all_scores = [0] * len(configurations["svm__kernel"]) * len(configurations["svm__C"])
params = []
N = 5
for _ in range(N):
    clf.fit(dataset, labels)
    df = pd.DataFrame(clf.cv_results_)
    params = df["params"]
    scores = df["mean_test_score"]
    for j in range(len(params)):
        all_scores[j] += scores[j]

mean_scores = [x / N for x in all_scores]
for i in range(len(params)):
    print(params[i], mean_scores[i])
