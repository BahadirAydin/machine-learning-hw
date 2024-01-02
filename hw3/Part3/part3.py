import numpy as np

np.set_printoptions(precision=3, threshold=10000)

from DataLoader import DataLoader

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV

# FOUR ALGORITHMS TO COMPARE
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

data_path = "data/credit.data"

dataset, labels = DataLoader.load_credit_with_onehot(data_path)

scaler = MinMaxScaler()

# random_state is used to ensure that the same splits are generated each time the evaluate_model() function is executed
outer_cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=5, random_state=1)


def evaluate_model_accuracy(model, cfg, rs):
    # random state on inner cv is used so for different metrics (accuracy and f1) we run on the same splits
    inner_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=rs)
    pipe = Pipeline([("scaler", scaler), ("model", model)])
    clf = GridSearchCV(
        pipe,
        param_grid=cfg,
        cv=inner_cv,
        n_jobs=-1,
        scoring="accuracy",
    )
    results = {}
    test_results = []

    for train_index, test_index in outer_cv.split(dataset, labels):
        X_train, X_test = dataset[train_index], dataset[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        clf.fit(X_train, y_train)
        cv_results = clf.cv_results_
        for k in cv_results["params"]:
            score = cv_results["mean_test_score"][cv_results["params"].index(k)]
            params = frozenset(k.items())
            if params not in results.keys():
                results[params] = []
            results[params].append(score)
        s = clf.score(X_test, y_test)
        test_results.append(s)
    mean = np.mean(test_results)
    std = np.std(test_results)
    confidence_interval = 1.96 * std / np.sqrt(len(test_results))

    return results, (
        round(mean - confidence_interval, 3),
        round(mean + confidence_interval, 3),
    )


def evaluate_model_f1(model, cfg, rs):
    # random state on inner cv is used so for different metrics (accuracy and f1) we run on the same splits
    inner_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=rs)
    pipe = Pipeline([("scaler", scaler), ("model", model)])
    clf = GridSearchCV(
        pipe,
        param_grid=cfg,
        cv=inner_cv,
        n_jobs=-1,
        scoring="f1",
    )
    results = {}
    test_results = []

    for train_index, test_index in outer_cv.split(dataset, labels):
        X_train, X_test = dataset[train_index], dataset[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        clf.fit(X_train, y_train)
        cv_results = clf.cv_results_
        for k in cv_results["params"]:
            score = cv_results["mean_test_score"][cv_results["params"].index(k)]
            params = frozenset(k.items())
            if params not in results.keys():
                results[params] = []
            results[params].append(score)
        s = clf.score(X_test, y_test)
        test_results.append(s)
    mean = np.mean(test_results)
    std = np.std(test_results)
    confidence_interval = 1.96 * std / np.sqrt(len(test_results))

    return results, (
        round(mean - confidence_interval, 3),
        round(mean + confidence_interval, 3),
    )


# SVM
svm = SVC()
cfg_svm = {
    "model__kernel": (
        "linear",
        "rbf",
    ),
    "model__C": [1, 10],
}
svm_results_acc, test_results1 = evaluate_model_accuracy(svm, cfg_svm, 1)
svm_results_f1, test_results2 = evaluate_model_f1(svm, cfg_svm, 1)


# KNN
knn = KNeighborsClassifier()
cfg_knn = {
    "model__n_neighbors": [3, 7],
    "model__weights": ["uniform", "distance"],
}
knn_results_acc, test_results3 = evaluate_model_accuracy(knn, cfg_knn, 2)
knn_results_f1, test_results4 = evaluate_model_f1(knn, cfg_knn, 2)

# Random Forest
rf = RandomForestClassifier()
cfg_rf = {
    "model__n_estimators": [20, 100],
    "model__criterion": ["gini", "entropy"],
}
rf_results_acc, test_results5 = evaluate_model_accuracy(rf, cfg_rf, 3)
rf_results_f1, test_results6 = evaluate_model_f1(rf, cfg_rf, 3)

# Decision Tree
dt = DecisionTreeClassifier()
cfg_dt = {
    "model__max_depth": [10, 20],
    "model__min_samples_split": [5, 10],
}
dt_results_acc, test_results7 = evaluate_model_accuracy(dt, cfg_dt, 4)
dt_results_f1, test_results8 = evaluate_model_f1(dt, cfg_dt, 4)


def calculate_confidence_interval(result):
    for k in result.keys():
        mean = np.mean(result[k])
        std = np.std(result[k])
        confidence_interval = 1.96 * std / np.sqrt(len(result[k]))
        result[k] = (
            round(mean - confidence_interval, 3),
            round(mean + confidence_interval, 3),
        )
    return result


svm_results_acc = calculate_confidence_interval(svm_results_acc)
knn_results_acc = calculate_confidence_interval(knn_results_acc)
rf_results_acc = calculate_confidence_interval(rf_results_acc)
dt_results_acc = calculate_confidence_interval(dt_results_acc)

svm_results_f1 = calculate_confidence_interval(svm_results_f1)
knn_results_f1 = calculate_confidence_interval(knn_results_f1)
rf_results_f1 = calculate_confidence_interval(rf_results_f1)
dt_results_f1 = calculate_confidence_interval(dt_results_f1)

print("SVM accuracy:\n ", svm_results_acc)
print("SVM f1:\n ", svm_results_f1)
print("KNN accuracy:\n ", knn_results_acc)
print("KNN f1:\n ", knn_results_f1)
print("RF accuracy:\n ", rf_results_acc)
print("RF f1:\n ", rf_results_f1)
print("DT accuracy:\n ", dt_results_acc)
print("DT f1:\n ", dt_results_f1)


print("Evaluation on test set")
print("SVM accuracy: ", test_results1)
print("SVM f1: ", test_results2)
print("KNN accuracy: ", test_results3)
print("KNN f1: ", test_results4)
print("RF accuracy: ", test_results5)
print("RF f1: ", test_results6)
print("DT accuracy: ", test_results7)
print("DT f1: ", test_results8)

# TRAIN DECISION TREE WITHOUT ONE-HOT ENCODING

dataset_DT, labels_DT = DataLoader.load_credit(data_path)
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(dataset_DT, labels_DT)
print("Feature importance: ", dt.feature_importances_)

# TRAIN SVC AND PROVIDE SUPPORT VECTORS (ONE-HOT ENCODING)
# EXTRACT FEATURES FROM SUPPORT VECTORS

svm = SVC(kernel="linear")
svm.fit(dataset, labels)
print("Save support vectors to file")
np.savetxt("support_vectors.csv", svm.support_vectors_, delimiter=",")
print("Number of support vectors for each class:\n ", svm.n_support_)

print("*" * 50)
