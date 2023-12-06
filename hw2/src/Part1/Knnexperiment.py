import pickle
from Distance import Distance
from Knn import KNN
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


dataset, labels = pickle.load(open("../data/part1_dataset.data", "rb"))
knn = KNN(dataset, labels, Distance.calculateMinkowskiDistance, 4)

num_folds = 10
iteration_num = 5
config = {
    "k": [3, 5, 10, 30],
    "distance_function": [
        Distance.calculateCosineDistance,
        Distance.calculateMahalanobisDistance,
        Distance.calculateMinkowskiDistance,
    ],
}


def calculate_confidence_interval(accuracies):
    mean = np.mean(accuracies)
    std = np.std(accuracies)
    confidence_interval = 1.96 * std / np.sqrt(len(accuracies))
    return (round(mean - confidence_interval,3),round( mean + confidence_interval,3))


best_accuracy = 0
best_model = None

shuffled_indices = np.random.permutation(len(dataset))
dataset = dataset[shuffled_indices]
labels = labels[shuffled_indices]

train_data, test_data = (
    dataset[: int(len(dataset) * 0.8)],
    dataset[int(len(dataset) * 0.8) :],
)
train_labels, test_labels = (
    labels[: int(len(labels) * 0.8)],
    labels[int(len(labels) * 0.8) :],
)

for k in config["k"]:
    for distance_function in config["distance_function"]:
        accuracies = []
        model = {"k": k, "distance_function": distance_function}

        for _ in range(iteration_num):
            skf = StratifiedKFold(n_splits=num_folds, shuffle=True)
            for train_index, test_index in skf.split(train_data, train_labels):
                X_train, X_test = (
                    train_data[train_index],
                    train_data[test_index],
                )
                Y_train, Y_test = (
                    train_labels[train_index],
                    train_labels[test_index],
                )

                knn = KNN(X_train, Y_train, distance_function, K=k)
                predictions = [knn.predict(i) for i in X_test]
                accuracy = accuracy_score(Y_test, predictions)
                accuracies.append(accuracy)

        print(f"K: {k}, Distance Function: {distance_function.__name__}")
        print(f"Accuracy: {np.mean(accuracies)}")
        print(f"Confidence Interval: {calculate_confidence_interval(accuracies)}")

        if np.mean(accuracies) > best_accuracy:
            best_accuracy = np.mean(accuracies)
            best_model = model


if best_model is None:
    print("No model found")
    exit()

print(f"Best Accuracy: {best_accuracy}")
print(f"Best Model: {best_model}")

print("Testing the generalization of the best model")
knn = KNN(
    test_data, test_labels, best_model["distance_function"], K=best_model["k"]
)
predictions = [knn.predict(i) for i in test_data]
accuracy = accuracy_score(test_labels, predictions)


print("Final Accuracy: ", accuracy)
