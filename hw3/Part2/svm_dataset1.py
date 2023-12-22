import pickle
import numpy as np
from sklearn.svm import SVC
import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt


X, Y = pickle.load(open("../data/part2_dataset1.data", "rb"))

configurations = [
    {"C": 1, "kernel": "rbf"},
    {"C": 2, "kernel": "rbf"},
    {"C": 1, "kernel": "linear"},
    {"C": 2, "kernel": "linear"},
]
# default value of C is 1 in SVC
# default value of kernel is rbf in SVC

# PLOTTING CODE TAKEN FROM: https://scikit-learn.org/0.18/auto_examples/svm/plot_iris.html
# (with minor changes)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = 0.02
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

for i, config in enumerate(configurations):
    print("Running for ", config)
    clf = SVC(**config)
    clf.fit(X, Y)

    plt.subplot(2, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.coolwarm)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title("C:" + str(config["C"]) + " Kernel:" + config["kernel"])
    plt.tight_layout()

plt.savefig("dataset1_result.png", dpi=300)
