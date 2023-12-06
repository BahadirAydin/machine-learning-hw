from Distance import Distance
import random
import numpy as np


class KMeans:
    def __init__(self, dataset, K=2):
        """
        :param dataset: 2D numpy array, the whole dataset to be clustered
        :param K: integer, the number of clusters to form
        """
        self.K = K
        self.dataset = dataset
        # each cluster is represented with an integer index
        # self.clusters stores the data points of each cluster in a dictionary
        self.clusters = {i: [] for i in range(K)}
        # self.cluster_centers stores the cluster mean vectors for each cluster in a dictionary
        self.cluster_centers = {i: None for i in range(K)}
        # you are free to add further variables and functions to the class

    def calculateLoss(self):
        """Loss function implementation of Equation 1"""
        # n is number of data samples, K is number of clusters
        # x_s is the s-th data sample, c_k is the k-th cluster mean vector
        # l is 1 if an instance s belongs to cluster k, otherwise 0
        # sum from s to n (sum from k to K (l * distance_function(x_s,c_k)))
        loss = 0
        for s in range(len(self.dataset)):
            for k in range(self.K):
                if s in self.clusters[k]:
                    loss += Distance.kMeansDistance(
                        self.dataset[s], self.cluster_centers[k]
                    )
        return loss

    def initialize_centers(self):
        """Initializes the cluster centers"""
        # randomly select K data points as cluster centers
        random_points = []
        for k in range(self.K):
            x = random.choice(self.dataset)
            p = random.choice(x)
            while p in random_points:
                x = random.choice(self.dataset)
                p = random.choice(x)

            self.cluster_centers[k] = p
            random_points.append(self.cluster_centers[k])

    def update_clusters(self, X):
        predictions = []
        for x in X:
            distances = []
            distances = [
                Distance.kMeansDistance(x, self.cluster_centers[k])
                for k in range(self.K)
            ]
            predictions.append(np.argmin(distances))
        return np.array(predictions)

    def update_centers(self, X, Y):
        for k in range(self.K):
            if len(X[Y == k]) > 0:
                self.cluster_centers[k] = np.mean(X[Y == k], axis=0)

    def run(self):
        """Kmeans algorithm implementation"""

        prev, curr = None, np.zeros(len(self.dataset))
        self.initialize_centers()
        while not np.all(prev == curr):
            prev = curr
            curr = self.update_clusters(self.dataset)
            self.update_centers(self.dataset, curr)

        self.clusters = {i: np.where(curr == i)[0] for i in range(self.K)}

        return self.cluster_centers, self.clusters, self.calculateLoss()
