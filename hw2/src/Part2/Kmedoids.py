from Distance import Distance
import numpy as np
import random


class KMemoids:
    def __init__(self, dataset, K=2, distance_metric="cosine"):
        """
        :param dataset: 2D numpy array, the whole dataset to be clustered
        :param K: integer, the number of clusters to form
        """
        self.K = K
        self.dataset = dataset
        self.distance_metric = distance_metric
        # each cluster is represented with an integer index
        # self.clusters stores the data points of each cluster in a dictionary
        # In this dictionary, you can keep either the data instance themselves or their corresponding indices in the dataset (self.dataset).
        self.clusters = {i: [] for i in range(K)}
        # self.cluster_medoids stores the cluster medoid for each cluster in a dictionary
        # # In this dictionary, you can keep either the data instance themselves or their corresponding indices in the dataset (self.dataset).
        self.cluster_medoids = {i: None for i in range(K)}
        self.distance_table = None
        self.pairwise_distances()
        # you are free to add further variables and functions to the class

    def calculateLoss(self):
        loss = 0
        for s in range(len(self.dataset)):
            for k in range(self.K):
                if s in self.clusters[k]:
                    loss += Distance.calculateCosineDistance(
                        self.dataset[s], self.dataset[self.cluster_medoids[k]]
                    )
        return loss

    def initialize_medoids(self):
        """Initializes the cluster centers"""
        # randomly select K data points as cluster centers
        self.cluster_medoids = np.random.choice(
            len(self.dataset), self.K, replace=False
        )

    def pairwise_distances(self):
        """Calculate pairwise distances between all data points and update self.distance_table"""
        num_samples = len(self.dataset)
        self.distance_table = np.zeros((num_samples, num_samples))

        for i in range(num_samples):
            for j in range(i, num_samples):
                distance_ij = Distance.calculateCosineDistance(
                    self.dataset[i], self.dataset[j]
                )
                # Update both symmetric entries in the distance_table
                self.distance_table[i, j] = distance_ij
                self.distance_table[j, i] = distance_ij

    def update_clusters(self, X):
        predictions = []
        for x in range(len(X)):
            distances = [self.distance_table[x, m] for m in self.cluster_medoids]
            predictions.append(np.argmin(distances))
        return np.array(predictions)

    def update_medoids(self, X, predictions):
        for k in range(self.K):
            medoid = X[predictions == k]
            medoid_indices = [
                np.where(np.all(self.dataset == m, axis=1))[0][0] for m in medoid
            ]
            sums = []
            if len(medoid) > 0:
                for i in range(len(medoid_indices)):
                    sum = 0
                    for j in range(len(medoid_indices)):
                        sum += self.distance_table[medoid_indices[i], medoid_indices[j]]
                    sums.append(sum)

                index = medoid_indices[np.argmin(sums)]
                # index elemenet is the medoid
                self.cluster_medoids[k] = index

    def run(self):
        """Kmedoids algorithm implementation"""
        prev_loss, curr_loss = 0, 1
        self.initialize_medoids()
        while prev_loss != curr_loss:
            prev_loss = curr_loss
            curr = self.update_clusters(self.dataset)
            self.update_medoids(self.dataset, curr)
            self.clusters = {i: np.where(curr == i)[0] for i in range(self.K)}
            curr_loss = self.calculateLoss()
        return self.cluster_medoids, self.clusters, self.calculateLoss()
