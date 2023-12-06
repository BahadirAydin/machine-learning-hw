import numpy as np


class KNN:
    def __init__(
        self,
        dataset,
        data_label,
        similarity_function,
        similarity_function_parameters=None,
        K=1,
    ):
        self.K = K
        self.dataset = dataset
        self.dataset_label = data_label
        self.similarity_function = similarity_function
        self.similarity_function_parameters = similarity_function_parameters

        if self.similarity_function.__name__ == "calculateMahalanobisDistance":
            self.similarity_function_parameters = np.linalg.inv(
                np.cov(self.dataset, rowvar=False)
            )

    def predict(self, instance):
        distances = []
        for d in self.dataset:
            if self.similarity_function_parameters is None:
                distances.append(self.similarity_function(instance, d))
            else:
                distances.append(
                    self.similarity_function(
                        instance, d, self.similarity_function_parameters
                    )
                )
        sorted_distances = np.argsort(distances)
        k_nearest = sorted_distances[: self.K]
        k_labels = [self.dataset_label[i] for i in k_nearest]
        # i wanted to do this old way without using np functionaliites probably it would be a one-liner if i would use np
        unique = {}
        for l in k_labels:
            if l in unique:
                unique[l] += 1
            else:
                unique[l] = 1

        return max(unique, key=unique.get)
