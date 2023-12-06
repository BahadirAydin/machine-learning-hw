import numpy as np
import math


class Distance:
    @staticmethod
    def calculateCosineDistance(x, y):
        return 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

    @staticmethod
    def calculateMinkowskiDistance(x, y, p=2):
        dim = len(x)
        sum = 0
        for i in range(dim):
            sum += abs(x[i] - y[i]) ** p
        return sum ** (1 / p)

    @staticmethod
    def calculateMahalanobisDistance(x, y, S_minus_1):
        a = np.transpose(x - y)
        b = S_minus_1
        c = x - y
        return np.sqrt(np.dot(np.dot(a, b), c))
